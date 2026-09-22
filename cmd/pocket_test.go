package cmd

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"runtime"
	"strings"
	"testing"
	"testing/iotest"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// wavHeader builds a RIFF/WAVE header. dataSize is written verbatim, so a streamed-WAV
// placeholder can be reproduced; extra chunks are inserted between fmt and data.
func wavHeader(sampleRate, channels, bits int, dataSize uint32, extra ...[]byte) []byte {
	var b bytes.Buffer
	b.WriteString("RIFF")
	binary.Write(&b, binary.LittleEndian, uint32(36+dataSize))
	b.WriteString("WAVE")
	b.WriteString("fmt ")
	binary.Write(&b, binary.LittleEndian, uint32(16))
	binary.Write(&b, binary.LittleEndian, uint16(1))
	binary.Write(&b, binary.LittleEndian, uint16(channels))
	binary.Write(&b, binary.LittleEndian, uint32(sampleRate))
	binary.Write(&b, binary.LittleEndian, uint32(sampleRate*channels*bits/8))
	binary.Write(&b, binary.LittleEndian, uint16(channels*bits/8))
	binary.Write(&b, binary.LittleEndian, uint16(bits))
	for _, chunk := range extra {
		b.Write(chunk)
	}
	b.WriteString("data")
	binary.Write(&b, binary.LittleEndian, dataSize)
	return b.Bytes()
}

// pcm16 encodes samples as 16-bit little-endian PCM.
func pcm16(samples ...int16) []byte {
	var b bytes.Buffer
	for _, s := range samples {
		binary.Write(&b, binary.LittleEndian, s)
	}
	return b.Bytes()
}

// pocketServer fakes `pocket-tts serve`: it checks the request and streams the WAV back in
// several flushed pieces, the way the real server does.
func pocketServer(t *testing.T, wantVoice string, pcm []byte) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, http.MethodPost, r.Method)
		assert.Equal(t, "/tts", r.URL.Path)
		require.NoError(t, r.ParseForm())
		assert.Equal(t, "Ready for review", r.PostForm.Get("text"))
		if wantVoice == "" {
			_, has := r.PostForm["voice_url"]
			assert.False(t, has, "no voice_url should be sent when no voice is chosen")
		} else {
			assert.Equal(t, wantVoice, r.PostForm.Get("voice_url"))
		}

		w.Header().Set("Content-Type", "audio/wav")
		w.WriteHeader(http.StatusOK)
		flusher := w.(http.Flusher)
		body := append(wavHeader(24000, 1, 16, 2_000_000_000), pcm...)
		for len(body) > 0 {
			n := min(7, len(body)) // odd-sized pieces split samples across reads
			w.Write(body[:n])
			flusher.Flush()
			body = body[n:]
		}
	}))
}

func TestReadWAVHeader(t *testing.T) {
	t.Run("streamed pocket-tts header with placeholder length", func(t *testing.T) {
		r := bytes.NewReader(append(wavHeader(24000, 1, 16, 2_000_000_000), 0x01, 0x02))
		rate, err := readWAVHeader(r)
		require.NoError(t, err)
		assert.Equal(t, 24000, rate)
		rest, _ := io.ReadAll(r)
		assert.Equal(t, []byte{0x01, 0x02}, rest, "reader must stop exactly at the PCM data")
	})

	t.Run("skips chunks between fmt and data", func(t *testing.T) {
		list := append([]byte("LIST"), 3, 0, 0, 0, 'a', 'b', 'c', 0) // odd size plus pad byte
		r := bytes.NewReader(append(wavHeader(22050, 1, 16, 4, list), 0xAA))
		rate, err := readWAVHeader(r)
		require.NoError(t, err)
		assert.Equal(t, 22050, rate)
		rest, _ := io.ReadAll(r)
		assert.Equal(t, []byte{0xAA}, rest)
	})

	t.Run("works on a one-byte-at-a-time reader", func(t *testing.T) {
		rate, err := readWAVHeader(iotest.OneByteReader(bytes.NewReader(wavHeader(24000, 1, 16, 0))))
		require.NoError(t, err)
		assert.Equal(t, 24000, rate)
	})

	for name, tc := range map[string]struct {
		data []byte
		want string
	}{
		"not a wav":       {[]byte(`{"detail":"Text cannot be empty"}`), "not a RIFF/WAVE"},
		"stereo":          {wavHeader(24000, 2, 16, 0), "unsupported format"},
		"8-bit":           {wavHeader(24000, 1, 8, 0), "unsupported format"},
		"truncated":       {wavHeader(24000, 1, 16, 0)[:20], "fmt chunk"},
		"data before fmt": {append([]byte("RIFF\x00\x00\x00\x00WAVE"), []byte("data\x00\x00\x00\x00")...), "data chunk before fmt"},
	} {
		t.Run(name, func(t *testing.T) {
			_, err := readWAVHeader(bytes.NewReader(tc.data))
			require.Error(t, err)
			assert.Contains(t, err.Error(), tc.want)
		})
	}
}

func TestPCMReaderStream(t *testing.T) {
	t.Run("fills buffers across short reads and odd byte splits", func(t *testing.T) {
		data := pcm16(0, 16384, -32768, 32767, -16384)
		s := NewPCMReaderStream(iotest.OneByteReader(bytes.NewReader(data)))

		buf := make([][2]float64, 3)
		n, ok := s.Stream(buf)
		assert.True(t, ok)
		assert.Equal(t, 3, n)
		assert.Equal(t, [2]float64{0, 0}, buf[0])
		assert.Equal(t, [2]float64{0.5, 0.5}, buf[1])
		assert.Equal(t, [2]float64{-1, -1}, buf[2])

		n, ok = s.Stream(buf)
		assert.True(t, ok)
		assert.Equal(t, 2, n, "the tail is shorter than the buffer")
		assert.InDelta(t, 32767.0/32768.0, buf[0][0], 1e-9)
		assert.Equal(t, [2]float64{-0.5, -0.5}, buf[1])

		n, ok = s.Stream(buf)
		assert.False(t, ok)
		assert.Equal(t, 0, n)
		assert.NoError(t, s.Err(), "running out of data is not an error")
		assert.Equal(t, 5, s.Position())
	})

	t.Run("carries an odd byte over to the next call", func(t *testing.T) {
		// Two reads of 3 bytes each: the second sample straddles them.
		data := pcm16(100, 200, 300)
		s := NewPCMReaderStream(io.MultiReader(bytes.NewReader(data[:3]), bytes.NewReader(data[3:])))
		buf := make([][2]float64, 1)
		var got []float64
		for {
			n, ok := s.Stream(buf)
			if !ok {
				break
			}
			for i := 0; i < n; i++ {
				got = append(got, buf[i][0]*32768)
			}
		}
		assert.Equal(t, []float64{100, 200, 300}, got)
	})

	t.Run("reports a broken stream", func(t *testing.T) {
		boom := errors.New("connection reset")
		s := NewPCMReaderStream(io.MultiReader(bytes.NewReader(pcm16(1, 2)), iotest.ErrReader(boom)))
		buf := make([][2]float64, 8)
		n, ok := s.Stream(buf)
		assert.True(t, ok)
		assert.Equal(t, 2, n)
		n, ok = s.Stream(buf)
		assert.False(t, ok)
		assert.Equal(t, 0, n)
		assert.ErrorIs(t, s.Err(), boom)
	})
}

func TestOpenPocketStream(t *testing.T) {
	pcm := pcm16(1, -1, 1000, -1000, 32767)

	t.Run("streams PCM with the header stripped", func(t *testing.T) {
		srv := pocketServer(t, "marius", pcm)
		defer srv.Close()

		body, rate, err := openPocketStream(context.Background(), srv.URL, "Ready for review", "marius")
		require.NoError(t, err)
		defer body.Close()
		assert.Equal(t, 24000, rate)
		got, err := io.ReadAll(body)
		require.NoError(t, err)
		assert.Equal(t, pcm, got)
	})

	t.Run("omits voice_url when no voice is chosen", func(t *testing.T) {
		srv := pocketServer(t, "", pcm)
		defer srv.Close()
		body, _, err := openPocketStream(context.Background(), srv.URL, "Ready for review", "")
		require.NoError(t, err)
		body.Close()
	})

	t.Run("a 4xx is reported with the server's detail and never falls back", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadRequest)
			io.WriteString(w, `{"detail":"voice_url must start with http://, https://, or hf://"}`)
		}))
		defer srv.Close()

		_, _, err := openPocketStream(context.Background(), srv.URL, "Ready for review", "nobody")
		var status *pocketStatusError
		require.ErrorAs(t, err, &status)
		assert.Equal(t, http.StatusBadRequest, status.code)
		assert.Equal(t, "voice_url must start with http://, https://, or hf://", status.detail)
		assert.False(t, pocketFallsBack(err))
	})

	t.Run("a 5xx falls back on Windows", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			http.Error(w, "Internal Server Error", http.StatusInternalServerError)
		}))
		defer srv.Close()

		_, _, err := openPocketStream(context.Background(), srv.URL, "Ready for review", "")
		require.Error(t, err)
		assert.Equal(t, runtime.GOOS == "windows", pocketFallsBack(err))
	})

	t.Run("a server that is not running is unreachable and falls back on Windows", func(t *testing.T) {
		srv := httptest.NewServer(http.NotFoundHandler())
		deadURL := srv.URL
		srv.Close()

		_, _, err := openPocketStream(context.Background(), deadURL, "Ready for review", "")
		var unreachable *pocketUnreachableError
		require.ErrorAs(t, err, &unreachable)
		assert.Equal(t, runtime.GOOS == "windows", pocketFallsBack(err))

		t.Setenv("POCKET_TTS_FALLBACK", "none")
		assert.False(t, pocketFallsBack(err), "POCKET_TTS_FALLBACK=none turns the fallback off")
	})

	t.Run("a non-WAV 200 is rejected", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			io.WriteString(w, "<html>not audio</html>")
		}))
		defer srv.Close()

		_, _, err := openPocketStream(context.Background(), srv.URL, "Ready for review", "")
		require.Error(t, err)
		assert.Contains(t, err.Error(), "unexpected audio")
		assert.False(t, pocketFallsBack(err))
	})
}

func TestPocketSettings(t *testing.T) {
	t.Run("url defaults and trims", func(t *testing.T) {
		t.Setenv("POCKET_TTS_URL", "")
		assert.Equal(t, DefaultPocketURL, pocketURL())
		t.Setenv("POCKET_TTS_URL", " http://127.0.0.1:8765/ ")
		assert.Equal(t, "http://127.0.0.1:8765", pocketURL())
	})

	t.Run("voice: argument, then env, then server default", func(t *testing.T) {
		t.Setenv("POCKET_TTS_VOICE", "")
		assert.Equal(t, "", pocketVoice(PocketTTSParams{Text: "x"}))
		t.Setenv("POCKET_TTS_VOICE", "jean")
		assert.Equal(t, "jean", pocketVoice(PocketTTSParams{Text: "x"}))
		assert.Equal(t, "marius", pocketVoice(PocketTTSParams{Text: "x", Voice: stringPtr("marius")}))
		assert.Equal(t, "jean", pocketVoice(PocketTTSParams{Text: "x", Voice: stringPtr(" ")}))
	})

	t.Run("a configured server is offered first by the interactive tool", func(t *testing.T) {
		t.Setenv("POCKET_TTS_URL", "")
		for _, p := range availableProviders() {
			assert.NotEqual(t, ProviderPocket, p.ID, "pocket is opt-in")
		}
		t.Setenv("POCKET_TTS_URL", "http://127.0.0.1:8765")
		providers := availableProviders()
		require.NotEmpty(t, providers)
		assert.Equal(t, ProviderPocket, providers[0].ID)
	})

	t.Run("recommendation args", func(t *testing.T) {
		args := providerRecommendationArgs(ProviderPocket, "hello", nil)
		assert.Equal(t, map[string]any{"text": "hello"}, args)
		args = providerRecommendationArgs(ProviderPocket, "hello", map[string]any{"voice": "vera"})
		assert.Equal(t, "vera", args["voice"])
		assert.NotNil(t, settingsSchemaForProvider(ProviderPocket))
	})
}

// withSaveOnly switches the package into save-without-playback mode for one test, so the
// whole speakPocket path runs without an audio device.
func withSaveOnly(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	prevDir, prevNoPlay := outputDir, noPlay
	outputDir, noPlay = dir, true
	t.Cleanup(func() { outputDir, noPlay = prevDir, prevNoPlay })
	return dir
}

func resultText(t *testing.T, res *mcp.CallToolResult) string {
	t.Helper()
	require.NotNil(t, res)
	require.Len(t, res.Content, 1)
	tc, ok := res.Content[0].(*mcp.TextContent)
	require.True(t, ok)
	return tc.Text
}

func TestSpeakPocket(t *testing.T) {
	t.Run("saves the streamed audio as a well-formed WAV", func(t *testing.T) {
		dir := withSaveOnly(t)
		pcm := pcm16(10, 20, 30, 40, 50, 60, 70)
		srv := pocketServer(t, "", pcm)
		defer srv.Close()
		t.Setenv("POCKET_TTS_URL", srv.URL)
		t.Setenv("POCKET_TTS_VOICE", "")

		res := speakPocket(context.Background(), PocketTTSParams{Text: "Ready for review"})
		assert.False(t, res.IsError, resultText(t, res))
		text := resultText(t, res)
		require.True(t, strings.HasPrefix(text, "Saved: "), text)
		path := strings.TrimPrefix(text, "Saved: ")
		assert.True(t, strings.HasPrefix(path, dir))

		saved, err := os.ReadFile(path)
		require.NoError(t, err)
		// A real length this time, not the streaming placeholder.
		assert.Equal(t, uint32(len(pcm)), binary.LittleEndian.Uint32(saved[40:44]))
		assert.Equal(t, pcm, saved[44:])
		rate, err := readWAVHeader(bytes.NewReader(saved))
		require.NoError(t, err)
		assert.Equal(t, 24000, rate)
	})

	t.Run("a rejected request is an error, not a fallback", func(t *testing.T) {
		withSaveOnly(t)
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusBadRequest)
			io.WriteString(w, `{"detail":"Text cannot be empty"}`)
		}))
		defer srv.Close()
		t.Setenv("POCKET_TTS_URL", srv.URL)

		res := speakPocket(context.Background(), PocketTTSParams{Text: "Ready for review"})
		assert.True(t, res.IsError)
		assert.Contains(t, resultText(t, res), "Text cannot be empty")
	})

	t.Run("an unreachable server is an error when the fallback is off", func(t *testing.T) {
		withSaveOnly(t)
		srv := httptest.NewServer(http.NotFoundHandler())
		deadURL := srv.URL
		srv.Close()
		t.Setenv("POCKET_TTS_URL", deadURL)
		t.Setenv("POCKET_TTS_FALLBACK", "none")

		res := speakPocket(context.Background(), PocketTTSParams{Text: "Ready for review"})
		assert.True(t, res.IsError)
		assert.Contains(t, resultText(t, res), "server not reachable")
	})

	t.Run("a cancelled request says so", func(t *testing.T) {
		withSaveOnly(t)
		srv := pocketServer(t, "", pcm16(1))
		defer srv.Close()
		t.Setenv("POCKET_TTS_URL", srv.URL)
		ctx, cancel := context.WithCancel(context.Background())
		cancel()

		res := speakPocket(ctx, PocketTTSParams{Text: "Ready for review"})
		assert.False(t, res.IsError)
		assert.Equal(t, "Pocket TTS request cancelled", resultText(t, res))
	})
}

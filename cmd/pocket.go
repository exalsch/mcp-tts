package cmd

import (
	"bufio"
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"runtime"
	"strings"
	"time"

	"github.com/charmbracelet/log"
	"github.com/gopxl/beep/v2"
	"github.com/gopxl/beep/v2/speaker"
	"github.com/modelcontextprotocol/go-sdk/mcp"
)

// Pocket TTS provider.
//
// Kyutai's Pocket TTS (https://github.com/kyutai-labs/pocket-tts) is a 100M-parameter model that
// runs on the CPU. `pocket-tts serve` keeps it loaded and answers POST /tts with a WAV that is
// streamed while it is still being generated, so this provider starts playing on the first
// chunk instead of waiting for the whole clip.
//
// mcp-tts does not start the server itself. Loading the model takes several seconds, which is
// far too long to pay on every call, so the server has to be running already. Point
// POCKET_TTS_URL at it (default http://127.0.0.1:8000, the `pocket-tts serve` default).
//
// On Windows, when the server cannot be reached, the call falls back to Windows SAPI so a
// spoken notification is never silently lost. Set POCKET_TTS_FALLBACK=none to get an error
// instead.

// PocketTTSParams are the pocket_tts tool arguments.
type PocketTTSParams struct {
	Text  string  `json:"text" mcp:"The text to speak aloud"`
	Voice *string `json:"voice,omitempty" mcp:"Built-in Pocket TTS voice (e.g. 'alba', 'marius', 'jean'); default: the server's default voice"`
}

// pocketHTTPClient fails fast when nothing is listening, so the SAPI fallback still feels
// immediate. There is deliberately no overall timeout: a long text streams for as long as it
// takes to speak. Cancellation comes from the request context instead.
var pocketHTTPClient = &http.Client{
	Transport: &http.Transport{
		Proxy:                 nil, // a loopback server must never be sent through a proxy
		DialContext:           (&net.Dialer{Timeout: 2 * time.Second}).DialContext,
		ResponseHeaderTimeout: 15 * time.Second,
	},
}

// pocketUnreachableError means no HTTP response came back at all: the server is not running,
// is still loading its model, or refused the connection.
type pocketUnreachableError struct{ err error }

func (e *pocketUnreachableError) Error() string { return "server not reachable: " + e.err.Error() }
func (e *pocketUnreachableError) Unwrap() error { return e.err }

// pocketStatusError is a non-200 answer from a server that is up.
type pocketStatusError struct {
	code   int
	detail string
}

func (e *pocketStatusError) Error() string {
	if e.detail == "" {
		return fmt.Sprintf("server returned HTTP %d", e.code)
	}
	return fmt.Sprintf("server returned HTTP %d: %s", e.code, e.detail)
}

// pocketURL returns the server base URL without a trailing slash.
func pocketURL() string {
	u := strings.TrimSpace(os.Getenv("POCKET_TTS_URL"))
	if u == "" {
		u = DefaultPocketURL
	}
	return strings.TrimRight(u, "/")
}

// pocketVoice picks the voice for a call: the argument, then POCKET_TTS_VOICE, then empty,
// which lets the server use the default voice it was started with.
func pocketVoice(input PocketTTSParams) string {
	if input.Voice != nil && strings.TrimSpace(*input.Voice) != "" {
		return strings.TrimSpace(*input.Voice)
	}
	return strings.TrimSpace(os.Getenv("POCKET_TTS_VOICE"))
}

// openPocketStream sends the text to the server and returns the raw PCM of the answer, already
// past the WAV header, together with its sample rate. The caller must close the reader.
func openPocketStream(ctx context.Context, baseURL, text, voice string) (io.ReadCloser, int, error) {
	form := url.Values{"text": {text}}
	if voice != "" {
		form.Set("voice_url", voice)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, baseURL+"/tts", strings.NewReader(form.Encode()))
	if err != nil {
		return nil, 0, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")

	res, err := pocketHTTPClient.Do(req)
	if err != nil {
		return nil, 0, &pocketUnreachableError{err: err}
	}

	if res.StatusCode != http.StatusOK {
		defer res.Body.Close()
		body, _ := io.ReadAll(io.LimitReader(res.Body, 4096))
		return nil, 0, &pocketStatusError{code: res.StatusCode, detail: pocketErrorDetail(body)}
	}

	br := bufio.NewReader(res.Body)
	sampleRate, err := readWAVHeader(br)
	if err != nil {
		res.Body.Close()
		return nil, 0, fmt.Errorf("unexpected audio from server: %w", err)
	}

	return struct {
		io.Reader
		io.Closer
	}{br, res.Body}, sampleRate, nil
}

// pocketErrorDetail pulls FastAPI's {"detail": ...} out of an error body, or returns the body.
func pocketErrorDetail(body []byte) string {
	var parsed struct {
		Detail any `json:"detail"`
	}
	if json.Unmarshal(body, &parsed) == nil && parsed.Detail != nil {
		if s, ok := parsed.Detail.(string); ok {
			return s
		}
		b, _ := json.Marshal(parsed.Detail)
		return string(b)
	}
	return strings.TrimSpace(string(body))
}

// readWAVHeader consumes a RIFF/WAVE header up to the start of the data chunk and returns the
// sample rate. The data chunk size is ignored on purpose: a streamed WAV cannot know its own
// length up front. pocket-tts writes a placeholder there, so the data runs until EOF.
// Only 16-bit mono PCM is accepted, because that is what PCMReaderStream plays.
func readWAVHeader(r io.Reader) (int, error) {
	var riff [12]byte
	if _, err := io.ReadFull(r, riff[:]); err != nil {
		return 0, fmt.Errorf("reading RIFF header: %w", err)
	}
	if string(riff[0:4]) != "RIFF" || string(riff[8:12]) != "WAVE" {
		return 0, errors.New("not a RIFF/WAVE stream")
	}

	sampleRate := 0
	for i := 0; i < 16; i++ {
		var hdr [8]byte
		if _, err := io.ReadFull(r, hdr[:]); err != nil {
			return 0, fmt.Errorf("reading chunk header: %w", err)
		}
		id := string(hdr[0:4])
		size := int64(binary.LittleEndian.Uint32(hdr[4:8]))

		switch id {
		case "fmt ":
			if size < 16 || size > 1024 {
				return 0, fmt.Errorf("fmt chunk has implausible size %d", size)
			}
			chunk := make([]byte, size+size%2)
			if _, err := io.ReadFull(r, chunk); err != nil {
				return 0, fmt.Errorf("reading fmt chunk: %w", err)
			}
			format := binary.LittleEndian.Uint16(chunk[0:2])
			channels := binary.LittleEndian.Uint16(chunk[2:4])
			rate := binary.LittleEndian.Uint32(chunk[4:8])
			bits := binary.LittleEndian.Uint16(chunk[14:16])
			if format != 1 || channels != 1 || bits != 16 {
				return 0, fmt.Errorf("unsupported format (format %d, %d channels, %d bits), want 16-bit mono PCM",
					format, channels, bits)
			}
			if rate == 0 {
				return 0, errors.New("sample rate is 0")
			}
			sampleRate = int(rate)
		case "data":
			if sampleRate == 0 {
				return 0, errors.New("data chunk before fmt chunk")
			}
			return sampleRate, nil
		default:
			if size > 1<<20 {
				return 0, fmt.Errorf("chunk %q is too large to skip (%d bytes)", id, size)
			}
			if _, err := io.CopyN(io.Discard, r, size+size%2); err != nil {
				return 0, fmt.Errorf("skipping chunk %q: %w", id, err)
			}
		}
	}
	return 0, errors.New("no data chunk found")
}

// pocketFallsBack reports whether a failure should be covered by speaking through SAPI. Only
// a server that is down or broken qualifies. A 4xx (an unknown voice, say) is the caller's
// mistake and is reported rather than papered over.
func pocketFallsBack(err error) bool {
	if runtime.GOOS != "windows" || strings.EqualFold(os.Getenv("POCKET_TTS_FALLBACK"), "none") {
		return false
	}
	var unreachable *pocketUnreachableError
	if errors.As(err, &unreachable) {
		return true
	}
	var status *pocketStatusError
	return errors.As(err, &status) && status.code >= 500
}

// speakPocket runs one pocket_tts call. The caller holds the TTS lock.
func speakPocket(ctx context.Context, input PocketTTSParams) *mcp.CallToolResult {
	text := input.Text
	voice := pocketVoice(input)
	baseURL := pocketURL()

	body, sampleRate, err := openPocketStream(ctx, baseURL, text, voice)
	if err != nil {
		if ctx.Err() != nil {
			return textResult("Pocket TTS request cancelled")
		}
		if pocketFallsBack(err) {
			return pocketSAPIFallback(ctx, text, baseURL, err)
		}
		log.Error("Pocket TTS request failed", "url", baseURL, "error", err)
		return errorResult(fmt.Sprintf("Error: Pocket TTS at %s failed: %v", baseURL, err))
	}
	defer body.Close()

	var pcm io.Reader = body
	var saved *bytes.Buffer
	if shouldSave() {
		saved = &bytes.Buffer{}
		pcm = io.TeeReader(body, saved)
	}

	if !shouldPlay() {
		if _, err := io.Copy(io.Discard, pcm); err != nil {
			return errorResult(fmt.Sprintf("Error: Pocket TTS stream interrupted: %v", err))
		}
		savedPath, err := saveWAV(saved.Bytes(), sampleRate, text)
		if err != nil {
			return errorResult(fmt.Sprintf("Error saving audio: %v", err))
		}
		return textResult(formatSaveResult(text, savedPath, false))
	}

	stream := NewPCMReaderStream(pcm)
	rate := beep.SampleRate(sampleRate)
	if err := initSpeaker(rate); err != nil {
		log.Error("Failed to initialize speaker", "error", err)
		return errorResult(fmt.Sprintf("Error: Failed to initialize speaker: %v", err))
	}
	playback := resampleToSpeaker(stream, rate)

	done := make(chan struct{})
	speaker.Play(beep.Seq(playback, beep.Callback(func() { close(done) })))
	log.Info("Speaking via Pocket TTS", "text", text, "voice", voice, "url", baseURL)

	select {
	case <-done:
	case <-ctx.Done():
		// Close the body first: the speaker holds its lock while Stream waits on the network,
		// so speaker.Clear would block until the next chunk arrived.
		body.Close()
		speaker.Clear()
		log.Info("Pocket TTS playback cancelled by user")
		return textResult("Pocket TTS playback cancelled")
	}

	if err := stream.Err(); err != nil {
		log.Error("Pocket TTS stream interrupted", "error", err)
		return errorResult(fmt.Sprintf("Error: Pocket TTS stream interrupted after %.1fs: %v",
			float64(stream.Position())/float64(sampleRate), err))
	}

	var savedPath string
	if saved != nil {
		if savedPath, err = saveWAV(saved.Bytes(), sampleRate, text); err != nil {
			log.Error("Failed to save WAV file", "error", err)
			savedPath = ""
		}
	}
	return textResult(formatSaveResult(text, savedPath, true))
}

// pocketSAPIFallback speaks through Windows SAPI when the Pocket TTS server is unavailable and
// says so in the result, so the caller can tell the server needs attention.
func pocketSAPIFallback(ctx context.Context, text, baseURL string, cause error) *mcp.CallToolResult {
	note := fmt.Sprintf("Pocket TTS at %s unavailable (%v)", baseURL, cause)
	log.Warn("Pocket TTS unavailable, falling back to Windows SAPI", "url", baseURL, "error", cause)
	if err := speakSAPI(ctx, text, nil, nil, ""); err != nil {
		if ctx.Err() != nil {
			return textResult("SAPI speech cancelled")
		}
		return errorResult(fmt.Sprintf("Error: %s. The Windows SAPI fallback failed too: %v", note, err))
	}
	return textResult(formatSaveResult(text, "", true) + "\n" + note + "; spoke via Windows SAPI instead")
}

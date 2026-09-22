/*
Copyright © 2025 blacktop

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
package cmd

import (
	"errors"
	"io"
	"sync/atomic"

	"github.com/gopxl/beep/v2"
)

// PCMStream implements beep.StreamSeeker for playing raw PCM audio data
type PCMStream struct {
	data       []byte
	sampleRate beep.SampleRate
	position   int
}

func (s *PCMStream) Stream(samples [][2]float64) (n int, ok bool) {
	if s.position >= len(s.data) {
		return 0, false
	}

	for i := range samples {
		if s.position+1 >= len(s.data) {
			return i, true
		}

		// Convert 16-bit little-endian PCM to float64
		sample16 := int16(s.data[s.position]) | int16(s.data[s.position+1])<<8
		sampleFloat := float64(sample16) / 32768.0

		// Mono to stereo
		samples[i][0] = sampleFloat
		samples[i][1] = sampleFloat

		s.position += 2
	}

	return len(samples), true
}

func (s *PCMStream) Err() error {
	return nil
}

func (s *PCMStream) Len() int {
	return len(s.data) / 2 // 16-bit samples
}

func (s *PCMStream) Position() int {
	return s.position / 2
}

func (s *PCMStream) Seek(p int) error {
	s.position = min(max(p*2, 0), len(s.data))
	return nil
}

// PCMReaderStream implements beep.Streamer over 16-bit little-endian mono PCM that is still
// arriving, such as a streamed HTTP body. Unlike PCMStream it never holds the whole clip, so
// playback can start on the first chunk instead of after the last one.
//
// Stream blocks until it has filled the buffer, as the beep.Streamer contract asks. When the
// network is slower than playback that shows up as a short gap in the audio, never as an early
// end of the clip.
type PCMReaderStream struct {
	r        io.Reader
	buf      []byte
	pending  []byte // odd byte carried over from the previous read
	position atomic.Int64
	err      error
}

// NewPCMReaderStream wraps r, which must yield raw PCM with any container header removed.
func NewPCMReaderStream(r io.Reader) *PCMReaderStream {
	return &PCMReaderStream{r: r}
}

func (s *PCMReaderStream) Stream(samples [][2]float64) (n int, ok bool) {
	if s.err != nil {
		return 0, false
	}

	need := len(samples) * 2
	if cap(s.buf) < need {
		s.buf = make([]byte, need)
	}
	buf := s.buf[:need]
	filled := copy(buf, s.pending)
	s.pending = s.pending[:0]

	read, err := io.ReadFull(s.r, buf[filled:])
	filled += read
	if err != nil {
		if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
			s.err = io.EOF
		} else {
			s.err = err
		}
	}

	n = filled / 2
	if filled%2 == 1 {
		s.pending = append(s.pending, buf[filled-1])
	}

	for i := 0; i < n; i++ {
		sample16 := int16(buf[2*i]) | int16(buf[2*i+1])<<8
		sampleFloat := float64(sample16) / 32768.0
		samples[i][0] = sampleFloat
		samples[i][1] = sampleFloat
	}
	s.position.Add(int64(n))

	if n == 0 {
		return 0, false
	}
	return n, true
}

// Err returns the read error that ended the stream, or nil when it simply ran out of data.
func (s *PCMReaderStream) Err() error {
	if errors.Is(s.err, io.EOF) {
		return nil
	}
	return s.err
}

// Position returns the number of samples streamed so far. Safe to call from another goroutine.
func (s *PCMReaderStream) Position() int {
	return int(s.position.Load())
}

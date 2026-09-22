//go:build !windows

package cmd

import (
	"context"
	"errors"
)

// Windows SAPI is only available on Windows. These stubs keep the package building elsewhere.
// root.go registers sapi_tts only when runtime.GOOS is "windows". The Pocket TTS fallback
// checks the same thing, so none of them is reached at runtime.

var errSAPIUnsupported = errors.New("windows SAPI is only available on Windows")

func IsSAPIVoiceInstalled(string) (bool, error) {
	return false, errSAPIUnsupported
}

func SAPIVoiceNotInstalledError(voiceName string) string {
	return "Voice \"" + voiceName + "\" is not available: Windows SAPI is only available on Windows"
}

func speakSAPI(context.Context, string, *int, *string, string) error {
	return errSAPIUnsupported
}

func sapiSavePath(string) string {
	return ""
}

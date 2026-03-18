package cmd

import (
	"context"
	"fmt"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"

	"github.com/charmbracelet/log"
)

// Windows SAPI voice cache
var (
	sapiVoiceCache     map[string]bool
	sapiVoiceCacheOnce sync.Once
	sapiVoiceCacheErr  error
)

// getInstalledSAPIVoices queries Windows SAPI for installed voices via PowerShell.
func getInstalledSAPIVoices() (map[string]bool, error) {
	sapiVoiceCacheOnce.Do(func() {
		sapiVoiceCache = make(map[string]bool)

		// Use PowerShell to list SAPI voices
		cmd := exec.Command("powershell", "-NoProfile", "-Command",
			`Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).GetInstalledVoices() | ForEach-Object { $_.VoiceInfo.Name }`)
		out, err := cmd.Output()
		if err != nil {
			sapiVoiceCacheErr = fmt.Errorf("failed to query SAPI voices: %w", err)
			return
		}

		for _, line := range strings.Split(string(out), "\n") {
			line = strings.TrimSpace(line)
			if line != "" {
				sapiVoiceCache[line] = true
			}
		}
	})

	return sapiVoiceCache, sapiVoiceCacheErr
}

// IsSAPIVoiceInstalled checks if a specific SAPI voice is installed.
func IsSAPIVoiceInstalled(voiceName string) (bool, error) {
	voices, err := getInstalledSAPIVoices()
	if err != nil {
		return false, err
	}
	return voices[voiceName], nil
}

// SAPIVoiceNotInstalledError returns a user-friendly error message for missing SAPI voices.
func SAPIVoiceNotInstalledError(voiceName string) string {
	return "Voice \"" + voiceName + "\" is not installed. " +
		"To download additional voices, go to: Settings → Time & Language → Speech → Manage voices"
}

// speakSAPI uses Windows SAPI via PowerShell to speak text.
func speakSAPI(ctx context.Context, text string, rate *int, voice *string, savePath string) error {
	// Build PowerShell script
	var sb strings.Builder
	sb.WriteString("Add-Type -AssemblyName System.Speech; ")
	sb.WriteString("$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer; ")

	if voice != nil && *voice != "" {
		// Validate voice name characters
		for _, r := range *voice {
			if !((r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') ||
				r == ' ' || r == '(' || r == ')' || r == '-' || r == '_') {
				return fmt.Errorf("voice contains invalid characters: %s", *voice)
			}
		}
		sb.WriteString(fmt.Sprintf("$synth.SelectVoice('%s'); ", *voice))
	}

	if rate != nil {
		// SAPI rate is -10 to 10; map from WPM (50-500) to SAPI range
		// Default 200 WPM -> rate 0, 50 WPM -> -10, 500 WPM -> 10
		sapiRate := mapWPMToSAPIRate(*rate)
		sb.WriteString(fmt.Sprintf("$synth.Rate = %d; ", sapiRate))
	}

	if savePath != "" {
		sb.WriteString(fmt.Sprintf("$synth.SetOutputToWaveFile('%s'); ", savePath))
	}

	// Escape single quotes in text for PowerShell
	escapedText := strings.ReplaceAll(text, "'", "''")
	sb.WriteString(fmt.Sprintf("$synth.Speak('%s'); ", escapedText))

	if savePath != "" {
		sb.WriteString("$synth.SetOutputToDefaultAudioDevice(); ")
	}

	sb.WriteString("$synth.Dispose()")

	log.Debug("Executing SAPI TTS via PowerShell", "voice", voice, "rate", rate)
	cmd := exec.CommandContext(ctx, "powershell", "-NoProfile", "-Command", sb.String())
	return cmd.Run()
}

// mapWPMToSAPIRate converts words-per-minute (50-500) to SAPI rate (-10 to 10).
func mapWPMToSAPIRate(wpm int) int {
	if wpm < 50 {
		wpm = 50
	}
	if wpm > 500 {
		wpm = 500
	}
	// Linear mapping: 50 WPM -> -10, 200 WPM -> 0, 500 WPM -> 10
	if wpm <= 200 {
		return (wpm - 200) * 10 / 150
	}
	return (wpm - 200) * 10 / 300
}

// sapiSavePath returns the WAV file path for saving SAPI audio.
func sapiSavePath(text string) string {
	if !shouldSave() {
		return ""
	}
	filename := generateFilename(text, "wav")
	return filepath.Join(outputDir, filename)
}

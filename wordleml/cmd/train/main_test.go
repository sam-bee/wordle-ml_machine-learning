package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestDryRunPrintsConfigurationWithoutCreatingFiles(t *testing.T) {
	temporary := t.TempDir()
	dataDir := filepath.Join(temporary, "vocabulary")
	t.Setenv("WORDLEML_DATA_DIR", dataDir)
	var stdout, stderr bytes.Buffer
	err := run(nil, &stdout, &stderr)
	if err != nil {
		t.Fatalf("run dry mode: %v\nstderr: %s", err, stderr.String())
	}
	for _, want := range []string{
		"data_dir=" + dataDir,
		"imitation_dir=" + filepath.Join(dataDir, "imitation"),
		"checkpoint_dir=" + filepath.Join(dataDir, "checkpoints", "first-run"),
		"tensorboard_dir=" + filepath.Join(dataDir, "tensorboard", "first-run"),
		"steps=0",
	} {
		if !strings.Contains(stdout.String(), want) {
			t.Errorf("dry-run output %q does not contain %q", stdout.String(), want)
		}
	}
	for _, path := range []string{
		dataDir,
		filepath.Join(dataDir, "checkpoints", "first-run"),
		filepath.Join(dataDir, "tensorboard", "first-run"),
	} {
		if _, err := os.Stat(path); !os.IsNotExist(err) {
			t.Errorf("dry run created %q: %v", path, err)
		}
	}
}

func TestRunRejectsInvalidTrainingFlags(t *testing.T) {
	for _, args := range [][]string{
		{"-batch-size=0"},
		{"-learning-rate=0"},
		{"-learning-rate=NaN"},
		{"-seed=0"},
		{"-steps=-1"},
	} {
		var stdout, stderr bytes.Buffer
		if err := run(args, &stdout, &stderr); err == nil {
			t.Errorf("run(%q) succeeded, want validation error", args)
		}
	}
}

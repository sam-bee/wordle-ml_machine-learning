package main

import (
	"bytes"
	"path/filepath"
	"testing"
)

func TestParseConfigRequiresFixedRunIdentityAndStage(t *testing.T) {
	for _, args := range [][]string{nil, {"-run-id=proof"}, {"-stage=mini"}, {"-run-id=proof", "-stage=unknown"}} {
		var stderr bytes.Buffer
		if _, err := parseConfig(args, &stderr); err == nil {
			t.Errorf("parseConfig(%q) succeeded", args)
		}
	}
}

func TestParseConfigUsesOnlyFixedProofFlags(t *testing.T) {
	var stderr bytes.Buffer
	config, err := parseConfig([]string{"-data-dir=/frozen", "-runs-dir=/runs", "-run-id=mini-resume", "-stage=mini", "-stop-at=500"}, &stderr)
	if err != nil {
		t.Fatalf("parseConfig: %v", err)
	}
	if config.dataDir != "/frozen" || config.runsDir != "/runs" || config.runID != "mini-resume" || config.stage != "mini" || config.stopAt != 500 {
		t.Fatalf("config = %#v", config)
	}
	if _, err := parseConfig([]string{"-data-dir=" + filepath.Join(t.TempDir(), "data"), "-run-id=bad", "-stage=mini", "-steps=10"}, &stderr); err == nil {
		t.Fatal("obsolete -steps flag was accepted")
	}
}

func TestParseConfigDefaultsRunsBesideDataOrUsesEnvironment(t *testing.T) {
	t.Setenv("WORDLEML_DATA_DIR", "/workspace/data")
	t.Setenv("WORDLEML_RUNS_DIR", "")
	var stderr bytes.Buffer
	config, err := parseConfig([]string{"-run-id=full-proof", "-stage=full"}, &stderr)
	if err != nil {
		t.Fatal(err)
	}
	if config.runsDir != "/workspace/runs" {
		t.Fatalf("default runs directory = %q, want /workspace/runs", config.runsDir)
	}
	t.Setenv("WORDLEML_RUNS_DIR", "/mounted-runs")
	config, err = parseConfig([]string{"-run-id=full-proof", "-stage=full"}, &stderr)
	if err != nil {
		t.Fatal(err)
	}
	if config.runsDir != "/mounted-runs" {
		t.Fatalf("environment runs directory = %q", config.runsDir)
	}
}

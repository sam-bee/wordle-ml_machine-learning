package main

import (
	"bytes"
	"testing"
)

func TestParseConfig(t *testing.T) {
	t.Setenv("WORDLEML_DATA_DIR", "/data")
	t.Setenv("WORDLEML_RUNS_DIR", "/runs")
	t.Setenv("WORDLEML_INFERENCE_ADDR", ":9000")
	t.Setenv("WORDLEML_INFERENCE_RUN_ID", "")
	config, err := parseConfig([]string{"-run-id=full-proof"}, &bytes.Buffer{})
	if err != nil {
		t.Fatal(err)
	}
	if config.runID != "full-proof" || config.address != ":9000" || config.dataDir != "/data" || config.runsDir != "/runs" {
		t.Fatalf("config = %+v", config)
	}
}

func TestParseConfigRequiresRunID(t *testing.T) {
	t.Setenv("WORDLEML_INFERENCE_RUN_ID", "")
	if _, err := parseConfig(nil, &bytes.Buffer{}); err == nil {
		t.Fatal("missing run ID unexpectedly accepted")
	}
}

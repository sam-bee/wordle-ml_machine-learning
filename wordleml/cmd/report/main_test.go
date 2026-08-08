package main

import (
	"bytes"
	"path/filepath"
	"testing"
)

func TestParseConfigRequiresAllProofRunIDs(t *testing.T) {
	for _, args := range [][]string{
		nil,
		{"-overfit-run-id=overfit"},
		{"-overfit-run-id=overfit", "-mini-run-id=mini"},
		{"-overfit-run-id=overfit", "-mini-run-id=mini", "-full-run-id=full", "-output="},
	} {
		if _, err := parseConfig(args, &bytes.Buffer{}); err == nil {
			t.Fatalf("parseConfig(%q) unexpectedly succeeded", args)
		}
	}
}

func TestParseConfigUsesFixedInputsAndEnvironment(t *testing.T) {
	t.Setenv("WORDLEML_RUNS_DIR", "/proof-runs")
	got, err := parseConfig([]string{
		"-overfit-run-id=overfit",
		"-mini-run-id=mini",
		"-full-run-id=full",
		"-output=/reports/proof.md",
	}, &bytes.Buffer{})
	if err != nil {
		t.Fatal(err)
	}
	if got.runsDir != "/proof-runs" || got.overfitRunID != "overfit" || got.miniRunID != "mini" || got.fullRunID != "full" || got.outputPath != "/reports/proof.md" {
		t.Fatalf("unexpected config: %+v", got)
	}

	t.Setenv("WORDLEML_RUNS_DIR", "")
	got, err = parseConfig([]string{"-overfit-run-id=o", "-mini-run-id=m", "-full-run-id=f"}, &bytes.Buffer{})
	if err != nil {
		t.Fatal(err)
	}
	if got.runsDir != "../runs" || got.outputPath != filepath.Join("..", "docs", "ml", "initial-training-proof-report.md") {
		t.Fatalf("unexpected defaults: %+v", got)
	}
}

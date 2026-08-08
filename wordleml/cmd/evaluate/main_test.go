package main

import (
	"bytes"
	"testing"
)

func TestParseConfigRejectsNonProofModes(t *testing.T) {
	for _, args := range [][]string{
		{"-run-id=x", "-checkpoint=test", "-mode=games10"},
		{"-run-id=x", "-checkpoint=best", "-mode=test"},
		{"-checkpoint=best", "-mode=games10"},
	} {
		if _, err := parseConfig(args, &bytes.Buffer{}); err == nil {
			t.Fatalf("parseConfig(%v) unexpectedly succeeded", args)
		}
	}
}

func TestParseConfigDefaultsAndFixedMode(t *testing.T) {
	got, err := parseConfig([]string{"-run-id=proof-1", "-mode=ablations"}, &bytes.Buffer{})
	if err != nil {
		t.Fatal(err)
	}
	if got.checkpoint != "best" || got.mode != "ablations" || got.runID != "proof-1" {
		t.Fatalf("unexpected config: %+v", got)
	}
}

func TestParseConfigAcceptsInitialCheckpoint(t *testing.T) {
	got, err := parseConfig([]string{"-run-id=overfit-proof", "-checkpoint=initial", "-mode=games10"}, &bytes.Buffer{})
	if err != nil {
		t.Fatal(err)
	}
	if got.checkpoint != "initial" {
		t.Fatalf("checkpoint = %q, want initial", got.checkpoint)
	}
}

package main

import (
	"io"
	"path/filepath"
	"testing"
)

func TestParseConfigUsesNormalDefaults(t *testing.T) {
	configuration, err := parseConfig([]string{"-model-dir=/tmp/export", "-data-dir=/tmp/data"}, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	if configuration.warmup != 20 || configuration.iterations != 200 || configuration.report != filepath.Join("/tmp/export", "benchmark-report.json") {
		t.Fatalf("configuration = %+v", configuration)
	}
}

func TestParseConfigUsesSmallProfileDefaults(t *testing.T) {
	configuration, err := parseConfig([]string{"-model-dir=/tmp/export", "-data-dir=/tmp/data", "-profile"}, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	if configuration.warmup != 2 || configuration.iterations != 10 {
		t.Fatalf("profile configuration = %+v", configuration)
	}
}

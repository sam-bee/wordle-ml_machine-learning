package main

import (
	"io"
	"path/filepath"
	"testing"
)

func TestParseConfigUsesModelDirectoryForDefaultReport(t *testing.T) {
	t.Setenv("MODEL_DIR", "/tmp/export")
	t.Setenv("DATA_DIR", "/tmp/data")
	configuration, err := parseConfig(nil, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	if configuration.dataDir != "/tmp/data" || configuration.modelDir != "/tmp/export" || configuration.report != filepath.Join("/tmp/export", "verification-report.json") {
		t.Fatalf("configuration = %+v", configuration)
	}
}

func TestParseConfigRejectsMissingModelDirectory(t *testing.T) {
	t.Setenv("MODEL_DIR", "")
	t.Setenv("WORDLEML_CUDA_MODEL_DIR", "")
	if _, err := parseConfig([]string{"-data-dir=/tmp/data"}, io.Discard); err == nil {
		t.Fatal("missing model directory was accepted")
	}
}

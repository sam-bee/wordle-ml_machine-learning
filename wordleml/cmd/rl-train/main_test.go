package main

import (
	"bytes"
	"path/filepath"
	"testing"
)

func TestParseConfigRequiresOnlyPPOAndIsolationPaths(t *testing.T) {
	var stderr bytes.Buffer
	config, err := parseConfig([]string{
		"--algorithm=ppo", "--config=../../../configs/rl/ppo-pilot-v1.json",
		"--supervised-checkpoint=../runs/baseline", "--run-dir=../runs/fresh",
	}, &stderr)
	if err != nil {
		t.Fatal(err)
	}
	if !filepath.IsAbs(config.configPath) || !filepath.IsAbs(config.dataDir) || !filepath.IsAbs(config.supervisedCheckpoint) || !filepath.IsAbs(config.runDir) {
		t.Fatalf("unexpected parsed paths: %+v", config)
	}
	if _, err := parseConfig([]string{"--algorithm=dqn", "--config=x", "--supervised-checkpoint=y", "--run-dir=z"}, &stderr); err == nil {
		t.Fatal("parseConfig accepted unsupported generic RL algorithm")
	}
}

package experiment

import (
	"os"
	"path/filepath"
	"testing"
)

func TestCommittedPilotConfigHasConservativeDefaults(t *testing.T) {
	path := filepath.Join("..", "..", "..", "configs", "rl", "ppo-pilot-v1.json")
	config, err := LoadConfig(path)
	if err != nil {
		t.Fatal(err)
	}
	if config.Gamma != 1 || config.GAELambda != 0.95 || config.ClipRange != 0.10 || config.PPOEpochs != 4 || config.MinibatchSize != 256 {
		t.Fatalf("unexpected PPO defaults: %+v", config)
	}
	if config.ActorLearningRate != 1e-5 || config.CriticLearningRate != 1e-4 || config.ValueLossCoefficient != 0.5 || config.EntropyCoefficient != 0.001 || config.MaximumGradientNorm != 1 || config.TargetOldPolicyKL != 0.01 {
		t.Fatalf("unexpected optimiser defaults: %+v", config)
	}
	if config.RolloutGames != 1024 || config.WarmupGames < 2000 || config.PilotSolutions != 512 || config.GamesPerPilotSolution != 2 || config.PilotIterations < 1 || config.PilotIterations > 3 {
		t.Fatalf("pilot is not bounded as declared: %+v", config)
	}
}

func TestLoadConfigRejectsUnknownAndOpenEndedConfig(t *testing.T) {
	path := filepath.Join(t.TempDir(), "bad.json")
	if err := os.WriteFile(path, []byte(`{"schema_version":1,"algorithm":"ppo","unknown":true}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadConfig(path); err == nil {
		t.Fatal("LoadConfig accepted unknown field")
	}

	config, err := LoadConfig(filepath.Join("..", "..", "..", "configs", "rl", "ppo-pilot-v1.json"))
	if err != nil {
		t.Fatal(err)
	}
	config.PilotIterations = 100
	if err := config.Validate(); err == nil {
		t.Fatal("Validate accepted open-ended iteration count")
	}
}

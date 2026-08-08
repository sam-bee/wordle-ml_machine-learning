package proofrun

import (
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/sam-bee/wordle-ml_machine-learning/runstate"
)

func TestFixedStageConfigs(t *testing.T) {
	for stage, want := range map[Stage]struct {
		batch  int
		lr     float64
		target int64
	}{
		Overfit: {128, 1e-3, 400},
		Mini:    {128, 3e-4, 1000},
		Full:    {256, 3e-4, 2000},
	} {
		config, err := ConfigFor(stage)
		if err != nil {
			t.Fatalf("ConfigFor(%q): %v", stage, err)
		}
		if config.BatchSize != want.batch || config.LearningRate != want.lr || config.TargetUpdates != want.target {
			t.Errorf("ConfigFor(%q) = %#v", stage, config)
		}
		if config.ValidationEvery != 100 || config.CheckpointEvery != 100 || config.ScalarEvery != 10 || config.Seed != Seed {
			t.Errorf("ConfigFor(%q) cadence/seed = %#v", stage, config)
		}
	}
	if _, err := ConfigFor("unknown"); err == nil {
		t.Fatal("ConfigFor accepted unknown stage")
	}
}

func TestValidationBatchSizeIsExactForFrozenSplit(t *testing.T) {
	if validationBatchSize != 100 || 2500%validationBatchSize != 0 {
		t.Fatalf("validation batch size %d does not divide 2,500 records", validationBatchSize)
	}
}

func TestResumeStateIsAbsoluteAndSeedBound(t *testing.T) {
	mini, err := ConfigFor(Mini)
	if err != nil {
		t.Fatal(err)
	}
	for name, state := range map[string]runstate.State{
		"valid at stop": {GlobalUpdate: 500, ShuffleSeed: Seed, DatasetEpoch: 1, ShuffledCursor: 63},
		"valid target":  {GlobalUpdate: 1000, ShuffleSeed: Seed},
	} {
		t.Run(name, func(t *testing.T) {
			if err := ValidateResumeState(mini, state); err != nil {
				t.Fatalf("ValidateResumeState: %v", err)
			}
		})
	}
	for name, state := range map[string]runstate.State{
		"wrong seed":    {GlobalUpdate: 500, ShuffleSeed: Seed + 1},
		"beyond target": {GlobalUpdate: 1001, ShuffleSeed: Seed},
	} {
		t.Run(name, func(t *testing.T) {
			if err := ValidateResumeState(mini, state); err == nil {
				t.Fatal("ValidateResumeState unexpectedly succeeded")
			}
		})
	}
}

func TestStopAtAndImmutableConfig(t *testing.T) {
	mini, err := ConfigFor(Mini)
	if err != nil {
		t.Fatal(err)
	}
	if err := ValidateStopAt(mini, 500); err != nil {
		t.Fatalf("valid mini stop: %v", err)
	}
	for _, stop := range []int64{1, 100, 600} {
		if err := ValidateStopAt(mini, stop); err == nil {
			t.Errorf("ValidateStopAt accepted mini stop %d", stop)
		}
	}
	overfit, _ := ConfigFor(Overfit)
	if err := ValidateStopAt(overfit, 500); err == nil {
		t.Error("ValidateStopAt accepted overfit stop")
	}
	if err := validateStopAtForRun(mini, true, 500); err == nil {
		t.Error("validateStopAtForRun accepted repeated mini stop")
	}
	if err := validateStopAtForRun(mini, true, 0); err != nil {
		t.Fatalf("validateStopAtForRun accepted normal resume: %v", err)
	}
	if err := validateStopAtForRun(mini, false, 0); err == nil {
		t.Error("validateStopAtForRun accepted a fresh mini run without the required update-500 stop")
	}
	if err := validateStopAtForRun(mini, false, 500); err != nil {
		t.Fatalf("validateStopAtForRun rejected the required fresh mini stop: %v", err)
	}

	layout, err := runstate.Create(filepath.Join(t.TempDir(), "runs"), "mini-proof")
	if err != nil {
		t.Fatal(err)
	}
	if err := WriteOrValidateConfig(layout, mini, false); err != nil {
		t.Fatalf("write config: %v", err)
	}
	if err := WriteOrValidateConfig(layout, mini, true); err != nil {
		t.Fatalf("validate same config: %v", err)
	}
	if err := WriteOrValidateConfig(layout, overfit, true); err == nil {
		t.Fatal("resume accepted mismatched config")
	}
}

func TestInvalidFreshMiniInvocationDoesNotCreateRunDirectory(t *testing.T) {
	mini, err := ConfigFor(Mini)
	if err != nil {
		t.Fatal(err)
	}
	runsDir := filepath.Join(t.TempDir(), "runs")
	if _, _, err := prepareLayoutForRun(runsDir, "mini-must-stop", mini, 0); err == nil {
		t.Fatal("fresh mini run without update-500 stop was accepted")
	}
	layout, err := runstate.New(runsDir, "mini-must-stop")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(layout.Dir); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("invalid invocation left run directory behind: %v", err)
	}
}

func TestCheckpointCadencesMustMatch(t *testing.T) {
	config, err := ConfigFor(Mini)
	if err != nil {
		t.Fatal(err)
	}
	config.CheckpointEvery++
	if err := validateCheckpointCadence(config); err == nil {
		t.Fatal("mismatched validation/checkpoint cadence was accepted")
	}
}

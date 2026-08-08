// Package proofrun implements the deliberately small fixed supervised-training
// proof stages. It has no tuning surface: stage names select the plan's fixed
// configuration.
package proofrun

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"

	"github.com/sam-bee/wordle-ml_machine-learning/runstate"
)

const (
	// Seed is the one fixed seed used by all proof stages.
	Seed int64 = 20260808

	validationEvery = int64(100)
	checkpointEvery = int64(100)
	scalarEvery     = int64(10)
)

// Stage is a fixed proof gate from the supervised-training plan.
type Stage string

const (
	Overfit Stage = "overfit"
	Mini    Stage = "mini"
	Full    Stage = "full"
)

// Config is recorded verbatim in config.json and is intentionally limited to
// the plan's fixed choices.
type Config struct {
	Stage            Stage   `json:"stage"`
	BatchSize        int     `json:"batch_size"`
	LearningRate     float64 `json:"learning_rate"`
	TargetUpdates    int64   `json:"target_updates"`
	ValidationEvery  int64   `json:"validation_every"`
	CheckpointEvery  int64   `json:"checkpoint_every"`
	ScalarEvery      int64   `json:"scalar_every"`
	Seed             int64   `json:"seed"`
	Precision        string  `json:"precision"`
	Objective        string  `json:"objective"`
	Optimizer        string  `json:"optimizer"`
	LearningRateMode string  `json:"learning_rate_schedule"`
	WeightDecay      float64 `json:"weight_decay"`
	GradientClipNorm float64 `json:"gradient_clip_global_norm"`
}

// ConfigFor returns the complete, fixed configuration for stage.
func ConfigFor(stage Stage) (Config, error) {
	config := Config{
		Stage:            stage,
		ValidationEvery:  validationEvery,
		CheckpointEvery:  checkpointEvery,
		ScalarEvery:      scalarEvery,
		Seed:             Seed,
		Precision:        "float32",
		Objective:        "masked_sparse_cross_entropy_teacher_top1",
		Optimizer:        "Adam",
		LearningRateMode: "constant",
		WeightDecay:      0,
		GradientClipNorm: 5,
	}
	switch stage {
	case Overfit:
		config.BatchSize = 128
		config.LearningRate = 1e-3
		config.TargetUpdates = 400
	case Mini:
		config.BatchSize = 128
		config.LearningRate = 3e-4
		config.TargetUpdates = 1000
	case Full:
		config.BatchSize = 256
		config.LearningRate = 3e-4
		config.TargetUpdates = 2000
	default:
		return Config{}, fmt.Errorf("unknown proof stage %q", stage)
	}
	if err := validateCheckpointCadence(config); err != nil {
		return Config{}, err
	}
	return config, nil
}

func validateCheckpointCadence(config Config) error {
	if config.ValidationEvery <= 0 || config.CheckpointEvery <= 0 {
		return errors.New("validation and checkpoint cadences must be positive")
	}
	if config.ValidationEvery != config.CheckpointEvery {
		return fmt.Errorf("validation cadence %d must equal checkpoint cadence %d", config.ValidationEvery, config.CheckpointEvery)
	}
	return nil
}

// ValidateStopAt accepts only the explicit normal mini-stage stop at a
// checkpoint boundary. A zero stop means run to the stage target.
func ValidateStopAt(config Config, stopAt int64) error {
	if stopAt == 0 {
		return nil
	}
	if config.Stage != Mini || stopAt != 500 {
		return fmt.Errorf("-stop-at is only supported as -stop-at=500 for the mini proof stage")
	}
	if err := validateCheckpointCadence(config); err != nil {
		return err
	}
	if stopAt%config.ValidationEvery != 0 || stopAt%config.CheckpointEvery != 0 || stopAt >= config.TargetUpdates {
		return fmt.Errorf("stop update %d is not a valid mini checkpoint boundary", stopAt)
	}
	return nil
}

// validateStopAtForRun rejects a second request to stop an already-resumed
// run. A 500-update stop is a one-time proof artifact, not a repeatable mode.
func validateStopAtForRun(config Config, resumed bool, stopAt int64) error {
	if err := ValidateStopAt(config, stopAt); err != nil {
		return err
	}
	if config.Stage == Mini && !resumed && stopAt != 500 {
		return errors.New("a fresh mini proof run must stop normally at update 500 before it can resume to 1000")
	}
	if resumed && stopAt != 0 {
		return errors.New("-stop-at cannot be used when resuming an existing run")
	}
	return nil
}

// ValidateResumeState verifies the application state before any resumed batch
// is materialized. Update targets are absolute rather than per-process.
func ValidateResumeState(config Config, state runstate.State) error {
	if err := state.Validate(); err != nil {
		return err
	}
	if state.ShuffleSeed != config.Seed {
		return fmt.Errorf("checkpoint shuffle seed %d differs from config seed %d", state.ShuffleSeed, config.Seed)
	}
	if state.GlobalUpdate > config.TargetUpdates {
		return fmt.Errorf("checkpoint update %d exceeds stage target %d", state.GlobalUpdate, config.TargetUpdates)
	}
	return nil
}

// WriteOrValidateConfig writes config for a newly-created run. For a resumed
// run it rejects any mismatch before the model or data are opened.
func WriteOrValidateConfig(layout runstate.Layout, config Config, resumed bool) error {
	if !resumed {
		return layout.WriteConfig(config)
	}
	contents, err := os.ReadFile(layout.ConfigPath)
	if err != nil {
		return fmt.Errorf("read immutable run config %q: %w", layout.ConfigPath, err)
	}
	var stored Config
	if err := json.Unmarshal(contents, &stored); err != nil {
		return fmt.Errorf("decode immutable run config %q: %w", layout.ConfigPath, err)
	}
	if stored != config {
		return errors.New("requested stage configuration does not match immutable config.json")
	}
	return nil
}

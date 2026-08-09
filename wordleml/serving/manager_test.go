package serving

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/sam-bee/wordle-ml_machine-learning/gameeval"
	"github.com/sam-bee/wordle-ml_machine-learning/proofrun"
	"github.com/sam-bee/wordle-ml_machine-learning/runmetadata"
	"github.com/sam-bee/wordle-ml_machine-learning/runstate"
)

type fakeManagedRuntime struct {
	identity ModelIdentity
	game     gameeval.GameResult
	closed   bool
}

func (runtime *fakeManagedRuntime) ModelIdentity() ModelIdentity { return runtime.identity }
func (runtime *fakeManagedRuntime) ValidationSolutions() []string {
	return []string{"ADEPT", "VODKA"}
}
func (runtime *fakeManagedRuntime) Play(_ context.Context, solution string) (gameeval.GameResult, error) {
	runtime.game.Solution = solution
	return runtime.game, nil
}
func (runtime *fakeManagedRuntime) Close() { runtime.closed = true }

func TestManagerSelectModelSwapsOnlyAfterSuccessfulLoad(t *testing.T) {
	first := &fakeManagedRuntime{identity: ModelIdentity{RunID: "full-1"}}
	second := &fakeManagedRuntime{identity: ModelIdentity{RunID: "production-1", Update: 2200}}
	manager := &Manager{
		options: Options{DataDir: "/data", RunsDir: "/runs", RunID: "full-1"},
		current: first,
		catalogue: func() ([]ModelSummary, error) {
			return []ModelSummary{{RunID: "full-1"}, {RunID: "production-1"}}, nil
		},
		load: func(_ context.Context, options Options) (managedRuntime, error) {
			if options.RunID != "production-1" {
				t.Fatalf("loaded run = %q", options.RunID)
			}
			return second, nil
		},
	}
	identity, err := manager.SelectModel(context.Background(), "production-1")
	if err != nil {
		t.Fatal(err)
	}
	if identity.RunID != "production-1" || !first.closed || second.closed {
		t.Fatalf("identity = %+v, first closed = %t, second closed = %t", identity, first.closed, second.closed)
	}
	game, err := manager.PlayGame(context.Background(), "VODKA")
	if err != nil {
		t.Fatal(err)
	}
	if game.Model.RunID != "production-1" || game.Solution != "VODKA" {
		t.Fatalf("game = %+v", game)
	}
}

func TestManagerKeepsCurrentModelWhenReplacementFails(t *testing.T) {
	first := &fakeManagedRuntime{identity: ModelIdentity{RunID: "full-1"}}
	manager := &Manager{
		options: Options{RunID: "full-1"},
		current: first,
		catalogue: func() ([]ModelSummary, error) {
			return []ModelSummary{{RunID: "full-1"}, {RunID: "broken"}}, nil
		},
		load: func(context.Context, Options) (managedRuntime, error) {
			return nil, errors.New("restore failed")
		},
	}
	if _, err := manager.SelectModel(context.Background(), "broken"); err == nil {
		t.Fatal("failed replacement unexpectedly selected")
	}
	if first.closed || manager.ModelIdentity().RunID != "full-1" {
		t.Fatalf("current model changed after failed load: closed = %t, identity = %+v", first.closed, manager.ModelIdentity())
	}
	if _, err := manager.SelectModel(context.Background(), "missing"); !errors.Is(err, ErrModelNotFound) {
		t.Fatalf("missing model error = %v", err)
	}
}

func TestDiscoverModelsReturnsCompletedCompatibleRuns(t *testing.T) {
	root := t.TempDir()
	dataDir := filepath.Join(root, "data")
	runsDir := filepath.Join(root, "runs")
	writeServingData(t, dataDir)
	writeServableRun(t, dataDir, runsDir, "proof-full", proofrun.Full, 2000)
	writeServableRun(t, dataDir, runsDir, "production", proofrun.Production, 2200)
	if _, err := runstate.Create(runsDir, "unfinished"); err != nil {
		t.Fatal(err)
	}

	models, err := DiscoverModels(dataDir, runsDir)
	if err != nil {
		t.Fatal(err)
	}
	if len(models) != 2 || models[0].RunID != "production" || models[1].RunID != "proof-full" {
		t.Fatalf("models = %+v", models)
	}
	if models[0].Stage != proofrun.Production || models[0].Update != 2200 || models[1].Stage != proofrun.Full || models[1].Update != 2000 {
		t.Fatalf("model summaries = %+v", models)
	}
}

func writeServingData(t *testing.T, dataDir string) {
	t.Helper()
	for _, relative := range []string{
		"imitation/wordle-validation.bin",
		"imitation/wordle-validation.json",
		"wordlist-action-space-4739.csv",
		"wordlist-valid-solutions-all-2309.csv",
		"wordlist-valid-solutions-validation-100.csv",
	} {
		path := filepath.Join(dataDir, relative)
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(relative+"\n"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
}

func writeServableRun(t *testing.T, dataDir, runsDir, runID string, stage proofrun.Stage, bestUpdate int64) {
	t.Helper()
	layout, err := runstate.Create(runsDir, runID)
	if err != nil {
		t.Fatal(err)
	}
	config, err := proofrun.ConfigFor(stage)
	if err != nil {
		t.Fatal(err)
	}
	if err := layout.WriteConfig(config); err != nil {
		t.Fatal(err)
	}
	if err := layout.WriteFinalMetricsJSON(proofrun.Result{Stage: stage, Passed: true, GlobalUpdate: config.TargetUpdates, BestValidationStep: bestUpdate}); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(layout.BestCheckpointDir, "checkpoint.bin"), []byte("checkpoint"), 0o644); err != nil {
		t.Fatal(err)
	}
	effectiveConfig, err := json.Marshal(config)
	if err != nil {
		t.Fatal(err)
	}
	artifact := func(relative string) runmetadata.Artifact {
		hash, err := runmetadata.FileSHA256(filepath.Join(dataDir, relative))
		if err != nil {
			t.Fatal(err)
		}
		return runmetadata.Artifact{Path: filepath.ToSlash(filepath.Join("data", relative)), SHA256: hash}
	}
	zeroHash := "0000000000000000000000000000000000000000000000000000000000000000"
	manifest := runmetadata.Manifest{
		SchemaVersion: runmetadata.SchemaVersion,
		CollectedAt:   time.Date(2026, 8, 9, 0, 0, 0, 0, time.UTC),
		Repositories: runmetadata.Repositories{
			MachineLearning: runmetadata.Repository{Path: ".", Commit: "training-commit"},
			SyntheticData:   runmetadata.Repository{Path: "synthetic", Commit: "synthetic-commit"},
			GameEngine:      runmetadata.Repository{Path: "game", Commit: "game-commit"},
		},
		Dataset: runmetadata.Dataset{Format: "WDIT", Version: "3", Files: []runmetadata.Artifact{
			artifact("imitation/wordle-validation.bin"), artifact("imitation/wordle-validation.json"),
		}},
		Vocabulary: []runmetadata.Artifact{
			artifact("wordlist-action-space-4739.csv"), artifact("wordlist-valid-solutions-all-2309.csv"),
		},
		Splits: runmetadata.Splits{
			Training:   []runmetadata.Artifact{{Path: "data/train.csv", SHA256: zeroHash}},
			Validation: []runmetadata.Artifact{artifact("wordlist-valid-solutions-validation-100.csv")},
			Test:       []runmetadata.Artifact{{Path: "data/test.csv", SHA256: zeroHash}},
		},
		ModelParameterCount: 1,
		Runtime: runmetadata.RuntimeMetadata{
			GoVersion: "go1.26", GOOS: "linux", GOARCH: "amd64", GoMLXVersion: "v0", Backend: "xla:cuda",
			GPUDetails: map[string]string{"name": "test"}, CUDADetails: map[string]string{"version": "test"},
			PJRTDetails: map[string]string{"version": "test"}, GoMLXDetails: map[string]string{"backend_name": "xla"},
		},
		Seed: config.Seed, EffectiveConfig: effectiveConfig,
	}
	if err := manifest.Validate(); err != nil {
		t.Fatal(err)
	}
	if err := layout.WriteMetadata(manifest); err != nil {
		t.Fatal(err)
	}
}

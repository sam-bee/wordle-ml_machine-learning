// Package serving loads immutable training checkpoints and exposes the
// production game path through a small, serialized inference runtime.
package serving

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"
	"sync"

	"github.com/gomlx/compute"
	"github.com/sam-bee/wordle-ml_machine-learning/gameeval"
	"github.com/sam-bee/wordle-ml_machine-learning/policy"
	"github.com/sam-bee/wordle-ml_machine-learning/proofeval"
	"github.com/sam-bee/wordle-ml_machine-learning/proofgames"
	"github.com/sam-bee/wordle-ml_machine-learning/proofrun"
	"github.com/sam-bee/wordle-ml_machine-learning/runmetadata"
	"github.com/sam-bee/wordle-ml_machine-learning/runstate"
	"github.com/sam-bee/wordle-ml_machine-learning/supervised"
	"github.com/sam-bee/wordle-ml_machine-learning/vocabulary"
)

// ErrInvalidSolution means a request did not name one of the fixed validation
// solutions. The final-test split deliberately remains unavailable.
var ErrInvalidSolution = errors.New("solution is not an allowed validation word")

// Options identifies one self-contained training run to serve.
type Options struct {
	DataDir string
	RunsDir string
	RunID   string
}

// ModelIdentity is returned with every game so a result remains attributable
// to the exact checkpoint and training source revision that produced it.
type ModelIdentity struct {
	RunID               string         `json:"run_id"`
	Stage               proofrun.Stage `json:"stage"`
	Checkpoint          string         `json:"checkpoint"`
	Update              int64          `json:"update"`
	TrainingCommit      string         `json:"training_commit"`
	ValidationSplitHash string         `json:"validation_split_hash"`
}

// Player is the narrow interface the model manager uses for one loaded run.
type Player interface {
	ModelIdentity() ModelIdentity
	ValidationSolutions() []string
	Play(context.Context, string) (gameeval.GameResult, error)
}

// Runtime owns one warm CUDA backend, restored session, and game evaluator.
// The channel is a context-aware one-request gate: GoMLX executors and Store
// variables are deliberately not used concurrently.
type Runtime struct {
	backend   compute.Backend
	session   *supervised.Session
	evaluator *gameeval.Evaluator
	identity  ModelIdentity
	solutions []string
	allowed   map[string]struct{}
	gate      chan struct{}
	closeOnce sync.Once
}

// Load validates the immutable run inputs, restores its best checkpoint,
// warms inference with one validation game, and returns a ready runtime.
// Unlike proof evaluation, serving does not require the current Git HEAD to
// equal the recorded training commit; compatible data, model shape, runtime,
// and checkpoint state are verified instead.
func Load(ctx context.Context, options Options) (*Runtime, error) {
	if strings.TrimSpace(options.DataDir) == "" || strings.TrimSpace(options.RunsDir) == "" {
		return nil, errors.New("data and runs directories are required")
	}
	if strings.TrimSpace(options.RunID) == "" {
		return nil, errors.New("run ID is required")
	}
	layout, err := runstate.Open(options.RunsDir, options.RunID)
	if err != nil {
		return nil, fmt.Errorf("open inference run: %w", err)
	}
	config, err := proofeval.ReadConfig(layout)
	if err != nil {
		return nil, err
	}
	final, err := readPassedResult(layout, config)
	if err != nil {
		return nil, err
	}
	manifest, err := readCompatibleManifest(layout, options.DataDir, config)
	if err != nil {
		return nil, err
	}
	vocab, err := vocabulary.Load(options.DataDir)
	if err != nil {
		return nil, fmt.Errorf("load serving vocabulary: %w", err)
	}

	backend, err := compute.NewWithConfig("xla:cuda")
	if err != nil {
		return nil, fmt.Errorf("create xla:cuda serving backend: %w", err)
	}
	backendOwnedByRuntime := false
	defer func() {
		if !backendOwnedByRuntime {
			backend.Finalize()
		}
	}()
	if err := runmetadata.VerifyEvaluationRuntime(manifest, backend.Name(), backend.Description()); err != nil {
		return nil, fmt.Errorf("verify serving runtime identity: %w", err)
	}
	session, state, err := proofeval.LoadSession(backend, layout, proofeval.Best, config)
	if err != nil {
		return nil, err
	}
	runtime := &Runtime{
		backend:   backend,
		session:   session,
		solutions: vocab.Validation(),
		allowed:   make(map[string]struct{}, vocabulary.NumValidationSolutions),
		gate:      make(chan struct{}, 1),
		identity: ModelIdentity{
			RunID: options.RunID, Stage: config.Stage, Checkpoint: string(proofeval.Best), Update: state.GlobalUpdate,
			TrainingCommit: manifest.Repositories.MachineLearning.Commit, ValidationSplitHash: vocab.Hashes().Validation,
		},
	}
	backendOwnedByRuntime = true
	loaded := false
	defer func() {
		if !loaded {
			runtime.Close()
		}
	}()
	for _, solution := range runtime.solutions {
		runtime.allowed[solution] = struct{}{}
	}
	if final.BestValidationStep != state.GlobalUpdate {
		return nil, fmt.Errorf("best checkpoint update %d differs from final metrics %d", state.GlobalUpdate, final.BestValidationStep)
	}
	if state.BestValidation == nil || state.BestValidation.Update != state.GlobalUpdate || state.BestValidation.Value != final.BestValidation.Loss {
		return nil, errors.New("best checkpoint validation state differs from final metrics")
	}
	scorer, err := proofgames.SessionScorer(session)
	if err != nil {
		return nil, err
	}
	runtime.evaluator, err = gameeval.New(gameeval.Config{Vocabulary: vocab, Score: scorer})
	if err != nil {
		return nil, fmt.Errorf("create serving game evaluator: %w", err)
	}
	if _, err := runtime.play(ctx, runtime.solutions[0]); err != nil {
		return nil, fmt.Errorf("warm serving inference: %w", err)
	}
	if count := int64(policy.TrainableParameterCount(session.Store.RootScope())); count != manifest.ModelParameterCount {
		return nil, fmt.Errorf("materialized model has %d parameters, immutable manifest records %d", count, manifest.ModelParameterCount)
	}
	loaded = true
	return runtime, nil
}

// ModelIdentity returns an immutable copy of the served checkpoint identity.
func (runtime *Runtime) ModelIdentity() ModelIdentity { return runtime.identity }

// ValidationSolutions returns the only solutions accepted by Play.
func (runtime *Runtime) ValidationSolutions() []string {
	return append([]string(nil), runtime.solutions...)
}

// Play normalizes and validates one solution, then runs the complete game while
// holding the runtime's one-request inference gate.
func (runtime *Runtime) Play(ctx context.Context, solution string) (gameeval.GameResult, error) {
	if runtime == nil || runtime.evaluator == nil {
		return gameeval.GameResult{}, errors.New("inference runtime is not ready")
	}
	solution = strings.ToUpper(strings.TrimSpace(solution))
	if _, ok := runtime.allowed[solution]; !ok {
		return gameeval.GameResult{}, ErrInvalidSolution
	}
	select {
	case runtime.gate <- struct{}{}:
		defer func() { <-runtime.gate }()
	case <-ctx.Done():
		return gameeval.GameResult{}, ctx.Err()
	}
	return runtime.play(ctx, solution)
}

func (runtime *Runtime) play(ctx context.Context, solution string) (gameeval.GameResult, error) {
	evaluation, err := runtime.evaluator.Evaluate(ctx, []string{solution})
	if err != nil {
		return gameeval.GameResult{}, err
	}
	if len(evaluation.Games) != 1 {
		return gameeval.GameResult{}, fmt.Errorf("single-game inference returned %d games", len(evaluation.Games))
	}
	return evaluation.Games[0], nil
}

// Close releases the restored session and CUDA backend. It is idempotent and
// must be called only when no inference request is using this runtime.
func (runtime *Runtime) Close() {
	if runtime == nil {
		return
	}
	runtime.closeOnce.Do(func() {
		if runtime.session != nil {
			runtime.session.Finalize()
			runtime.session = nil
		}
		if runtime.backend != nil {
			runtime.backend.Finalize()
			runtime.backend = nil
		}
		runtime.evaluator = nil
	})
}

func readPassedResult(layout runstate.Layout, config proofrun.Config) (proofrun.Result, error) {
	contents, err := os.ReadFile(layout.FinalMetricsPath)
	if err != nil {
		return proofrun.Result{}, fmt.Errorf("read final inference metrics: %w", err)
	}
	var result proofrun.Result
	if err := json.Unmarshal(contents, &result); err != nil {
		return proofrun.Result{}, fmt.Errorf("decode final inference metrics: %w", err)
	}
	if result.Stage != config.Stage || !result.Passed || result.GlobalUpdate != config.TargetUpdates || result.BestValidationStep < 0 || result.BestValidationStep > result.GlobalUpdate {
		return proofrun.Result{}, fmt.Errorf("inference run has not completed its %q gate", config.Stage)
	}
	return result, nil
}

func readCompatibleManifest(layout runstate.Layout, dataDir string, config proofrun.Config) (runmetadata.Manifest, error) {
	contents, err := os.ReadFile(layout.MetadataPath)
	if err != nil {
		return runmetadata.Manifest{}, fmt.Errorf("read inference metadata: %w", err)
	}
	var manifest runmetadata.Manifest
	if err := json.Unmarshal(contents, &manifest); err != nil {
		return runmetadata.Manifest{}, fmt.Errorf("decode inference metadata: %w", err)
	}
	if err := runmetadata.VerifyEvaluationInputs(manifest, dataDir); err != nil {
		return runmetadata.Manifest{}, fmt.Errorf("verify inference data identity: %w", err)
	}
	var effective proofrun.Config
	if err := json.Unmarshal(manifest.EffectiveConfig, &effective); err != nil {
		return runmetadata.Manifest{}, fmt.Errorf("decode inference effective config: %w", err)
	}
	if effective != config {
		return runmetadata.Manifest{}, errors.New("run config differs from the immutable metadata effective config")
	}
	return manifest, nil
}

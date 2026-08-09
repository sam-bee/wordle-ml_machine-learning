// Package cudaweb joins the CUDA scorer to the existing Wordle game evaluator
// and a small same-process browser API.
//
// It deliberately knows nothing about GoMLX, checkpoint restoration, or cgo.
// The command supplies an already validated, ready CUDA scorer.
package cudaweb

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"

	"github.com/sam-bee/wordle-ml_machine-learning/gameeval"
	"github.com/sam-bee/wordle-ml_machine-learning/inferenceapi"
	"github.com/sam-bee/wordle-ml_machine-learning/modelstate"
	"github.com/sam-bee/wordle-ml_machine-learning/vocabulary"
)

var (
	// ErrInvalidSolution means a request named a word outside the fixed
	// validation population. The final-test vocabulary is never exposed here.
	ErrInvalidSolution = inferenceapi.ErrInvalidSolution
	// ErrModelNotFound means a request tried to select something other than the
	// one CUDA model loaded by this process.
	ErrModelNotFound = inferenceapi.ErrModelNotFound
)

// Scorer is the small part of cudainfer.Backend used by the game layer. Score
// returns all raw, finite action logits in canonical action-ID order.
type Scorer interface {
	Score(context.Context, modelstate.Inputs) ([]float32, error)
}

// Model describes the one immutable exported model served by this process.
// The fields are deliberately also the API response so a demo screenshot can
// identify both weights and the actual CUDA device used for inference.
type Model struct {
	Backend            string `json:"backend"`
	ModelFormat        string `json:"model_format"`
	RunID              string `json:"run_id"`
	Stage              string `json:"stage,omitempty"`
	Checkpoint         string `json:"checkpoint"`
	Update             int64  `json:"update"`
	TrainingCommit     string `json:"training_commit"`
	WeightsSHA256      string `json:"weights_sha256"`
	ParameterCount     int    `json:"parameter_count"`
	DeviceName         string `json:"device_name"`
	ComputeCapability  string `json:"compute_capability"`
	CUDARuntimeVersion string `json:"cuda_runtime_version"`
	CUDADriverVersion  string `json:"cuda_driver_version"`
}

// Options supplies the already-loaded CUDA scorer and the vocabulary used to
// encode and play validation-only games.
type Options struct {
	Vocabulary *vocabulary.Vocabulary
	Scorer     Scorer
	Model      Model
}

// GameResponse records the model identity alongside one complete, authoritative
// Wordle trajectory.
type GameResponse = inferenceapi.GameResponse

// Service owns the fixed model's evaluator. gate keeps the entire game
// together: concurrent browser requests cannot interleave turns, even though
// their individual CUDA scores are also serialized by the backend worker.
type Service struct {
	scorer    Scorer
	evaluator *gameeval.Evaluator
	model     Model
	solutions []string
	allowed   map[string]struct{}
	gate      chan struct{}

	closeOnce sync.Once
}

// New creates a ready one-model service. The vocabulary must have been loaded
// without the final-test split by the caller.
func New(options Options) (*Service, error) {
	if options.Vocabulary == nil {
		return nil, errors.New("vocabulary is required")
	}
	if len(options.Vocabulary.Test()) != 0 || options.Vocabulary.Hashes().Test != "" {
		return nil, errors.New("CUDA web vocabulary must be loaded without the final-test split")
	}
	if options.Scorer == nil {
		return nil, errors.New("CUDA scorer is required")
	}
	if strings.TrimSpace(options.Model.RunID) == "" || strings.TrimSpace(options.Model.Checkpoint) == "" {
		return nil, errors.New("model run ID and checkpoint are required")
	}
	if options.Model.Update < 0 {
		return nil, fmt.Errorf("model update must not be negative: %d", options.Model.Update)
	}
	if options.Model.Backend == "" {
		options.Model.Backend = "cuda-cgo"
	}

	service := &Service{
		scorer:    options.Scorer,
		model:     options.Model,
		solutions: options.Vocabulary.Validation(),
		allowed:   make(map[string]struct{}, vocabulary.NumValidationSolutions),
		gate:      make(chan struct{}, 1),
	}
	if len(service.solutions) != vocabulary.NumValidationSolutions {
		return nil, fmt.Errorf("validation population has %d words, want %d", len(service.solutions), vocabulary.NumValidationSolutions)
	}
	for _, solution := range service.solutions {
		service.allowed[solution] = struct{}{}
	}

	var err error
	service.evaluator, err = gameeval.New(gameeval.Config{
		Vocabulary: options.Vocabulary,
		Score: func(ctx context.Context, position gameeval.Position) ([]float32, error) {
			return service.scorer.Score(ctx, position.Inputs)
		},
	})
	if err != nil {
		return nil, fmt.Errorf("create CUDA game evaluator: %w", err)
	}
	return service, nil
}

// Model returns the immutable identity of this process's sole model.
func (service *Service) Model() Model {
	if service == nil {
		return Model{}
	}
	return service.model
}

// ModelIdentity returns the public identity of the singleton model.
func (service *Service) ModelIdentity() inferenceapi.ModelIdentity {
	if service == nil {
		return inferenceapi.ModelIdentity{}
	}
	return inferenceapi.ModelIdentity{
		RunID:          service.model.RunID,
		Stage:          service.model.Stage,
		Checkpoint:     service.model.Checkpoint,
		Update:         service.model.Update,
		TrainingCommit: service.model.TrainingCommit,
	}
}

// AvailableModels returns the singleton catalogue consumed by the existing
// browser UI. There is no discovery or hot swapping in this first demo.
func (service *Service) AvailableModels() ([]inferenceapi.ModelSummary, error) {
	if service == nil {
		return nil, errors.New("CUDA web service is not ready")
	}
	return []inferenceapi.ModelSummary{{
		RunID:          service.model.RunID,
		Stage:          service.model.Stage,
		Checkpoint:     service.model.Checkpoint,
		Update:         service.model.Update,
		TrainingCommit: service.model.TrainingCommit,
	}}, nil
}

// ValidationSolutions returns only the frozen validation population.
func (service *Service) ValidationSolutions() []string {
	if service == nil {
		return nil
	}
	return append([]string(nil), service.solutions...)
}

// SelectModel accepts the active model idempotently and rejects all others.
// CUDA model hot-swapping is intentionally outside this first demo's scope.
func (service *Service) SelectModel(_ context.Context, runID string) (inferenceapi.ModelIdentity, error) {
	if service == nil {
		return inferenceapi.ModelIdentity{}, errors.New("CUDA web service is not ready")
	}
	if runID != service.model.RunID {
		return inferenceapi.ModelIdentity{}, fmt.Errorf("run %q: %w", runID, ErrModelNotFound)
	}
	return service.ModelIdentity(), nil
}

// Play validates one hidden validation word then runs its full six-turn game
// while retaining a single game-level gate.
func (service *Service) PlayGame(ctx context.Context, solution string) (GameResponse, error) {
	if service == nil || service.evaluator == nil {
		return GameResponse{}, errors.New("CUDA web service is not ready")
	}
	solution = strings.ToUpper(strings.TrimSpace(solution))
	if _, ok := service.allowed[solution]; !ok {
		return GameResponse{}, ErrInvalidSolution
	}
	select {
	case service.gate <- struct{}{}:
		defer func() { <-service.gate }()
	case <-ctx.Done():
		return GameResponse{}, ctx.Err()
	}

	evaluation, err := service.evaluator.Evaluate(ctx, []string{solution})
	if err != nil {
		return GameResponse{}, fmt.Errorf("play validation game: %w", err)
	}
	if len(evaluation.Games) != 1 {
		return GameResponse{}, fmt.Errorf("single-game CUDA inference returned %d games", len(evaluation.Games))
	}
	return GameResponse{Model: service.ModelIdentity(), GameResult: evaluation.Games[0]}, nil
}

// RuntimeInfo identifies the CUDA boundary, artifact, and device. The shared
// direct HTTP API includes it with health and model responses.
func (service *Service) RuntimeInfo() inferenceapi.RuntimeInfo {
	if service == nil {
		return inferenceapi.RuntimeInfo{}
	}
	model := service.model
	return inferenceapi.RuntimeInfo{
		Backend:          model.Backend,
		ModelFormat:      model.ModelFormat,
		RunID:            model.RunID,
		Checkpoint:       model.Checkpoint,
		CheckpointUpdate: model.Update,
		TrainingCommit:   model.TrainingCommit,
		WeightsSHA256:    model.WeightsSHA256,
		ParameterCount:   int64(model.ParameterCount),
		Device: &inferenceapi.DeviceInfo{
			Name:               model.DeviceName,
			ComputeCapability:  model.ComputeCapability,
			CUDARuntimeVersion: model.CUDARuntimeVersion,
			DriverVersion:      model.CUDADriverVersion,
		},
	}
}

// Close makes future lifecycle expansion safe without owning the native
// backend: the command remains responsible for closing its cgo worker after
// HTTP shutdown. It is intentionally idempotent.
func (service *Service) Close() error {
	if service == nil {
		return nil
	}
	service.closeOnce.Do(func() {
		service.evaluator = nil
	})
	return nil
}

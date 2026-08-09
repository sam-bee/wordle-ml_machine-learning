package serving

import (
	"context"
	"net/http"

	"github.com/sam-bee/wordle-ml_machine-learning/gameeval"
	"github.com/sam-bee/wordle-ml_machine-learning/inferenceapi"
)

// GameResponse is a complete, attributable Wordle trajectory.
//
// It remains a serving type so existing callers retain the proofrun.Stage
// field type in ModelIdentity. inferenceapi owns the GoMLX-free wire format.
type GameResponse struct {
	Model ModelIdentity `json:"model"`
	gameeval.GameResult
}

// Service is the historical model-selectable inference surface used by the
// GoMLX serving runtime. New inference binaries should implement
// inferenceapi.Service directly.
type Service interface {
	ModelIdentity() ModelIdentity
	ValidationSolutions() []string
	PlayGame(context.Context, string) (GameResponse, error)
	AvailableModels() ([]ModelSummary, error)
	SelectModel(context.Context, string) (ModelIdentity, error)
}

// NewHandler preserves the existing /v1 serving API while delegating request
// validation, JSON responses, and error mapping to inferenceapi.
func NewHandler(service Service) (http.Handler, error) {
	if service == nil {
		return inferenceapi.NewHandlerWithOptions(nil, inferenceapi.HandlerOptions{Prefix: "/v1"})
	}
	return inferenceapi.NewHandlerWithOptions(servingServiceAdapter{service: service}, inferenceapi.HandlerOptions{Prefix: "/v1"})
}

type servingServiceAdapter struct {
	service Service
}

func (adapter servingServiceAdapter) ModelIdentity() inferenceapi.ModelIdentity {
	return apiModelIdentity(adapter.service.ModelIdentity())
}

func (adapter servingServiceAdapter) PlayableSolutions() []string {
	return adapter.service.ValidationSolutions()
}

func (adapter servingServiceAdapter) PlayGame(ctx context.Context, solution string) (inferenceapi.GameResponse, error) {
	game, err := adapter.service.PlayGame(ctx, solution)
	return inferenceapi.GameResponse{Model: apiModelIdentity(game.Model), GameResult: game.GameResult}, err
}

func (adapter servingServiceAdapter) AvailableModels() ([]inferenceapi.ModelSummary, error) {
	models, err := adapter.service.AvailableModels()
	if err != nil {
		return nil, err
	}
	converted := make([]inferenceapi.ModelSummary, len(models))
	for index, model := range models {
		converted[index] = inferenceapi.ModelSummary{
			RunID:          model.RunID,
			Stage:          string(model.Stage),
			Checkpoint:     model.Checkpoint,
			Update:         model.Update,
			TrainingCommit: model.TrainingCommit,
		}
	}
	return converted, nil
}

func (adapter servingServiceAdapter) SelectModel(ctx context.Context, runID string) (inferenceapi.ModelIdentity, error) {
	model, err := adapter.service.SelectModel(ctx, runID)
	return apiModelIdentity(model), err
}

func apiModelIdentity(model ModelIdentity) inferenceapi.ModelIdentity {
	return inferenceapi.ModelIdentity{
		RunID:               model.RunID,
		Stage:               string(model.Stage),
		Checkpoint:          model.Checkpoint,
		Update:              model.Update,
		TrainingCommit:      model.TrainingCommit,
		ValidationSplitHash: model.ValidationSplitHash,
	}
}

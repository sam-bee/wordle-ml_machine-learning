package inferenceapi

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/sam-bee/wordle-ml_machine-learning/gameeval"
)

type singletonService struct {
	identity      ModelIdentity
	runtime       RuntimeInfo
	solutions     []string
	game          gameeval.GameResult
	gameErr       error
	requested     string
	selectionCall int
}

func (service *singletonService) ModelIdentity() ModelIdentity { return service.identity }

func (service *singletonService) ValidationSolutions() []string {
	return append([]string(nil), service.solutions...)
}

func (service *singletonService) PlayGame(_ context.Context, solution string) (GameResponse, error) {
	service.requested = solution
	if service.gameErr != nil {
		return GameResponse{}, service.gameErr
	}
	return GameResponse{Model: service.identity, GameResult: service.game}, nil
}

func (service *singletonService) AvailableModels() ([]ModelSummary, error) {
	return []ModelSummary{{
		RunID:          service.identity.RunID,
		Stage:          service.identity.Stage,
		Checkpoint:     service.identity.Checkpoint,
		Update:         service.identity.Update,
		TrainingCommit: service.identity.TrainingCommit,
	}}, nil
}

func (service *singletonService) SelectModel(_ context.Context, runID string) (ModelIdentity, error) {
	service.selectionCall++
	if runID != service.identity.RunID {
		return ModelIdentity{}, ErrModelNotFound
	}
	return service.identity, nil
}

func (service *singletonService) RuntimeInfo() RuntimeInfo { return service.runtime }

func TestDirectHandlerServesRoutesAndIdempotentSingletonSelection(t *testing.T) {
	service := &singletonService{
		identity: ModelIdentity{
			RunID: "seed-replication-1", Stage: "seed-replication", Checkpoint: "best", Update: 2600,
			TrainingCommit: "training-commit", ValidationSplitHash: "validation-hash",
		},
		runtime: RuntimeInfo{
			Backend: "cuda-cgo", ModelFormat: "wordle-cuda-f32-v1", RunID: "seed-replication-1", Checkpoint: "best", CheckpointUpdate: 2600,
			TrainingCommit: "training-commit", WeightsSHA256: "weights-hash", ParameterCount: 1046596,
			Device: &DeviceInfo{Name: "NVIDIA GeForce RTX 5070 Ti", ComputeCapability: "12.0", CUDARuntimeVersion: "13.2", DriverVersion: "595.84"},
		},
		solutions: []string{"ADEPT", "VODKA"},
		game: gameeval.GameResult{
			Solution: "VODKA", Solved: true, Guesses: 2,
			Turns: []gameeval.TurnResult{{Turn: 1, RawTopActionID: 4, RawTopGuess: "ARISE", Guess: "ARISE", Feedback: "-----", ShortlistSizeBefore: 2309, ShortlistSizeAfter: 167}},
		},
	}
	handler, err := NewHandler(service)
	if err != nil {
		t.Fatal(err)
	}

	health := httptest.NewRecorder()
	handler.ServeHTTP(health, httptest.NewRequest(http.MethodGet, "/healthz", nil))
	if health.Code != http.StatusOK {
		t.Fatalf("health status = %d, body = %s", health.Code, health.Body.String())
	}
	var healthPayload struct {
		Status  string        `json:"status"`
		Model   ModelIdentity `json:"model"`
		Runtime RuntimeInfo   `json:"runtime"`
	}
	if err := json.Unmarshal(health.Body.Bytes(), &healthPayload); err != nil {
		t.Fatal(err)
	}
	if healthPayload.Status != "ready" || healthPayload.Model.RunID != service.identity.RunID || healthPayload.Runtime.Backend != "cuda-cgo" || healthPayload.Runtime.Device == nil || healthPayload.Runtime.Device.ComputeCapability != "12.0" {
		t.Fatalf("health payload = %+v", healthPayload)
	}

	for _, path := range []string{"/api/models", "/api/solutions"} {
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, httptest.NewRequest(http.MethodGet, path, nil))
		if response.Code != http.StatusOK || !strings.Contains(response.Body.String(), `"backend":"cuda-cgo"`) {
			t.Fatalf("GET %s = (%d, %s)", path, response.Code, response.Body.String())
		}
	}

	gameRequest := httptest.NewRequest(http.MethodPost, "/api/games", strings.NewReader(`{"solution":"vodka"}`))
	gameRequest.Header.Set("Content-Type", "application/json")
	gameResponse := httptest.NewRecorder()
	handler.ServeHTTP(gameResponse, gameRequest)
	if gameResponse.Code != http.StatusOK || service.requested != "vodka" || !strings.Contains(gameResponse.Body.String(), `"solution":"VODKA"`) {
		t.Fatalf("game = (%d, %s), requested = %q", gameResponse.Code, gameResponse.Body.String(), service.requested)
	}

	for attempt := 0; attempt < 2; attempt++ {
		request := httptest.NewRequest(http.MethodPut, "/api/models", strings.NewReader(`{"run_id":"seed-replication-1"}`))
		request.Header.Set("Content-Type", "application/json")
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, request)
		if response.Code != http.StatusOK || !strings.Contains(response.Body.String(), `"run_id":"seed-replication-1"`) {
			t.Fatalf("selection %d = (%d, %s)", attempt, response.Code, response.Body.String())
		}
	}
	if service.selectionCall != 2 || service.ModelIdentity() != healthPayload.Model {
		t.Fatalf("singleton selection calls = %d, identity = %+v", service.selectionCall, service.ModelIdentity())
	}
}

func TestDirectHandlerMapsErrorsAndRejectsInvalidRequests(t *testing.T) {
	service := &singletonService{
		identity:  ModelIdentity{RunID: "only", Checkpoint: "best"},
		solutions: []string{"ADEPT"},
		gameErr:   ErrInvalidSolution,
	}
	handler, err := NewHandler(service)
	if err != nil {
		t.Fatal(err)
	}
	for _, test := range []struct {
		name        string
		method      string
		path        string
		contentType string
		body        string
		wantStatus  int
	}{
		{"wrong content type", http.MethodPost, "/api/games", "text/plain", `{}`, http.StatusUnsupportedMediaType},
		{"unknown field", http.MethodPost, "/api/games", "application/json", `{"solution":"ADEPT","extra":true}`, http.StatusBadRequest},
		{"invalid solution", http.MethodPost, "/api/games", "application/json", `{"solution":"XXXXX"}`, http.StatusBadRequest},
		{"unknown singleton", http.MethodPut, "/api/models", "application/json", `{"run_id":"missing"}`, http.StatusNotFound},
	} {
		t.Run(test.name, func(t *testing.T) {
			request := httptest.NewRequest(test.method, test.path, strings.NewReader(test.body))
			request.Header.Set("Content-Type", test.contentType)
			response := httptest.NewRecorder()
			handler.ServeHTTP(response, request)
			if response.Code != test.wantStatus {
				t.Fatalf("status = %d, want %d; body = %s", response.Code, test.wantStatus, response.Body.String())
			}
		})
	}

	service.gameErr = context.DeadlineExceeded
	request := httptest.NewRequest(http.MethodPost, "/api/games", strings.NewReader(`{"solution":"ADEPT"}`))
	request.Header.Set("Content-Type", "application/json")
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusGatewayTimeout {
		t.Fatalf("timeout status = %d, body = %s", response.Code, response.Body.String())
	}
}

func TestHandlerRejectsNilServiceAndInvalidPrefix(t *testing.T) {
	if _, err := NewHandler(nil); err == nil || err.Error() != "inference service is required" {
		t.Fatalf("nil service error = %v", err)
	}
	service := &singletonService{}
	if _, err := NewHandlerWithOptions(service, HandlerOptions{Prefix: "/api/"}); err == nil || !strings.Contains(err.Error(), "trailing slash") {
		t.Fatalf("invalid prefix error = %v", err)
	}
}

package serving

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/sam-bee/wordle-ml_machine-learning/gameeval"
)

type fakeService struct {
	identity  ModelIdentity
	models    []ModelSummary
	solutions []string
	game      gameeval.GameResult
	err       error
	requested string
	selected  string
}

func (service *fakeService) ModelIdentity() ModelIdentity { return service.identity }
func (service *fakeService) ValidationSolutions() []string {
	return append([]string(nil), service.solutions...)
}
func (service *fakeService) PlayGame(_ context.Context, solution string) (GameResponse, error) {
	service.requested = solution
	return GameResponse{Model: service.identity, GameResult: service.game}, service.err
}
func (service *fakeService) AvailableModels() ([]ModelSummary, error) {
	return append([]ModelSummary(nil), service.models...), service.err
}
func (service *fakeService) SelectModel(_ context.Context, runID string) (ModelIdentity, error) {
	service.selected = runID
	if service.err != nil {
		return ModelIdentity{}, service.err
	}
	service.identity.RunID = runID
	return service.identity, nil
}

func TestHandlerServesSolutionsAndGame(t *testing.T) {
	service := &fakeService{
		identity:  ModelIdentity{RunID: "full-1", Checkpoint: "best", Update: 2000, TrainingCommit: "abc", ValidationSplitHash: "def"},
		models:    []ModelSummary{{RunID: "full-1", Stage: "full", Checkpoint: "best", Update: 2000}, {RunID: "production-1", Stage: "production", Checkpoint: "best", Update: 2200}},
		solutions: []string{"ADEPT", "VODKA"},
		game: gameeval.GameResult{
			Solution: "VODKA", Solved: true, Guesses: 2,
			Turns: []gameeval.TurnResult{{Turn: 1, RawTopActionID: 7, RawTopGuess: "ARISE", Guess: "ARISE", Feedback: "-----", ShortlistSizeBefore: 2309, ShortlistSizeAfter: 167}},
		},
	}
	handler, err := NewHandler(service)
	if err != nil {
		t.Fatal(err)
	}

	solutions := httptest.NewRecorder()
	handler.ServeHTTP(solutions, httptest.NewRequest(http.MethodGet, "/v1/solutions", nil))
	if solutions.Code != http.StatusOK || !strings.Contains(solutions.Body.String(), `"VODKA"`) {
		t.Fatalf("solutions response = (%d, %s)", solutions.Code, solutions.Body.String())
	}

	request := httptest.NewRequest(http.MethodPost, "/v1/games", strings.NewReader(`{"solution":"vodka"}`))
	request.Header.Set("Content-Type", "application/json")
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", response.Code, response.Body.String())
	}
	if service.requested != "vodka" {
		t.Fatalf("requested solution = %q", service.requested)
	}
	var result struct {
		Model ModelIdentity `json:"model"`
		gameeval.GameResult
	}
	if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if result.Model.Update != 2000 || result.Solution != "VODKA" || len(result.Turns) != 1 || result.Turns[0].Feedback != "-----" {
		t.Fatalf("game response = %+v", result)
	}
	if got := response.Header().Get("Cache-Control"); got != "no-store" {
		t.Fatalf("Cache-Control = %q", got)
	}
}

func TestHandlerListsAndSelectsModels(t *testing.T) {
	service := &fakeService{
		identity: ModelIdentity{RunID: "full-1", Stage: "full", Checkpoint: "best", Update: 2000},
		models: []ModelSummary{
			{RunID: "full-1", Stage: "full", Checkpoint: "best", Update: 2000},
			{RunID: "production-1", Stage: "production", Checkpoint: "best", Update: 2200},
		},
	}
	handler, err := NewHandler(service)
	if err != nil {
		t.Fatal(err)
	}

	models := httptest.NewRecorder()
	handler.ServeHTTP(models, httptest.NewRequest(http.MethodGet, "/v1/models", nil))
	if models.Code != http.StatusOK || !strings.Contains(models.Body.String(), `"active":{"run_id":"full-1"`) || !strings.Contains(models.Body.String(), `"run_id":"production-1"`) {
		t.Fatalf("models response = (%d, %s)", models.Code, models.Body.String())
	}

	request := httptest.NewRequest(http.MethodPut, "/v1/models", strings.NewReader(`{"run_id":"production-1"}`))
	request.Header.Set("Content-Type", "application/json")
	selected := httptest.NewRecorder()
	handler.ServeHTTP(selected, request)
	if selected.Code != http.StatusOK || service.selected != "production-1" || !strings.Contains(selected.Body.String(), `"run_id":"production-1"`) {
		t.Fatalf("selection response = (%d, %s), selected = %q", selected.Code, selected.Body.String(), service.selected)
	}
}

func TestHandlerRejectsUnavailableModel(t *testing.T) {
	service := &fakeService{err: ErrModelNotFound}
	handler, err := NewHandler(service)
	if err != nil {
		t.Fatal(err)
	}
	request := httptest.NewRequest(http.MethodPut, "/v1/models", strings.NewReader(`{"run_id":"missing"}`))
	request.Header.Set("Content-Type", "application/json")
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusNotFound {
		t.Fatalf("status = %d, want %d; body = %s", response.Code, http.StatusNotFound, response.Body.String())
	}
}

func TestHandlerRejectsInvalidRequests(t *testing.T) {
	service := &fakeService{err: ErrInvalidSolution}
	handler, err := NewHandler(service)
	if err != nil {
		t.Fatal(err)
	}
	for _, test := range []struct {
		name        string
		contentType string
		body        string
		wantStatus  int
	}{
		{"content type", "text/plain", `{}`, http.StatusUnsupportedMediaType},
		{"malformed", "application/json", `{`, http.StatusBadRequest},
		{"unknown field", "application/json", `{"solution":"ADEPT","extra":true}`, http.StatusBadRequest},
		{"two values", "application/json", `{"solution":"ADEPT"}{}`, http.StatusBadRequest},
		{"invalid solution", "application/json", `{"solution":"XXXXX"}`, http.StatusBadRequest},
	} {
		t.Run(test.name, func(t *testing.T) {
			request := httptest.NewRequest(http.MethodPost, "/v1/games", strings.NewReader(test.body))
			request.Header.Set("Content-Type", test.contentType)
			response := httptest.NewRecorder()
			handler.ServeHTTP(response, request)
			if response.Code != test.wantStatus {
				t.Fatalf("status = %d, want %d; body = %s", response.Code, test.wantStatus, response.Body.String())
			}
		})
	}
}

func TestHandlerMapsTimeout(t *testing.T) {
	service := &fakeService{err: context.DeadlineExceeded}
	handler, err := NewHandler(service)
	if err != nil {
		t.Fatal(err)
	}
	request := httptest.NewRequest(http.MethodPost, "/v1/games", strings.NewReader(`{"solution":"ADEPT"}`))
	request.Header.Set("Content-Type", "application/json")
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusGatewayTimeout {
		t.Fatalf("status = %d, want %d", response.Code, http.StatusGatewayTimeout)
	}
}

func TestNewHandlerRequiresService(t *testing.T) {
	if _, err := NewHandler(nil); err == nil || err.Error() != "inference service is required" {
		t.Fatalf("error = %v", err)
	}
}

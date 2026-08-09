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

type fakePlayer struct {
	identity  ModelIdentity
	solutions []string
	game      gameeval.GameResult
	err       error
	requested string
}

func (player *fakePlayer) ModelIdentity() ModelIdentity { return player.identity }
func (player *fakePlayer) ValidationSolutions() []string {
	return append([]string(nil), player.solutions...)
}
func (player *fakePlayer) Play(_ context.Context, solution string) (gameeval.GameResult, error) {
	player.requested = solution
	return player.game, player.err
}

func TestHandlerServesSolutionsAndGame(t *testing.T) {
	player := &fakePlayer{
		identity:  ModelIdentity{RunID: "full-1", Checkpoint: "best", Update: 2000, TrainingCommit: "abc", ValidationSplitHash: "def"},
		solutions: []string{"ADEPT", "VODKA"},
		game: gameeval.GameResult{
			Solution: "VODKA", Solved: true, Guesses: 2,
			Turns: []gameeval.TurnResult{{Turn: 1, RawTopActionID: 7, RawTopGuess: "ARISE", Guess: "ARISE", Feedback: "-----", ShortlistSizeBefore: 2309, ShortlistSizeAfter: 167}},
		},
	}
	handler, err := NewHandler(player)
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
	if player.requested != "vodka" {
		t.Fatalf("requested solution = %q", player.requested)
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

func TestHandlerRejectsInvalidRequests(t *testing.T) {
	player := &fakePlayer{err: ErrInvalidSolution}
	handler, err := NewHandler(player)
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
	player := &fakePlayer{err: context.DeadlineExceeded}
	handler, err := NewHandler(player)
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

func TestNewHandlerRequiresPlayer(t *testing.T) {
	if _, err := NewHandler(nil); err == nil || err.Error() != "inference player is required" {
		t.Fatalf("error = %v", err)
	}
}

package cudaweb

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/sam-bee/wordle-ml_machine-learning/modelstate"
	"github.com/sam-bee/wordle-ml_machine-learning/vocabulary"
)

func TestHandlerServesOneModelAndCompleteValidationGames(t *testing.T) {
	service, backend := newTestService(t, nil)
	handler, err := NewHandler(service)
	if err != nil {
		t.Fatalf("NewHandler: %v", err)
	}

	health := getJSON(t, handler, "/healthz")
	if got := health["status"]; got != "ready" {
		t.Fatalf("health status = %#v, want ready", got)
	}
	runtime := health["runtime"].(map[string]any)
	if got := runtime["backend"]; got != "cuda-cgo" {
		t.Fatalf("health runtime backend = %#v, want cuda-cgo", got)
	}
	device := runtime["device"].(map[string]any)
	if got := device["name"]; got != "NVIDIA test GPU" {
		t.Fatalf("health device = %#v", got)
	}

	models := getJSON(t, handler, "/api/models")
	entries := models["models"].([]any)
	if len(entries) != 1 {
		t.Fatalf("models length = %d, want 1", len(entries))
	}
	if got := entries[0].(map[string]any)["run_id"]; got != "seed-replication-20260809-132505Z" {
		t.Fatalf("run ID = %#v", got)
	}

	solutions := getJSON(t, handler, "/api/solutions")
	values := solutions["solutions"].([]any)
	if len(values) != vocabulary.NumValidationSolutions {
		t.Fatalf("solutions length = %d, want %d", len(values), vocabulary.NumValidationSolutions)
	}
	solution := values[0].(string)

	request := httptest.NewRequest(http.MethodPut, "/api/models", strings.NewReader(`{"run_id":"seed-replication-20260809-132505Z"}`))
	request.Header.Set("Content-Type", "application/json")
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("idempotent model selection status = %d body=%s", response.Code, response.Body.String())
	}

	request = httptest.NewRequest(http.MethodPut, "/api/models", strings.NewReader(`{"run_id":"not-a-model"}`))
	request.Header.Set("Content-Type", "application/json")
	response = httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusNotFound {
		t.Fatalf("unknown model status = %d body=%s", response.Code, response.Body.String())
	}

	request = httptest.NewRequest(http.MethodPost, "/api/games", strings.NewReader(`{"solution":"`+solution+`"}`))
	request.Header.Set("Content-Type", "application/json")
	response = httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("game status = %d body=%s", response.Code, response.Body.String())
	}
	var game GameResponse
	if err := json.Unmarshal(response.Body.Bytes(), &game); err != nil {
		t.Fatalf("decode game: %v", err)
	}
	if game.Model.RunID != "seed-replication-20260809-132505Z" || game.Model.Update != 2600 || game.Solution != solution {
		t.Fatalf("game identity = %+v solution=%q", game.Model, game.Solution)
	}
	if calls := backend.Calls(); calls == 0 {
		t.Fatal("game made no CUDA scorer calls")
	}

	request = httptest.NewRequest(http.MethodPost, "/api/games", strings.NewReader(`{"solution":"ZZZZZ"}`))
	request.Header.Set("Content-Type", "application/json")
	response = httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusBadRequest {
		t.Fatalf("non-validation solution status = %d body=%s", response.Code, response.Body.String())
	}

	request = httptest.NewRequest(http.MethodGet, "/", nil)
	response = httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusOK || !strings.Contains(response.Body.String(), "hand-written CUDA via cgo") {
		t.Fatalf("UI status=%d body=%q", response.Code, response.Body.String())
	}
}

func TestPlaySerializesWholeGamesAndHonorsQueuedCancellation(t *testing.T) {
	started := make(chan struct{})
	release := make(chan struct{})
	var first sync.Once
	service, backend := newTestService(t, func(ctx context.Context, _ modelstate.Inputs) error {
		first.Do(func() {
			close(started)
			select {
			case <-release:
			case <-ctx.Done():
			}
		})
		return nil
	})
	solution := service.ValidationSolutions()[0]

	firstDone := make(chan error, 1)
	go func() {
		_, err := service.PlayGame(context.Background(), solution)
		firstDone <- err
	}()
	select {
	case <-started:
	case <-time.After(5 * time.Second):
		t.Fatal("first game did not start")
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := service.PlayGame(ctx, solution); !errors.Is(err, context.Canceled) {
		t.Fatalf("queued game error = %v, want context canceled", err)
	}
	if calls := backend.Calls(); calls != 1 {
		t.Fatalf("queued game reached scorer: calls=%d, want 1", calls)
	}
	close(release)
	select {
	case err := <-firstDone:
		if err != nil {
			t.Fatalf("first game: %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("first game did not finish")
	}
}

func TestNewRejectsInvalidSetup(t *testing.T) {
	vocab := testVocabulary(t)
	model := testModel()
	if _, err := New(Options{Scorer: &fakeBackend{}, Model: model}); err == nil {
		t.Fatal("New without vocabulary succeeded")
	}
	if _, err := New(Options{Vocabulary: vocab, Model: model}); err == nil {
		t.Fatal("New without scorer succeeded")
	}
	model.RunID = ""
	if _, err := New(Options{Vocabulary: vocab, Scorer: &fakeBackend{}, Model: model}); err == nil {
		t.Fatal("New without run ID succeeded")
	}
}

func TestNewRejectsVocabularyContainingFinalTestSplit(t *testing.T) {
	vocab := completeTestVocabulary(t)
	if _, err := New(Options{Vocabulary: vocab, Scorer: &fakeBackend{}, Model: testModel()}); err == nil || !strings.Contains(err.Error(), "final-test") {
		t.Fatalf("New with final-test vocabulary error = %v", err)
	}
}

func completeTestVocabulary(t *testing.T) *vocabulary.Vocabulary {
	t.Helper()
	dir := t.TempDir()
	actions := make([]string, vocabulary.NumActions)
	for index := range actions {
		actions[index] = generatedWord(index)
	}
	solutions := actions[:vocabulary.NumSolutions]
	writeWords := func(filename string, words []string) {
		t.Helper()
		if err := os.WriteFile(filepath.Join(dir, filename), []byte(strings.Join(words, "\n")+"\n"), 0o644); err != nil {
			t.Fatalf("write %s: %v", filename, err)
		}
	}
	writeWords("wordlist-action-space-4739.csv", actions)
	writeWords("wordlist-valid-solutions-all-2309.csv", solutions)
	writeWords("wordlist-valid-solutions-train-2109.csv", solutions[:vocabulary.NumTrainingSolutions])
	writeWords("wordlist-valid-solutions-validation-100.csv", solutions[vocabulary.NumTrainingSolutions:vocabulary.NumTrainingSolutions+vocabulary.NumValidationSolutions])
	writeWords("wordlist-valid-solutions-test-100.csv", solutions[vocabulary.NumTrainingSolutions+vocabulary.NumValidationSolutions:])
	vocab, err := vocabulary.Load(dir)
	if err != nil {
		t.Fatalf("load generated complete vocabulary: %v", err)
	}
	return vocab
}

func generatedWord(index int) string {
	letters := [5]byte{}
	for position := len(letters) - 1; position >= 0; position-- {
		letters[position] = 'A' + byte(index%26)
		index /= 26
	}
	return string(letters[:])
}

func newTestService(t *testing.T, beforeScore func(context.Context, modelstate.Inputs) error) (*Service, *fakeBackend) {
	t.Helper()
	backend := &fakeBackend{beforeScore: beforeScore}
	service, err := New(Options{Vocabulary: testVocabulary(t), Scorer: backend, Model: testModel()})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return service, backend
}

func getJSON(t *testing.T, handler http.Handler, path string) map[string]any {
	t.Helper()
	request := httptest.NewRequest(http.MethodGet, path, nil)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("GET %s status = %d body=%s", path, response.Code, response.Body.String())
	}
	var value map[string]any
	if err := json.Unmarshal(response.Body.Bytes(), &value); err != nil {
		t.Fatalf("decode GET %s: %v", path, err)
	}
	return value
}

func testVocabulary(t *testing.T) *vocabulary.Vocabulary {
	t.Helper()
	vocab, err := vocabulary.LoadWithoutFinalTest(filepath.Join("..", "..", "data"))
	if err != nil {
		t.Fatalf("LoadWithoutFinalTest: %v", err)
	}
	return vocab
}

func testModel() Model {
	return Model{
		Backend:            "cuda-cgo",
		ModelFormat:        "wordle-cuda-f32-v1",
		RunID:              "seed-replication-20260809-132505Z",
		Stage:              "seed-replication",
		Checkpoint:         "best",
		Update:             2600,
		TrainingCommit:     "2718164bb80460757592b90aa86b96eb6d596018",
		WeightsSHA256:      strings.Repeat("a", 64),
		ParameterCount:     1046596,
		DeviceName:         "NVIDIA test GPU",
		ComputeCapability:  "12.0",
		CUDARuntimeVersion: "13.1",
		CUDADriverVersion:  "595.84",
	}
}

type fakeBackend struct {
	mu          sync.Mutex
	calls       int
	beforeScore func(context.Context, modelstate.Inputs) error
}

func (backend *fakeBackend) Score(ctx context.Context, inputs modelstate.Inputs) ([]float32, error) {
	backend.mu.Lock()
	backend.calls++
	beforeScore := backend.beforeScore
	backend.mu.Unlock()
	if beforeScore != nil {
		if err := beforeScore(ctx, inputs); err != nil {
			return nil, err
		}
	}
	logits := make([]float32, vocabulary.NumActions)
	for actionID, candidate := range inputs.RemainingActionMask {
		if candidate != 0 {
			logits[actionID] = 1
		}
	}
	return logits, nil
}

func (backend *fakeBackend) Calls() int {
	backend.mu.Lock()
	defer backend.mu.Unlock()
	return backend.calls
}

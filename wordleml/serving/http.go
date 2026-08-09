package serving

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"log"
	"mime"
	"net/http"
	"time"

	"github.com/sam-bee/wordle-ml_machine-learning/gameeval"
)

const (
	maximumRequestBytes       = 1024
	defaultGameTimeout        = 30 * time.Second
	defaultModelSwitchTimeout = 90 * time.Second
)

// GameResponse is a complete, attributable Wordle trajectory.
type GameResponse struct {
	Model ModelIdentity `json:"model"`
	gameeval.GameResult
}

// Service is the model-selectable inference surface used by the HTTP API.
type Service interface {
	ModelIdentity() ModelIdentity
	ValidationSolutions() []string
	PlayGame(context.Context, string) (GameResponse, error)
	AvailableModels() ([]ModelSummary, error)
	SelectModel(context.Context, string) (ModelIdentity, error)
}

// NewHandler returns the internal inference API. The supplied Service has an
// active model before construction, so healthz is also a readiness signal.
func NewHandler(service Service) (http.Handler, error) {
	if service == nil {
		return nil, errors.New("inference service is required")
	}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", func(response http.ResponseWriter, _ *http.Request) {
		writeJSON(response, http.StatusOK, map[string]any{"status": "ready", "model": service.ModelIdentity()})
	})
	mux.HandleFunc("GET /v1/models", func(response http.ResponseWriter, _ *http.Request) {
		models, err := service.AvailableModels()
		if err != nil {
			log.Printf("list inference models: %v", err)
			writeError(response, http.StatusInternalServerError, "list models")
			return
		}
		writeJSON(response, http.StatusOK, map[string]any{"active": service.ModelIdentity(), "models": models})
	})
	mux.HandleFunc("PUT /v1/models", func(response http.ResponseWriter, request *http.Request) {
		handleModelSelection(response, request, service)
	})
	mux.HandleFunc("GET /v1/solutions", func(response http.ResponseWriter, _ *http.Request) {
		writeJSON(response, http.StatusOK, map[string]any{"model": service.ModelIdentity(), "solutions": service.ValidationSolutions()})
	})
	mux.HandleFunc("POST /v1/games", func(response http.ResponseWriter, request *http.Request) {
		handleGame(response, request, service)
	})
	return noStore(mux), nil
}

func handleGame(response http.ResponseWriter, request *http.Request, service Service) {
	var input struct {
		Solution string `json:"solution"`
	}
	if !decodeRequest(response, request, &input, "request body must be one JSON object containing solution") {
		return
	}
	ctx, cancel := context.WithTimeout(request.Context(), defaultGameTimeout)
	defer cancel()
	game, err := service.PlayGame(ctx, input.Solution)
	if err != nil {
		switch {
		case errors.Is(err, ErrInvalidSolution):
			writeError(response, http.StatusBadRequest, "solution must be one of the advertised validation words")
		case errors.Is(err, context.Canceled), errors.Is(err, context.DeadlineExceeded):
			writeError(response, http.StatusGatewayTimeout, "inference request timed out")
		default:
			log.Printf("inference game failed: %v", err)
			writeError(response, http.StatusInternalServerError, "inference failed")
		}
		return
	}
	writeJSON(response, http.StatusOK, game)
}

func handleModelSelection(response http.ResponseWriter, request *http.Request, service Service) {
	var input struct {
		RunID string `json:"run_id"`
	}
	if !decodeRequest(response, request, &input, "request body must be one JSON object containing run_id") {
		return
	}
	ctx, cancel := context.WithTimeout(request.Context(), defaultModelSwitchTimeout)
	defer cancel()
	model, err := service.SelectModel(ctx, input.RunID)
	if err != nil {
		switch {
		case errors.Is(err, ErrModelNotFound):
			writeError(response, http.StatusNotFound, "model is not available")
		case errors.Is(err, context.Canceled), errors.Is(err, context.DeadlineExceeded):
			writeError(response, http.StatusGatewayTimeout, "model load timed out")
		default:
			log.Printf("select inference model %q: %v", input.RunID, err)
			writeError(response, http.StatusInternalServerError, "load model")
		}
		return
	}
	writeJSON(response, http.StatusOK, map[string]any{"model": model})
}

func decodeRequest(response http.ResponseWriter, request *http.Request, value any, malformedMessage string) bool {
	mediaType, _, err := mime.ParseMediaType(request.Header.Get("Content-Type"))
	if err != nil || mediaType != "application/json" {
		writeError(response, http.StatusUnsupportedMediaType, "Content-Type must be application/json")
		return false
	}
	request.Body = http.MaxBytesReader(response, request.Body, maximumRequestBytes)
	decoder := json.NewDecoder(request.Body)
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(value); err != nil {
		var tooLarge *http.MaxBytesError
		if errors.As(err, &tooLarge) {
			writeError(response, http.StatusRequestEntityTooLarge, "request body is too large")
			return false
		}
		writeError(response, http.StatusBadRequest, malformedMessage)
		return false
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		writeError(response, http.StatusBadRequest, "request body must contain one JSON object")
		return false
	}
	return true
}

func writeError(response http.ResponseWriter, status int, message string) {
	writeJSON(response, status, map[string]string{"error": message})
}

func writeJSON(response http.ResponseWriter, status int, value any) {
	response.Header().Set("Content-Type", "application/json; charset=utf-8")
	response.WriteHeader(status)
	_ = json.NewEncoder(response).Encode(value)
}

func noStore(next http.Handler) http.Handler {
	return http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		response.Header().Set("Cache-Control", "no-store")
		response.Header().Set("X-Content-Type-Options", "nosniff")
		next.ServeHTTP(response, request)
	})
}

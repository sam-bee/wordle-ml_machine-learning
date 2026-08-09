// Package inferenceapi provides the GoMLX-free HTTP contract shared by
// inference applications. It deliberately knows about complete Wordle games
// and model identity, but not about how logits are produced.
package inferenceapi

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"mime"
	"net/http"
	"strings"
	"time"

	"github.com/sam-bee/wordle-ml_machine-learning/gameeval"
)

const (
	maximumRequestBytes       = 1024
	defaultGameTimeout        = 30 * time.Second
	defaultModelSwitchTimeout = 90 * time.Second
	defaultRoutePrefix        = "/api"
)

var (
	// ErrInvalidSolution means a requested solution is outside the fixed
	// population exposed by an inference application.
	ErrInvalidSolution = errors.New("solution is not an allowed validation word")
	// ErrModelNotFound means the requested model is not exposed by an
	// inference application.
	ErrModelNotFound = errors.New("model is not available")
)

// DeviceInfo identifies the CUDA device which owns an inference model. The
// fields are intentionally strings because CUDA presents its runtime and
// driver versions as formatted values rather than application-level numbers.
type DeviceInfo struct {
	Name               string `json:"name,omitempty"`
	ComputeCapability  string `json:"compute_capability,omitempty"`
	CUDARuntimeVersion string `json:"cuda_runtime_version,omitempty"`
	DriverVersion      string `json:"driver_version,omitempty"`
}

// RuntimeInfo records the executable inference boundary and loaded artifact.
// It is optional so the pre-existing GoMLX serving JSON remains compatible;
// CUDA applications can publish it from health and model responses.
type RuntimeInfo struct {
	Backend          string      `json:"backend,omitempty"`
	ModelFormat      string      `json:"model_format,omitempty"`
	RunID            string      `json:"run_id,omitempty"`
	Checkpoint       string      `json:"checkpoint,omitempty"`
	CheckpointUpdate int64       `json:"checkpoint_update,omitempty"`
	TrainingCommit   string      `json:"training_commit,omitempty"`
	WeightsSHA256    string      `json:"weights_sha256,omitempty"`
	ParameterCount   int64       `json:"parameter_count,omitempty"`
	Device           *DeviceInfo `json:"device,omitempty"`
}

// ModelIdentity makes an inference result attributable to one trained model.
// Stage remains a string here so this runtime-facing package cannot import the
// GoMLX-dependent training packages.
type ModelIdentity struct {
	RunID               string `json:"run_id"`
	Stage               string `json:"stage"`
	Checkpoint          string `json:"checkpoint"`
	Update              int64  `json:"update"`
	TrainingCommit      string `json:"training_commit"`
	ValidationSplitHash string `json:"validation_split_hash,omitempty"`
}

// ModelSummary is the lightweight identity shown in the model picker.
type ModelSummary struct {
	RunID          string `json:"run_id"`
	Stage          string `json:"stage"`
	Checkpoint     string `json:"checkpoint"`
	Update         int64  `json:"update"`
	TrainingCommit string `json:"training_commit"`
}

// GameResponse is one complete, attributable Wordle trajectory.
type GameResponse struct {
	Model ModelIdentity `json:"model"`
	gameeval.GameResult
}

// Service is the model-selectable inference surface used by the HTTP API.
// Implementations own validation-solution restrictions and any backend
// serialization; this package only validates HTTP input and maps errors.
type Service interface {
	ModelIdentity() ModelIdentity
	ValidationSolutions() []string
	PlayGame(context.Context, string) (GameResponse, error)
	AvailableModels() ([]ModelSummary, error)
	SelectModel(context.Context, string) (ModelIdentity, error)
}

// RuntimeInfoProvider is implemented by a service which can identify its
// numerical backend and device. It is deliberately optional for backwards
// compatibility with the existing GoMLX serving endpoint.
type RuntimeInfoProvider interface {
	RuntimeInfo() RuntimeInfo
}

// HandlerOptions configures a direct same-origin API. Prefix must be a clean
// absolute path such as "/api" or "/v1"; it never ends in a slash.
type HandlerOptions struct {
	Prefix string
}

// NewHandler serves the direct same-origin API at /api. CUDA applications use
// this default; callers retaining the historical /v1 contract can use
// NewHandlerWithOptions.
func NewHandler(service Service) (http.Handler, error) {
	return NewHandlerWithOptions(service, HandlerOptions{Prefix: defaultRoutePrefix})
}

// NewHandlerWithOptions returns the complete inference API at options.Prefix.
// The supplied service has an active model before construction, so healthz is
// also a readiness signal.
func NewHandlerWithOptions(service Service, options HandlerOptions) (http.Handler, error) {
	if service == nil {
		return nil, errors.New("inference service is required")
	}
	prefix, err := normalizePrefix(options.Prefix)
	if err != nil {
		return nil, err
	}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", func(response http.ResponseWriter, _ *http.Request) {
		writeJSON(response, http.StatusOK, readyResponse(service))
	})
	mux.HandleFunc("GET "+prefix+"/models", func(response http.ResponseWriter, _ *http.Request) {
		models, err := service.AvailableModels()
		if err != nil {
			log.Printf("list inference models: %v", err)
			writeError(response, http.StatusInternalServerError, "list models")
			return
		}
		payload := map[string]any{"active": service.ModelIdentity(), "models": models}
		addRuntimeInfo(payload, service)
		writeJSON(response, http.StatusOK, payload)
	})
	mux.HandleFunc("PUT "+prefix+"/models", func(response http.ResponseWriter, request *http.Request) {
		handleModelSelection(response, request, service)
	})
	mux.HandleFunc("GET "+prefix+"/solutions", func(response http.ResponseWriter, _ *http.Request) {
		payload := map[string]any{"model": service.ModelIdentity(), "solutions": service.ValidationSolutions()}
		addRuntimeInfo(payload, service)
		writeJSON(response, http.StatusOK, payload)
	})
	mux.HandleFunc("POST "+prefix+"/games", func(response http.ResponseWriter, request *http.Request) {
		handleGame(response, request, service)
	})
	return noStore(mux), nil
}

func normalizePrefix(prefix string) (string, error) {
	if prefix == "" {
		prefix = defaultRoutePrefix
	}
	if !strings.HasPrefix(prefix, "/") || prefix == "/" || strings.HasSuffix(prefix, "/") || strings.Contains(prefix, "//") {
		return "", fmt.Errorf("API prefix %q must be a non-root absolute path without a trailing slash", prefix)
	}
	return prefix, nil
}

func readyResponse(service Service) map[string]any {
	payload := map[string]any{"status": "ready", "model": service.ModelIdentity()}
	addRuntimeInfo(payload, service)
	return payload
}

func addRuntimeInfo(payload map[string]any, service Service) {
	provider, ok := service.(RuntimeInfoProvider)
	if !ok {
		return
	}
	payload["runtime"] = provider.RuntimeInfo()
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
	payload := map[string]any{"model": model}
	addRuntimeInfo(payload, service)
	writeJSON(response, http.StatusOK, payload)
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

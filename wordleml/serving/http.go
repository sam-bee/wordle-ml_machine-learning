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
	maximumRequestBytes = 1024
	defaultGameTimeout  = 30 * time.Second
)

// GameResponse is a complete, attributable Wordle trajectory.
type GameResponse struct {
	Model ModelIdentity `json:"model"`
	gameeval.GameResult
}

// NewHandler returns the internal inference API. The supplied Player is ready
// before the handler is constructed, so healthz is also a readiness signal.
func NewHandler(player Player) (http.Handler, error) {
	if player == nil {
		return nil, errors.New("inference player is required")
	}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", func(response http.ResponseWriter, _ *http.Request) {
		writeJSON(response, http.StatusOK, map[string]any{"status": "ready", "model": player.ModelIdentity()})
	})
	mux.HandleFunc("GET /v1/solutions", func(response http.ResponseWriter, _ *http.Request) {
		writeJSON(response, http.StatusOK, map[string]any{"model": player.ModelIdentity(), "solutions": player.ValidationSolutions()})
	})
	mux.HandleFunc("POST /v1/games", func(response http.ResponseWriter, request *http.Request) {
		handleGame(response, request, player)
	})
	return noStore(mux), nil
}

func handleGame(response http.ResponseWriter, request *http.Request, player Player) {
	mediaType, _, err := mime.ParseMediaType(request.Header.Get("Content-Type"))
	if err != nil || mediaType != "application/json" {
		writeError(response, http.StatusUnsupportedMediaType, "Content-Type must be application/json")
		return
	}
	request.Body = http.MaxBytesReader(response, request.Body, maximumRequestBytes)
	decoder := json.NewDecoder(request.Body)
	decoder.DisallowUnknownFields()
	var input struct {
		Solution string `json:"solution"`
	}
	if err := decoder.Decode(&input); err != nil {
		var tooLarge *http.MaxBytesError
		if errors.As(err, &tooLarge) {
			writeError(response, http.StatusRequestEntityTooLarge, "request body is too large")
			return
		}
		writeError(response, http.StatusBadRequest, "request body must be one JSON object containing solution")
		return
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		writeError(response, http.StatusBadRequest, "request body must contain one JSON object")
		return
	}
	ctx, cancel := context.WithTimeout(request.Context(), defaultGameTimeout)
	defer cancel()
	game, err := player.Play(ctx, input.Solution)
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
	writeJSON(response, http.StatusOK, GameResponse{Model: player.ModelIdentity(), GameResult: game})
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

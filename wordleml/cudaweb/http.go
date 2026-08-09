package cudaweb

import (
	"embed"
	"errors"
	"io/fs"
	"net/http"

	"github.com/sam-bee/wordle-ml_machine-learning/inferenceapi"
)

//go:embed static
var staticFiles embed.FS

// NewHandler joins the shared direct same-process API with the embedded
// browser UI. No route proxies to a second inference process.
func NewHandler(service *Service) (http.Handler, error) {
	if service == nil || service.evaluator == nil {
		return nil, errors.New("CUDA web service is required")
	}
	api, err := inferenceapi.NewHandler(service)
	if err != nil {
		return nil, err
	}
	assets, err := fs.Sub(staticFiles, "static")
	if err != nil {
		return nil, err
	}
	mux := http.NewServeMux()
	mux.Handle("/healthz", api)
	mux.Handle("/api/", api)
	mux.Handle("/", http.FileServerFS(assets))
	return securityHeaders(mux), nil
}

func securityHeaders(next http.Handler) http.Handler {
	return http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		response.Header().Set("Content-Security-Policy", "default-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'; img-src 'self'; base-uri 'none'; frame-ancestors 'none'")
		response.Header().Set("Referrer-Policy", "no-referrer")
		response.Header().Set("X-Content-Type-Options", "nosniff")
		next.ServeHTTP(response, request)
	})
}

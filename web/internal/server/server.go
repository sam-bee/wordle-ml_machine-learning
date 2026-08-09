package server

import (
	"bytes"
	"embed"
	"encoding/json"
	"errors"
	"io"
	"io/fs"
	"net/http"
	"net/url"
	"strings"
	"time"
)

//go:embed static
var staticFiles embed.FS

const (
	defaultInferenceURL         = "http://inference:8090"
	maximumProxyRequestBytes    = 4096
	maximumProxyResponseBytes   = 1 << 20
	defaultInferenceHTTPTimeout = 45 * time.Second
)

// Config supplies the internal inference upstream. It is never sent to the
// browser; /api routes remain same-origin through this server.
type Config struct {
	InferenceURL string
	HTTPClient   *http.Client
}

// Handler returns the application using the Compose inference-service URL.
// Tests which only exercise static and health routes can use this convenience.
func Handler() http.Handler {
	handler, err := NewHandler(Config{InferenceURL: defaultInferenceURL})
	if err != nil {
		panic(err)
	}
	return handler
}

// NewHandler returns the static application, health endpoint, and constrained
// inference proxy.
func NewHandler(config Config) (http.Handler, error) {
	assets, err := fs.Sub(staticFiles, "static")
	if err != nil {
		return nil, err
	}
	baseURL, err := parseInferenceURL(config.InferenceURL)
	if err != nil {
		return nil, err
	}
	client := config.HTTPClient
	if client == nil {
		client = &http.Client{Timeout: defaultInferenceHTTPTimeout}
	}

	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", func(response http.ResponseWriter, _ *http.Request) {
		response.Header().Set("Content-Type", "text/plain; charset=utf-8")
		response.WriteHeader(http.StatusOK)
		_, _ = response.Write([]byte("ok\n"))
	})
	mux.HandleFunc("GET /api/solutions", func(response http.ResponseWriter, request *http.Request) {
		proxyInference(response, request, client, baseURL, "/v1/solutions")
	})
	mux.HandleFunc("POST /api/games", func(response http.ResponseWriter, request *http.Request) {
		proxyInference(response, request, client, baseURL, "/v1/games")
	})
	mux.Handle("GET /", http.FileServerFS(assets))

	return securityHeaders(mux), nil
}

func parseInferenceURL(value string) (*url.URL, error) {
	if strings.TrimSpace(value) == "" {
		value = defaultInferenceURL
	}
	parsed, err := url.Parse(value)
	if err != nil || (parsed.Scheme != "http" && parsed.Scheme != "https") || parsed.Host == "" || parsed.User != nil || parsed.RawQuery != "" || parsed.Fragment != "" {
		return nil, errors.New("INFERENCE_URL must be an absolute HTTP(S) URL without credentials, query, or fragment")
	}
	parsed.Path = strings.TrimRight(parsed.Path, "/")
	return parsed, nil
}

func proxyInference(response http.ResponseWriter, request *http.Request, client *http.Client, baseURL *url.URL, path string) {
	request.Body = http.MaxBytesReader(response, request.Body, maximumProxyRequestBytes)
	body, err := io.ReadAll(request.Body)
	if err != nil {
		var tooLarge *http.MaxBytesError
		if errors.As(err, &tooLarge) {
			writeProxyError(response, http.StatusRequestEntityTooLarge, "request body is too large")
			return
		}
		writeProxyError(response, http.StatusBadRequest, "read request body")
		return
	}
	upstreamURL := *baseURL
	upstreamURL.Path += path
	upstreamURL.RawPath = ""
	upstream, err := http.NewRequestWithContext(request.Context(), request.Method, upstreamURL.String(), bytes.NewReader(body))
	if err != nil {
		writeProxyError(response, http.StatusBadGateway, "create inference request")
		return
	}
	if contentType := request.Header.Get("Content-Type"); contentType != "" {
		upstream.Header.Set("Content-Type", contentType)
	}
	upstream.Header.Set("Accept", "application/json")
	result, err := client.Do(upstream)
	if err != nil {
		writeProxyError(response, http.StatusBadGateway, "inference service is unavailable")
		return
	}
	defer result.Body.Close()
	contents, err := io.ReadAll(io.LimitReader(result.Body, maximumProxyResponseBytes+1))
	if err != nil || len(contents) > maximumProxyResponseBytes {
		writeProxyError(response, http.StatusBadGateway, "invalid inference response")
		return
	}
	contentType := result.Header.Get("Content-Type")
	if contentType == "" {
		contentType = "application/json; charset=utf-8"
	}
	response.Header().Set("Content-Type", contentType)
	response.Header().Set("Cache-Control", "no-store")
	response.WriteHeader(result.StatusCode)
	_, _ = response.Write(contents)
}

func writeProxyError(response http.ResponseWriter, status int, message string) {
	response.Header().Set("Content-Type", "application/json; charset=utf-8")
	response.Header().Set("Cache-Control", "no-store")
	response.WriteHeader(status)
	_ = json.NewEncoder(response).Encode(map[string]string{"error": message})
}

func securityHeaders(next http.Handler) http.Handler {
	return http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		response.Header().Set("Content-Security-Policy", "default-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'; img-src 'self'; base-uri 'none'; frame-ancestors 'none'")
		response.Header().Set("Referrer-Policy", "no-referrer")
		response.Header().Set("X-Content-Type-Options", "nosniff")
		next.ServeHTTP(response, request)
	})
}

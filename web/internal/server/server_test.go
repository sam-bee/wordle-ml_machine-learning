package server

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestSplashPage(t *testing.T) {
	request := httptest.NewRequest(http.MethodGet, "/", nil)
	response := httptest.NewRecorder()

	Handler().ServeHTTP(response, request)

	if response.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", response.Code, http.StatusOK)
	}
	if !strings.Contains(response.Body.String(), "Wordle ML") {
		t.Fatalf("splash page does not contain project name: %q", response.Body.String())
	}
	if !strings.Contains(response.Body.String(), `id="model"`) {
		t.Fatalf("splash page does not contain the model selector: %q", response.Body.String())
	}
}

func TestHealth(t *testing.T) {
	request := httptest.NewRequest(http.MethodGet, "/healthz", nil)
	response := httptest.NewRecorder()

	Handler().ServeHTTP(response, request)

	result := response.Result()
	defer result.Body.Close()
	body, err := io.ReadAll(result.Body)
	if err != nil {
		t.Fatal(err)
	}
	if result.StatusCode != http.StatusOK || string(body) != "ok\n" {
		t.Fatalf("health response = (%d, %q), want (200, %q)", result.StatusCode, body, "ok\n")
	}
}

func TestInferenceProxy(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		response.Header().Set("Content-Type", "application/json")
		switch request.URL.Path {
		case "/v1/models":
			if request.Method == http.MethodGet {
				_, _ = response.Write([]byte(`{"active":{"run_id":"full-1"},"models":[{"run_id":"full-1"},{"run_id":"production-1"}]}`))
				return
			}
			if request.Method != http.MethodPut || request.Header.Get("Content-Type") != "application/json" {
				t.Errorf("model request = (%s, %q)", request.Method, request.Header.Get("Content-Type"))
			}
			body, err := io.ReadAll(request.Body)
			if err != nil {
				t.Error(err)
			}
			if string(body) != `{"run_id":"production-1"}` {
				t.Errorf("model body = %s", body)
			}
			_, _ = response.Write([]byte(`{"model":{"run_id":"production-1"}}`))
		case "/v1/solutions":
			if request.Method != http.MethodGet {
				t.Errorf("solutions method = %s", request.Method)
			}
			_, _ = response.Write([]byte(`{"solutions":["ADEPT","VODKA"]}`))
		case "/v1/games":
			if request.Method != http.MethodPost || request.Header.Get("Content-Type") != "application/json" {
				t.Errorf("game request = (%s, %q)", request.Method, request.Header.Get("Content-Type"))
			}
			body, err := io.ReadAll(request.Body)
			if err != nil {
				t.Error(err)
			}
			if string(body) != `{"solution":"VODKA"}` {
				t.Errorf("game body = %s", body)
			}
			_, _ = response.Write([]byte(`{"solution":"VODKA","solved":true,"turns":[]}`))
		default:
			http.NotFound(response, request)
		}
	}))
	defer upstream.Close()
	handler, err := NewHandler(Config{InferenceURL: upstream.URL, HTTPClient: upstream.Client()})
	if err != nil {
		t.Fatal(err)
	}

	for _, request := range []*http.Request{
		httptest.NewRequest(http.MethodGet, "/api/models", nil),
		httptest.NewRequest(http.MethodPut, "/api/models", strings.NewReader(`{"run_id":"production-1"}`)),
		httptest.NewRequest(http.MethodGet, "/api/solutions", nil),
		httptest.NewRequest(http.MethodPost, "/api/games", strings.NewReader(`{"solution":"VODKA"}`)),
	} {
		if request.Method == http.MethodPost || request.Method == http.MethodPut {
			request.Header.Set("Content-Type", "application/json")
		}
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, request)
		if response.Code != http.StatusOK || !json.Valid(response.Body.Bytes()) {
			t.Fatalf("proxy response = (%d, %s)", response.Code, response.Body.String())
		}
	}
}

func TestInferenceProxyUnavailable(t *testing.T) {
	handler, err := NewHandler(Config{InferenceURL: "http://127.0.0.1:1", HTTPClient: &http.Client{}})
	if err != nil {
		t.Fatal(err)
	}
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, httptest.NewRequest(http.MethodGet, "/api/solutions", nil))
	if response.Code != http.StatusBadGateway {
		t.Fatalf("status = %d, want %d", response.Code, http.StatusBadGateway)
	}
}

func TestNewHandlerRejectsInvalidInferenceURL(t *testing.T) {
	if _, err := NewHandler(Config{InferenceURL: "file:///tmp/socket"}); err == nil {
		t.Fatal("invalid inference URL unexpectedly accepted")
	}
}

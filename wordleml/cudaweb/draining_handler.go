package cudaweb

import (
	"net/http"
	"sync"
)

// DrainingHandler wraps an HTTP handler with an explicit admission gate. Once
// BeginDrain returns, no new request can reach the wrapped CUDA/game handler;
// Wait then proves that all previously admitted requests have returned.
type DrainingHandler struct {
	next http.Handler

	mu       sync.Mutex
	draining bool
	active   int
	drained  *sync.Cond
}

// NewDrainingHandler wraps next for lifecycle coordination.
func NewDrainingHandler(next http.Handler) *DrainingHandler {
	if next == nil {
		panic("CUDA web draining handler requires a next handler")
	}
	drainer := &DrainingHandler{next: next}
	drainer.drained = sync.NewCond(&drainer.mu)
	return drainer
}

// ServeHTTP admits a request unless shutdown has started. New requests after
// BeginDrain receive a plain retryable response and cannot reach CUDA.
func (drainer *DrainingHandler) ServeHTTP(response http.ResponseWriter, request *http.Request) {
	drainer.mu.Lock()
	if drainer.draining {
		drainer.mu.Unlock()
		http.Error(response, "server is shutting down", http.StatusServiceUnavailable)
		return
	}
	drainer.active++
	drainer.mu.Unlock()
	defer func() {
		drainer.mu.Lock()
		drainer.active--
		if drainer.draining && drainer.active == 0 {
			drainer.drained.Broadcast()
		}
		drainer.mu.Unlock()
	}()
	drainer.next.ServeHTTP(response, request)
}

// BeginDrain rejects later requests. It is idempotent and may race safely with
// request admission.
func (drainer *DrainingHandler) BeginDrain() {
	if drainer == nil {
		return
	}
	drainer.mu.Lock()
	drainer.draining = true
	if drainer.active == 0 {
		drainer.drained.Broadcast()
	}
	drainer.mu.Unlock()
}

// Wait blocks until every request admitted before BeginDrain has returned.
func (drainer *DrainingHandler) Wait() {
	if drainer == nil {
		return
	}
	drainer.mu.Lock()
	for !drainer.draining || drainer.active != 0 {
		drainer.drained.Wait()
	}
	drainer.mu.Unlock()
}

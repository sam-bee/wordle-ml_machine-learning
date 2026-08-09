package cudaweb

import (
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestDrainingHandlerRejectsLateRequestsAndWaitsForAdmittedHandler(t *testing.T) {
	started := make(chan struct{})
	release := make(chan struct{})
	next := http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		close(started)
		<-release
	})
	drainer := NewDrainingHandler(next)
	firstDone := make(chan struct{})
	go func() {
		drainer.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest(http.MethodGet, "/", nil))
		close(firstDone)
	}()
	waitFor(t, started, "first handler start")

	drainer.BeginDrain()
	late := httptest.NewRecorder()
	drainer.ServeHTTP(late, httptest.NewRequest(http.MethodGet, "/", nil))
	if late.Code != http.StatusServiceUnavailable {
		t.Fatalf("late request status = %d, want %d", late.Code, http.StatusServiceUnavailable)
	}
	waitDone := make(chan struct{})
	go func() {
		drainer.Wait()
		close(waitDone)
	}()
	select {
	case <-waitDone:
		t.Fatal("drainer returned before admitted handler completed")
	case <-time.After(30 * time.Millisecond):
	}
	close(release)
	waitFor(t, firstDone, "first handler completion")
	waitFor(t, waitDone, "drainer completion")
}

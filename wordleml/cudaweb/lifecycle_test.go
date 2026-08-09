package cudaweb

import (
	"context"
	"errors"
	"net/http"
	"reflect"
	"testing"
	"time"
)

func TestCloseAfterHTTPDrainClosesServiceThenBackendAndJoinsFailures(t *testing.T) {
	serviceErr := errors.New("service close failed")
	backendErr := errors.New("backend close failed")
	var events []string
	err := CloseAfterHTTPDrain(
		recordingCloser{name: "service", events: &events, err: serviceErr},
		recordingCloser{name: "backend", events: &events, err: backendErr},
	)
	if !errors.Is(err, serviceErr) {
		t.Fatalf("close error = %v, missing service close failure", err)
	}
	if !errors.Is(err, backendErr) {
		t.Fatalf("close error = %v, missing backend close failure", err)
	}
	if want := []string{"service", "backend"}; !reflect.DeepEqual(events, want) {
		t.Fatalf("close order = %v, want %v", events, want)
	}
}

func TestServeUntilContextWaitsForShutdownDrainBeforeReturning(t *testing.T) {
	server := newLifecycleServer(http.ErrServerClosed)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan error, 1)
	go func() { done <- ServeUntilContext(ctx, server, time.Second, nil) }()
	waitFor(t, server.listenStarted, "listener start")

	cancel()
	waitFor(t, server.shutdownStarted, "shutdown start")
	// Shutdown closes the listener first, while the active handler drain is
	// still pending. Returning from the helper here would allow a caller to
	// destroy its CUDA backend before that handler has completed.
	close(server.releaseListen)
	ensureNotReturned(t, done)
	close(server.releaseShutdown)
	if err := receiveResult(t, done); err != nil {
		t.Fatalf("ServeUntilContext: %v", err)
	}
}

func TestServeUntilContextCancelsAndWaitsAfterListenerFailure(t *testing.T) {
	listenErr := errors.New("listener failed")
	server := newLifecycleServer(listenErr)
	close(server.releaseListen)
	done := make(chan error, 1)
	go func() { done <- ServeUntilContext(context.Background(), server, time.Second, nil) }()
	waitFor(t, server.listenStarted, "listener start")
	waitFor(t, server.shutdownStarted, "shutdown after listener failure")
	ensureNotReturned(t, done)
	close(server.releaseShutdown)
	if err := receiveResult(t, done); !errors.Is(err, listenErr) {
		t.Fatalf("ServeUntilContext error = %v, want listener failure", err)
	}
}

func TestServeUntilContextWaitsForRequestDrainAfterShutdownFailure(t *testing.T) {
	shutdownErr := errors.New("shutdown deadline")
	server := newLifecycleServer(http.ErrServerClosed)
	server.shutdownErr = shutdownErr
	drainer := newBlockingDrainer()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan error, 1)
	go func() { done <- ServeUntilContext(ctx, server, time.Second, drainer) }()
	waitFor(t, server.listenStarted, "listener start")

	cancel()
	waitFor(t, drainer.begun, "request-drain start")
	waitFor(t, server.shutdownStarted, "shutdown start")
	close(server.releaseListen)
	close(server.releaseShutdown)
	waitFor(t, drainer.waitStarted, "request-drain wait")
	ensureNotReturned(t, done)
	close(drainer.release)
	if err := receiveResult(t, done); !errors.Is(err, shutdownErr) {
		t.Fatalf("ServeUntilContext error = %v, want shutdown failure", err)
	}
}

func TestServeUntilContextRejectsInvalidArguments(t *testing.T) {
	server := newLifecycleServer(http.ErrServerClosed)
	if err := ServeUntilContext(nil, server, time.Second, nil); err == nil {
		t.Fatal("nil context was accepted")
	}
	if err := ServeUntilContext(context.Background(), nil, time.Second, nil); err == nil {
		t.Fatal("nil server was accepted")
	}
	if err := ServeUntilContext(context.Background(), server, 0, nil); err == nil {
		t.Fatal("zero timeout was accepted")
	}
}

type lifecycleServer struct {
	listenStarted   chan struct{}
	releaseListen   chan struct{}
	shutdownStarted chan struct{}
	releaseShutdown chan struct{}
	listenErr       error
	shutdownErr     error
}

func newLifecycleServer(listenErr error) *lifecycleServer {
	return &lifecycleServer{
		listenStarted:   make(chan struct{}),
		releaseListen:   make(chan struct{}),
		shutdownStarted: make(chan struct{}),
		releaseShutdown: make(chan struct{}),
		listenErr:       listenErr,
	}
}

func (server *lifecycleServer) ListenAndServe() error {
	close(server.listenStarted)
	<-server.releaseListen
	return server.listenErr
}

func (server *lifecycleServer) Shutdown(ctx context.Context) error {
	close(server.shutdownStarted)
	select {
	case <-server.releaseShutdown:
		return server.shutdownErr
	case <-ctx.Done():
		return ctx.Err()
	}
}

type blockingDrainer struct {
	begun       chan struct{}
	waitStarted chan struct{}
	release     chan struct{}
}

type recordingCloser struct {
	name   string
	events *[]string
	err    error
}

func (closer recordingCloser) Close() error {
	*closer.events = append(*closer.events, closer.name)
	return closer.err
}

func newBlockingDrainer() *blockingDrainer {
	return &blockingDrainer{
		begun:       make(chan struct{}),
		waitStarted: make(chan struct{}),
		release:     make(chan struct{}),
	}
}

func (drainer *blockingDrainer) BeginDrain() { close(drainer.begun) }

func (drainer *blockingDrainer) Wait() {
	close(drainer.waitStarted)
	<-drainer.release
}

func waitFor(t *testing.T, value <-chan struct{}, description string) {
	t.Helper()
	select {
	case <-value:
	case <-time.After(5 * time.Second):
		t.Fatalf("timed out waiting for %s", description)
	}
}

func ensureNotReturned(t *testing.T, result <-chan error) {
	t.Helper()
	select {
	case err := <-result:
		t.Fatalf("ServeUntilContext returned before shutdown drained: %v", err)
	case <-time.After(30 * time.Millisecond):
	}
}

func receiveResult(t *testing.T, result <-chan error) error {
	t.Helper()
	select {
	case err := <-result:
		return err
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for ServeUntilContext")
		return nil
	}
}

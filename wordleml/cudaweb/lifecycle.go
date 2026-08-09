package cudaweb

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"time"
)

// GracefulServer is the narrow HTTP lifecycle contract used by
// ServeUntilContext. Keeping it independent of the CUDA build tag makes the
// shutdown ordering testable without a native library.
type GracefulServer interface {
	ListenAndServe() error
	Shutdown(context.Context) error
}

// RequestDrainer prevents admission of new requests and waits until every
// already-admitted handler returns. It closes the small gap left if a timed
// Server.Shutdown returns before a slow handler has finished.
type RequestDrainer interface {
	BeginDrain()
	Wait()
}

// RuntimeCloser is a component that must stay alive until HTTP handlers have
// drained. It is deliberately small so the command lifecycle can be tested
// without a cgo CUDA backend.
type RuntimeCloser interface {
	Close() error
}

// CloseAfterHTTPDrain releases the service before its CUDA backend. Call it
// only after ServeUntilContext returns: it attempts both closes and preserves
// each failure for errors.Is callers.
func CloseAfterHTTPDrain(service, backend RuntimeCloser) error {
	var serviceErr, backendErr error
	if service != nil {
		if err := service.Close(); err != nil {
			serviceErr = fmt.Errorf("close CUDA web service: %w", err)
		}
	}
	if backend != nil {
		if err := backend.Close(); err != nil {
			backendErr = fmt.Errorf("close CUDA backend: %w", err)
		}
	}
	return errors.Join(serviceErr, backendErr)
}

// ServeUntilContext runs server until either its listener exits or ctx is
// cancelled. It always waits for Shutdown before returning. Callers can safely
// release a CUDA backend only after this function has returned: active HTTP
// handlers have then drained and no handler can still own request inputs. If
// drainer is supplied, it is closed before Server.Shutdown and waited even
// when that server shutdown reaches its deadline.
func ServeUntilContext(ctx context.Context, server GracefulServer, shutdownTimeout time.Duration, drainer RequestDrainer) error {
	if ctx == nil {
		return errors.New("server context is required")
	}
	if server == nil {
		return errors.New("HTTP server is required")
	}
	if shutdownTimeout <= 0 {
		return fmt.Errorf("shutdown timeout must be positive, got %s", shutdownTimeout)
	}

	runContext, cancel := context.WithCancel(ctx)
	defer cancel()
	shutdownDone := make(chan error, 1)
	go func() {
		<-runContext.Done()
		if drainer != nil {
			drainer.BeginDrain()
		}
		shutdownContext, shutdownCancel := context.WithTimeout(context.Background(), shutdownTimeout)
		defer shutdownCancel()
		shutdownErr := server.Shutdown(shutdownContext)
		if drainer != nil {
			drainer.Wait()
		}
		shutdownDone <- shutdownErr
	}()

	listenErr := server.ListenAndServe()
	// A listener failure also triggers and waits for Shutdown. This avoids a
	// race where a defer releases native state while a handler is still running.
	cancel()
	shutdownErr := <-shutdownDone
	var result error
	if listenErr != nil && !errors.Is(listenErr, http.ErrServerClosed) {
		result = fmt.Errorf("serve HTTP server: %w", listenErr)
	}
	if shutdownErr != nil && !errors.Is(shutdownErr, http.ErrServerClosed) {
		result = errors.Join(result, fmt.Errorf("shutdown HTTP server: %w", shutdownErr))
	}
	return result
}

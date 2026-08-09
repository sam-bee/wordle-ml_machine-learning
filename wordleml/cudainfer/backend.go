// Package cudainfer owns the Go side of one hand-written CUDA policy model.
//
// CUDA evaluates the neural network only. The callers of Backend retain the
// Wordle rules, legal-action filtering, and deterministic action selection.
package cudainfer

import (
	"context"
	"errors"
	"fmt"
	"math"

	"github.com/sam-bee/wordle-ml_machine-learning/cudamodel"
	"github.com/sam-bee/wordle-ml_machine-learning/modelstate"
	"github.com/sam-bee/wordle-ml_machine-learning/vocabulary"
)

// ErrClosed is returned when inference is requested after Backend.Close.
var ErrClosed = errors.New("CUDA inference backend is closed")

// Info describes the CUDA device selected when the native model was created.
// The native backend fills these values once, before it accepts inference.
type Info struct {
	DeviceName         string `json:"device_name"`
	ComputeCapability  string `json:"compute_capability"`
	CUDARuntimeVersion string `json:"cuda_runtime_version"`
	CUDADriverVersion  string `json:"cuda_driver_version"`
}

// Backend performs one synchronous full policy forward pass at a time.
// Returned scores are raw, unmasked logits in canonical action-ID order.
type Backend interface {
	Score(context.Context, modelstate.Inputs) ([]float32, error)
	Info() Info
	Close() error
}

// nativeModel is deliberately small so worker lifetime and serialization are
// testable without a CUDA device. Its implementations must complete Score
// synchronously and must not retain references to inputs after it returns.
type nativeModel interface {
	Score(modelstate.Inputs) ([]float32, error)
	Info() Info
	Close() error
}

type nativeFactory func() (nativeModel, error)

type backend struct {
	worker *worker
}

func newBackend(factory nativeFactory) (*backend, error) {
	if factory == nil {
		return nil, errors.New("CUDA native-model factory is nil")
	}
	worker, err := newWorker(factory)
	if err != nil {
		return nil, err
	}
	return &backend{worker: worker}, nil
}

// Score submits a complete forward pass to the locked CUDA worker. Context
// cancellation is observed before and while enqueueing. After enqueueing, the
// method waits for the synchronous native result: the worker may still be
// reading the caller-owned input slices, so returning early would be unsafe.
func (backend *backend) Score(ctx context.Context, inputs modelstate.Inputs) ([]float32, error) {
	if backend == nil || backend.worker == nil {
		return nil, ErrClosed
	}
	return backend.worker.Score(ctx, inputs)
}

// Info returns the device details captured during native model creation.
func (backend *backend) Info() Info {
	if backend == nil || backend.worker == nil {
		return Info{}
	}
	return backend.worker.Info()
}

// Close waits for accepted inference requests, destroys the native model on
// its owning OS thread, and is safe to call more than once.
func (backend *backend) Close() error {
	if backend == nil || backend.worker == nil {
		return nil
	}
	return backend.worker.Close()
}

func validateInputs(inputs modelstate.Inputs) error {
	if _, err := cudamodel.ValidateInputs(inputs); err != nil {
		return fmt.Errorf("validate CUDA inputs: %w", err)
	}
	return nil
}

func validateLogits(logits []float32) error {
	if len(logits) != vocabulary.NumActions {
		return fmt.Errorf("native CUDA model returned %d logits, want %d", len(logits), vocabulary.NumActions)
	}
	for index, value := range logits {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return fmt.Errorf("native CUDA model returned non-finite logit at action %d", index)
		}
	}
	return nil
}

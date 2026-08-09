package cudainfer

import (
	"context"
	"errors"
	"math"
	"sync"
	"testing"
	"time"

	"github.com/sam-bee/wordle-ml_machine-learning/modelstate"
	"github.com/sam-bee/wordle-ml_machine-learning/vocabulary"
)

func TestScoreRejectsInvalidInputsBeforeNativeInference(t *testing.T) {
	model := &fakeNativeModel{}
	backend := newTestBackend(t, model)
	t.Cleanup(func() { _ = backend.Close() })

	for name, mutate := range map[string]func(*modelstate.Inputs){
		"candidate mask length": func(inputs *modelstate.Inputs) {
			inputs.CandidateMask = inputs.CandidateMask[:vocabulary.NumSolutions-1]
		},
		"candidate stats length": func(inputs *modelstate.Inputs) {
			inputs.CandidateStats = inputs.CandidateStats[:modelstate.CandidateStatsSize-1]
		},
		"turn": func(inputs *modelstate.Inputs) { inputs.Turn = 6 },
		"remaining action mask length": func(inputs *modelstate.Inputs) {
			inputs.RemainingActionMask = inputs.RemainingActionMask[:vocabulary.NumActions-1]
		},
		"empty candidates":          func(inputs *modelstate.Inputs) { clear(inputs.CandidateMask) },
		"non-finite candidate mask": func(inputs *modelstate.Inputs) { inputs.CandidateMask[0] = float32(math.NaN()) },
		"non-finite candidate stats": func(inputs *modelstate.Inputs) {
			inputs.CandidateStats[0] = float32(math.Inf(1))
		},
		"non-finite remaining mask": func(inputs *modelstate.Inputs) {
			inputs.RemainingActionMask[0] = float32(math.NaN())
		},
	} {
		t.Run(name, func(t *testing.T) {
			inputs := validInputs()
			mutate(&inputs)
			if _, err := backend.Score(context.Background(), inputs); err == nil {
				t.Fatal("Score accepted invalid inputs")
			}
		})
	}
	if calls := model.callCount(); calls != 0 {
		t.Fatalf("native Score calls = %d, want 0", calls)
	}
}

func TestScoreRejectsCancelledContextBeforeEnqueue(t *testing.T) {
	model := &fakeNativeModel{}
	backend := newTestBackend(t, model)
	t.Cleanup(func() { _ = backend.Close() })

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := backend.Score(ctx, validInputs()); !errors.Is(err, context.Canceled) {
		t.Fatalf("Score error = %v, want context.Canceled", err)
	}
	if calls := model.callCount(); calls != 0 {
		t.Fatalf("native Score calls = %d, want 0", calls)
	}
}

func TestScoreRejectsCancellationWhileWaitingToEnqueue(t *testing.T) {
	model := &fakeNativeModel{}
	backend := newTestBackend(t, model)
	t.Cleanup(func() { _ = backend.Close() })

	// Keep Score behind the enqueue gate until its initial ctx.Err call has
	// completed. Cancellation then occurs before the worker can accept the
	// request, which must not reach native inference.
	backend.worker.stateMu.Lock()
	parent, cancel := context.WithCancel(context.Background())
	ctx := &firstErrGateContext{
		Context:          parent,
		firstErrCalled:   make(chan struct{}),
		firstErrReturned: make(chan struct{}),
		continueFirstErr: make(chan struct{}),
	}
	returned := make(chan error, 1)
	go func() {
		_, err := backend.Score(ctx, validInputs())
		returned <- err
	}()
	<-ctx.firstErrCalled
	close(ctx.continueFirstErr)
	<-ctx.firstErrReturned
	cancel()
	backend.worker.stateMu.Unlock()

	if err := <-returned; !errors.Is(err, context.Canceled) {
		t.Fatalf("Score error = %v, want context.Canceled", err)
	}
	if calls := model.callCount(); calls != 0 {
		t.Fatalf("native Score calls = %d, want 0", calls)
	}
}

func TestWorkerSerializesConcurrentCallers(t *testing.T) {
	model := &fakeNativeModel{}
	backend := newTestBackend(t, model)
	t.Cleanup(func() { _ = backend.Close() })

	const callers = 24
	var group sync.WaitGroup
	errCh := make(chan error, callers)
	for caller := range callers {
		group.Add(1)
		go func() {
			defer group.Done()
			inputs := validInputs()
			inputs.Turn = int32(caller % 6)
			logits, err := backend.Score(context.Background(), inputs)
			if err != nil {
				errCh <- err
				return
			}
			if got, want := logits[0], float32(inputs.Turn); got != want {
				errCh <- errors.New("caller received another request's logits")
			}
		}()
	}
	group.Wait()
	close(errCh)
	for err := range errCh {
		t.Error(err)
	}
	if calls := model.callCount(); calls != callers {
		t.Fatalf("native Score calls = %d, want %d", calls, callers)
	}
	if maximum := model.maximumConcurrent(); maximum != 1 {
		t.Fatalf("native maximum concurrency = %d, want 1", maximum)
	}
}

func TestScoreWaitsForAcceptedNativeCallAfterContextCancellation(t *testing.T) {
	started := make(chan struct{})
	release := make(chan struct{})
	model := &fakeNativeModel{started: started, release: release}
	backend := newTestBackend(t, model)
	t.Cleanup(func() { _ = backend.Close() })

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	returned := make(chan error, 1)
	go func() {
		_, err := backend.Score(ctx, validInputs())
		returned <- err
	}()
	<-started
	cancel()
	select {
	case err := <-returned:
		t.Fatalf("Score returned before native inference finished: %v", err)
	case <-time.After(25 * time.Millisecond):
	}
	close(release)
	if err := <-returned; err != nil {
		t.Fatalf("Score after native completion: %v", err)
	}
}

func TestCloseWaitsForAcceptedRequestAndDestroysOnce(t *testing.T) {
	started := make(chan struct{})
	release := make(chan struct{})
	model := &fakeNativeModel{started: started, release: release}
	backend := newTestBackend(t, model)

	scored := make(chan error, 1)
	go func() {
		_, err := backend.Score(context.Background(), validInputs())
		scored <- err
	}()
	<-started
	closed := make(chan error, 1)
	go func() { closed <- backend.Close() }()
	select {
	case err := <-closed:
		t.Fatalf("Close returned before accepted inference finished: %v", err)
	case <-time.After(25 * time.Millisecond):
	}
	close(release)
	if err := <-scored; err != nil {
		t.Fatalf("accepted Score: %v", err)
	}
	if err := <-closed; err != nil {
		t.Fatalf("Close: %v", err)
	}
	if err := backend.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
	if closes := model.closeCount(); closes != 1 {
		t.Fatalf("native Close calls = %d, want 1", closes)
	}
	if _, err := backend.Score(context.Background(), validInputs()); !errors.Is(err, ErrClosed) {
		t.Fatalf("Score after Close error = %v, want ErrClosed", err)
	}
}

func TestClosePropagatesNativeDestroyFailure(t *testing.T) {
	want := errors.New("cudaFree(logits) failed")
	backend := newTestBackend(t, &fakeNativeModel{closeErr: want})
	if err := backend.Close(); !errors.Is(err, want) {
		t.Fatalf("Close error = %v, want %v", err, want)
	}
	if err := backend.Close(); !errors.Is(err, want) {
		t.Fatalf("second Close error = %v, want %v", err, want)
	}
}

func TestCreationFailureIsReturned(t *testing.T) {
	want := errors.New("no approved CUDA device")
	if _, err := newBackend(func() (nativeModel, error) { return nil, want }); !errors.Is(err, want) {
		t.Fatalf("newBackend error = %v, want %v", err, want)
	}
}

func TestInfoComesFromNativeModel(t *testing.T) {
	want := Info{DeviceName: "NVIDIA test", ComputeCapability: "12.0", CUDARuntimeVersion: "13.1", CUDADriverVersion: "595.84"}
	backend := newTestBackend(t, &fakeNativeModel{info: want})
	t.Cleanup(func() { _ = backend.Close() })
	if got := backend.Info(); got != want {
		t.Fatalf("Info() = %#v, want %#v", got, want)
	}
}

func newTestBackend(t *testing.T, model *fakeNativeModel) *backend {
	t.Helper()
	backend, err := newBackend(func() (nativeModel, error) { return model, nil })
	if err != nil {
		t.Fatalf("newBackend: %v", err)
	}
	return backend
}

func validInputs() modelstate.Inputs {
	inputs := modelstate.Inputs{
		CandidateMask:       make([]float32, vocabulary.NumSolutions),
		CandidateStats:      make([]float32, modelstate.CandidateStatsSize),
		Turn:                0,
		RemainingActionMask: make([]float32, vocabulary.NumActions),
	}
	inputs.CandidateMask[0] = 1
	inputs.RemainingActionMask[0] = 1
	return inputs
}

type fakeNativeModel struct {
	mu       sync.Mutex
	info     Info
	calls    int
	closing  int
	inFlight int
	maximum  int
	started  chan struct{}
	release  chan struct{}
	closeErr error
}

func (model *fakeNativeModel) Score(inputs modelstate.Inputs) ([]float32, error) {
	model.mu.Lock()
	model.calls++
	model.inFlight++
	if model.inFlight > model.maximum {
		model.maximum = model.inFlight
	}
	started := model.started
	release := model.release
	model.mu.Unlock()
	if started != nil {
		close(started)
	}
	if release != nil {
		<-release
	}
	logits := make([]float32, vocabulary.NumActions)
	logits[0] = float32(inputs.Turn)
	model.mu.Lock()
	model.inFlight--
	model.mu.Unlock()
	return logits, nil
}

func (model *fakeNativeModel) Info() Info { return model.info }

func (model *fakeNativeModel) Close() error {
	model.mu.Lock()
	defer model.mu.Unlock()
	model.closing++
	return model.closeErr
}

func (model *fakeNativeModel) callCount() int {
	model.mu.Lock()
	defer model.mu.Unlock()
	return model.calls
}

func (model *fakeNativeModel) closeCount() int {
	model.mu.Lock()
	defer model.mu.Unlock()
	return model.closing
}

func (model *fakeNativeModel) maximumConcurrent() int {
	model.mu.Lock()
	defer model.mu.Unlock()
	return model.maximum
}

type firstErrGateContext struct {
	context.Context

	firstErrCalled   chan struct{}
	firstErrReturned chan struct{}
	continueFirstErr chan struct{}
	first            sync.Once
}

func (ctx *firstErrGateContext) Err() error {
	first := false
	ctx.first.Do(func() { first = true })
	if !first {
		return ctx.Context.Err()
	}
	close(ctx.firstErrCalled)
	<-ctx.continueFirstErr
	err := ctx.Context.Err()
	close(ctx.firstErrReturned)
	return err
}

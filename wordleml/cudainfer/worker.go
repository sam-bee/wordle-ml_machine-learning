package cudainfer

import (
	"context"
	"fmt"
	"runtime"
	"sync"

	"github.com/sam-bee/wordle-ml_machine-learning/modelstate"
)

type worker struct {
	requests chan workerRequest

	// stateMu keeps Close from publishing its terminal request until every
	// Score call that began enqueueing has either been accepted or cancelled.
	// It is held only until a score request is on the channel, never through
	// native inference.
	stateMu sync.RWMutex
	closed  bool

	closeOnce sync.Once
	closeErr  error
	info      Info
}

type workerRequest struct {
	score *scoreRequest
	close chan error
}

type scoreRequest struct {
	inputs modelstate.Inputs
	result chan scoreResult
}

type scoreResult struct {
	logits []float32
	err    error
}

type workerReady struct {
	worker *worker
	err    error
}

func newWorker(factory nativeFactory) (*worker, error) {
	ready := make(chan workerReady, 1)
	go func() {
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()

		model, err := factory()
		if err != nil {
			ready <- workerReady{err: fmt.Errorf("create native CUDA model: %w", err)}
			return
		}
		if model == nil {
			ready <- workerReady{err: fmt.Errorf("create native CUDA model: factory returned nil model")}
			return
		}

		worker := &worker{
			requests: make(chan workerRequest, 1),
			info:     model.Info(),
		}
		ready <- workerReady{worker: worker}

		for request := range worker.requests {
			if request.close != nil {
				request.close <- model.Close()
				return
			}
			logits, err := model.Score(request.score.inputs)
			if err == nil {
				err = validateLogits(logits)
			}
			request.score.result <- scoreResult{logits: logits, err: err}
		}
	}()

	result := <-ready
	return result.worker, result.err
}

func (worker *worker) Score(ctx context.Context, inputs modelstate.Inputs) ([]float32, error) {
	if ctx == nil {
		return nil, fmt.Errorf("inference context is nil")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := validateInputs(inputs); err != nil {
		return nil, err
	}

	request := workerRequest{
		score: &scoreRequest{
			inputs: inputs,
			result: make(chan scoreResult, 1),
		},
	}
	worker.stateMu.RLock()
	if worker.closed {
		worker.stateMu.RUnlock()
		return nil, ErrClosed
	}
	// Score may have waited for stateMu after its initial cancellation check.
	// Check again before publishing the request: once the request is on the
	// worker channel, it must complete so that native inference cannot retain
	// Go pointers after this method returns.
	if err := ctx.Err(); err != nil {
		worker.stateMu.RUnlock()
		return nil, err
	}
	select {
	case worker.requests <- request:
		worker.stateMu.RUnlock()
	case <-ctx.Done():
		worker.stateMu.RUnlock()
		return nil, ctx.Err()
	}

	// Do not select on ctx.Done here. The native call might already use the
	// input slices, and waiting preserves the cgo pointer-lifetime guarantee.
	result := <-request.score.result
	return result.logits, result.err
}

func (worker *worker) Info() Info {
	if worker == nil {
		return Info{}
	}
	return worker.info
}

func (worker *worker) Close() error {
	if worker == nil {
		return nil
	}
	worker.closeOnce.Do(func() {
		worker.stateMu.Lock()
		worker.closed = true
		result := make(chan error, 1)
		// This terminal request is enqueued after every Score request that was
		// accepted while holding stateMu's read lock. The locked worker therefore
		// drains those requests before destroying its native handle.
		worker.requests <- workerRequest{close: result}
		worker.stateMu.Unlock()
		worker.closeErr = <-result
	})
	return worker.closeErr
}

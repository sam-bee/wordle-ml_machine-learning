//go:build cuda_cgo

package cudainfer

/*
#cgo CFLAGS: -I${SRCDIR}/../cuda/inference
#cgo LDFLAGS: -L${SRCDIR}/../../build/cuda -L/usr/local/cuda/lib64 -lwordle_cuda -lcudart -lstdc++ -ldl -lpthread
#include "wordle_cuda.h"
*/
import "C"

import (
	"fmt"
	"runtime"
	"strings"
	"unsafe"

	"github.com/sam-bee/wordle-ml_machine-learning/cudamodel"
	"github.com/sam-bee/wordle-ml_machine-learning/modelstate"
)

// Load reads and validates one portable CUDA model before creating its
// C-owned GPU handle on the worker's locked OS thread. It exposes no CPU or
// GoMLX fallback: callers either receive the CUDA backend or an error.
func Load(modelDir string, expected cudamodel.VocabularyHashes) (Backend, cudamodel.Manifest, error) {
	model, err := cudamodel.Load(modelDir, expected)
	if err != nil {
		return nil, cudamodel.Manifest{}, fmt.Errorf("load CUDA model artifact: %w", err)
	}
	weights := model.Weights()
	backend, err := newBackend(func() (nativeModel, error) {
		return newCGOModel(weights)
	})
	if err != nil {
		return nil, cudamodel.Manifest{}, err
	}
	return backend, model.Manifest, nil
}

type cgoModel struct {
	model *C.wordle_cuda_model
	info  Info
}

func newCGOModel(weights []float32) (*cgoModel, error) {
	if len(weights) != cudamodel.ParameterCount {
		return nil, fmt.Errorf("CUDA model has %d weights, want %d", len(weights), cudamodel.ParameterCount)
	}
	var handle *C.wordle_cuda_model
	result := C.wordle_cuda_model_create(
		(*C.float)(unsafe.Pointer(unsafe.SliceData(weights))),
		C.size_t(len(weights)),
		&handle,
	)
	runtime.KeepAlive(weights)
	if result != C.WORDLE_CUDA_OK {
		return nil, cudaError("create CUDA model")
	}
	if handle == nil {
		return nil, fmt.Errorf("create CUDA model: native library returned a nil handle")
	}

	var nativeInfo C.wordle_cuda_model_info
	if result := C.wordle_cuda_model_get_info(handle, &nativeInfo); result != C.WORDLE_CUDA_OK {
		informationErr := cudaError("get CUDA model information")
		if destroyResult := C.wordle_cuda_model_destroy(handle); destroyResult != C.WORDLE_CUDA_OK {
			return nil, fmt.Errorf("%v; cleanup: %v", informationErr, cudaError("destroy CUDA model"))
		}
		return nil, informationErr
	}
	return &cgoModel{model: handle, info: infoFromC(nativeInfo)}, nil
}

func (model *cgoModel) Score(inputs modelstate.Inputs) ([]float32, error) {
	if model == nil || model.model == nil {
		return nil, ErrClosed
	}
	if err := validateInputs(inputs); err != nil {
		return nil, err
	}

	logits := make([]float32, cudamodel.NumActions)
	rc := C.wordle_cuda_model_infer(
		model.model,
		(*C.float)(unsafe.Pointer(unsafe.SliceData(inputs.CandidateMask))),
		(*C.float)(unsafe.Pointer(unsafe.SliceData(inputs.CandidateStats))),
		C.int32_t(inputs.Turn),
		(*C.float)(unsafe.Pointer(unsafe.SliceData(inputs.RemainingActionMask))),
		(*C.float)(unsafe.Pointer(unsafe.SliceData(logits))),
	)
	runtime.KeepAlive(inputs.CandidateMask)
	runtime.KeepAlive(inputs.CandidateStats)
	runtime.KeepAlive(inputs.RemainingActionMask)
	runtime.KeepAlive(logits)
	if rc != C.WORDLE_CUDA_OK {
		return nil, cudaError("run CUDA inference")
	}
	return logits, nil
}

func (model *cgoModel) Info() Info {
	if model == nil {
		return Info{}
	}
	return model.info
}

func (model *cgoModel) Close() error {
	if model == nil || model.model == nil {
		return nil
	}
	result := C.wordle_cuda_model_destroy(model.model)
	model.model = nil
	if result != C.WORDLE_CUDA_OK {
		return cudaError("destroy CUDA model")
	}
	return nil
}

func cudaError(operation string) error {
	message := strings.TrimSpace(C.GoString(C.wordle_cuda_last_error()))
	if message == "" {
		message = "native CUDA library returned no diagnostic"
	}
	return fmt.Errorf("%s: %s", operation, message)
}

func infoFromC(info C.wordle_cuda_model_info) Info {
	return Info{
		DeviceName:         C.GoString(&info.device_name[0]),
		ComputeCapability:  fmt.Sprintf("%d.%d", int(info.compute_major), int(info.compute_minor)),
		CUDARuntimeVersion: cudaVersionString(int(info.cuda_runtime_version)),
		CUDADriverVersion:  cudaVersionString(int(info.cuda_driver_version)),
	}
}

func cudaVersionString(version int) string {
	if version <= 0 {
		return ""
	}
	return fmt.Sprintf("%d.%d", version/1000, version%1000/10)
}

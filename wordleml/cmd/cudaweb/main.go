//go:build cuda_cgo

// Command cudaweb serves the one exported Wordle policy through the
// hand-written CUDA cgo backend. It has no GoMLX, PJRT, or XLA dependency.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"github.com/sam-bee/wordle-ml_machine-learning/cudainfer"
	"github.com/sam-bee/wordle-ml_machine-learning/cudamodel"
	"github.com/sam-bee/wordle-ml_machine-learning/cudaweb"
	"github.com/sam-bee/wordle-ml_machine-learning/modelstate"
	"github.com/sam-bee/wordle-ml_machine-learning/vocabulary"
)

type config struct {
	address  string
	dataDir  string
	modelDir string
}

func main() {
	if err := run(os.Args[1:], os.Stdout, os.Stderr); err != nil {
		fmt.Fprintf(os.Stderr, "cudaweb: %v\n", err)
		os.Exit(1)
	}
}

func run(args []string, stdout, stderr io.Writer) (runErr error) {
	config, err := parseConfig(args, stderr)
	if err != nil {
		return err
	}
	vocab, err := vocabulary.LoadWithoutFinalTest(config.dataDir)
	if err != nil {
		return fmt.Errorf("load validation-only vocabulary: %w", err)
	}
	hashes := vocab.Hashes()
	backend, manifest, err := cudainfer.Load(config.modelDir, cudamodel.VocabularyHashes{
		Solutions: hashes.Solutions,
		Actions:   hashes.Actions,
	})
	if err != nil {
		return err
	}
	var service *cudaweb.Service
	defer func() {
		runErr = errors.Join(runErr, cudaweb.CloseAfterHTTPDrain(service, backend))
	}()
	if err := warmOpening(context.Background(), backend, vocab); err != nil {
		return fmt.Errorf("warm CUDA opening inference: %w", err)
	}
	info := backend.Info()
	service, err = cudaweb.New(cudaweb.Options{
		Vocabulary: vocab,
		Scorer:     backend,
		Model: cudaweb.Model{
			Backend:            "cuda-cgo",
			ModelFormat:        manifest.Format,
			RunID:              manifest.RunID,
			Checkpoint:         manifest.Checkpoint,
			Update:             int64(manifest.CheckpointUpdate),
			TrainingCommit:     manifest.TrainingCommit,
			WeightsSHA256:      manifest.WeightsSHA256,
			ParameterCount:     manifest.ParameterCount,
			DeviceName:         info.DeviceName,
			ComputeCapability:  info.ComputeCapability,
			CUDARuntimeVersion: info.CUDARuntimeVersion,
			CUDADriverVersion:  info.CUDADriverVersion,
		},
	})
	if err != nil {
		return err
	}
	handler, err := cudaweb.NewHandler(service)
	if err != nil {
		return err
	}
	drainingHandler := cudaweb.NewDrainingHandler(handler)

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	server := &http.Server{
		Addr:              config.address,
		Handler:           drainingHandler,
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       10 * time.Second,
		WriteTimeout:      2 * time.Minute,
		IdleTimeout:       60 * time.Second,
	}
	fmt.Fprintf(stdout, "serving CUDA model run=%s checkpoint=%s update=%d device=%q address=%s\n", manifest.RunID, manifest.Checkpoint, manifest.CheckpointUpdate, info.DeviceName, config.address)
	if err := cudaweb.ServeUntilContext(ctx, server, 45*time.Second, drainingHandler); err != nil {
		return err
	}
	return nil
}

func warmOpening(ctx context.Context, backend cudainfer.Backend, vocab *vocabulary.Vocabulary) error {
	if backend == nil || vocab == nil {
		return errors.New("CUDA backend and vocabulary are required")
	}
	bits := make([]byte, modelstate.CandidateBitsetBytes)
	for solutionID := range vocabulary.NumSolutions {
		bits[solutionID/8] |= 1 << uint(solutionID%8)
	}
	encoder, err := modelstate.NewEncoder(vocab)
	if err != nil {
		return err
	}
	inputs, err := encoder.Encode(bits, 0)
	if err != nil {
		return err
	}
	_, err = backend.Score(ctx, inputs)
	return err
}

func parseConfig(args []string, stderr io.Writer) (config, error) {
	dataDir := os.Getenv("WORDLEML_DATA_DIR")
	if dataDir == "" {
		dataDir = "../data"
	}
	address := os.Getenv("WORDLEML_CUDAWEB_ADDR")
	if address == "" {
		address = ":8083"
	}
	modelDir := os.Getenv("WORDLEML_CUDA_MODEL_DIR")
	flags := flag.NewFlagSet("cudaweb", flag.ContinueOnError)
	flags.SetOutput(stderr)
	flags.Usage = func() {
		fmt.Fprintln(stderr, "usage: cudaweb -model-dir=<validated-cuda-export> [flags]")
		flags.PrintDefaults()
	}
	var config config
	flags.StringVar(&config.address, "addr", address, "HTTP listen address")
	flags.StringVar(&config.dataDir, "data-dir", dataDir, "directory containing frozen vocabularies")
	flags.StringVar(&config.modelDir, "model-dir", modelDir, "directory containing manifest.json and weights.f32le")
	if err := flags.Parse(args); err != nil {
		return config, err
	}
	if flags.NArg() != 0 {
		return config, fmt.Errorf("unexpected argument %q", flags.Arg(0))
	}
	if strings.TrimSpace(config.address) == "" {
		return config, errors.New("-addr must not be empty")
	}
	if strings.TrimSpace(config.dataDir) == "" {
		return config, errors.New("-data-dir must not be empty")
	}
	if strings.TrimSpace(config.modelDir) == "" {
		return config, errors.New("-model-dir is required")
	}
	cleanModelDir, err := filepath.Abs(config.modelDir)
	if err != nil {
		return config, fmt.Errorf("make model directory absolute: %w", err)
	}
	config.modelDir = filepath.Clean(cleanModelDir)
	return config, nil
}

//go:build cuda_cgo

// Command cudabench measures one deterministic golden input on the
// hand-written CUDA policy while verifying every output against the reference.
package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"

	"github.com/sam-bee/wordle-ml_machine-learning/cudacheck"
	"github.com/sam-bee/wordle-ml_machine-learning/cudainfer"
	"github.com/sam-bee/wordle-ml_machine-learning/cudamodel"
	"github.com/sam-bee/wordle-ml_machine-learning/cudaref"
	"github.com/sam-bee/wordle-ml_machine-learning/vocabulary"
)

func main() {
	if err := run(os.Args[1:], os.Stdout, os.Stderr); err != nil {
		fmt.Fprintf(os.Stderr, "cudabench: %v\n", err)
		os.Exit(1)
	}
}

func run(args []string, stdout, stderr io.Writer) (resultErr error) {
	configuration, err := parseConfig(args, stderr)
	if err != nil {
		return err
	}
	vocab, err := vocabulary.LoadWithoutFinalTest(configuration.dataDir)
	if err != nil {
		return fmt.Errorf("load sealed vocabulary: %w", err)
	}
	hashes := vocab.Hashes()
	expected := cudamodel.VocabularyHashes{Solutions: hashes.Solutions, Actions: hashes.Actions}
	golden, err := cudaref.LoadGoldenVectors(configuration.modelDir)
	if err != nil {
		return fmt.Errorf("load golden vectors: %w", err)
	}
	if len(golden.Vectors) == 0 {
		return fmt.Errorf("golden vector set is empty")
	}
	backend, manifest, err := cudainfer.Load(configuration.modelDir, expected)
	if err != nil {
		return fmt.Errorf("create CUDA backend: %w", err)
	}
	defer func() {
		if closeErr := backend.Close(); closeErr != nil {
			resultErr = errors.Join(resultErr, fmt.Errorf("close CUDA backend: %w", closeErr))
		}
	}()
	report, benchmarkErr := cudacheck.Benchmark(context.Background(), cudacheck.BenchmarkOptions{
		Backend: backend, Vector: golden.Vectors[0], Tolerance: configuration.tolerance,
		Warmup: configuration.warmup, Iterations: configuration.iterations,
	})
	report.Identity = cudacheck.IdentityFor(manifest, backend.Info())
	if err := cudacheck.WriteJSON(configuration.report, report); err != nil {
		return fmt.Errorf("write benchmark report: %w", err)
	}
	fmt.Fprintf(stdout, "CUDA benchmark: passed=%t vector=%s cold=%dns measured=%d min=%dns mean=%dns p50=%dns p95=%dns max=%dns report=%s\n", report.Passed, report.VectorID, report.ColdCallNS, report.WarmCall.Count, report.WarmCall.Minimum, report.WarmCall.Mean, report.WarmCall.P50, report.WarmCall.P95, report.WarmCall.Maximum, configuration.report)
	if report.Failure != "" {
		fmt.Fprintf(stdout, "failure: %s\n", report.Failure)
	}
	if benchmarkErr != nil {
		return fmt.Errorf("benchmark CUDA model: %w", benchmarkErr)
	}
	if !report.Passed {
		return fmt.Errorf("CUDA benchmark verification failed; inspect %s", configuration.report)
	}
	return nil
}

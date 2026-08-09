//go:build cuda_cgo

// Command cudaverify validates the portable model and hand-written CUDA
// inference against sealed golden logits and the fixed validation games.
package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/signal"
	"reflect"
	"syscall"

	"github.com/sam-bee/wordle-ml_machine-learning/cudacheck"
	"github.com/sam-bee/wordle-ml_machine-learning/cudainfer"
	"github.com/sam-bee/wordle-ml_machine-learning/cudamodel"
	"github.com/sam-bee/wordle-ml_machine-learning/cudaref"
	"github.com/sam-bee/wordle-ml_machine-learning/vocabulary"
)

func main() {
	if err := run(os.Args[1:], os.Stdout, os.Stderr); err != nil {
		fmt.Fprintf(os.Stderr, "cudaverify: %v\n", err)
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
	model, err := cudamodel.Load(configuration.modelDir, expected)
	if err != nil {
		return fmt.Errorf("load portable model: %w", err)
	}
	golden, err := cudaref.LoadGoldenVectors(configuration.modelDir)
	if err != nil {
		return fmt.Errorf("load golden vectors: %w", err)
	}
	games, err := cudacheck.LoadGoldenGames(configuration.modelDir)
	if err != nil {
		return fmt.Errorf("load golden games: %w", err)
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
	if !reflect.DeepEqual(manifest, model.Manifest) {
		return errors.New("CUDA backend manifest differs from portable model manifest")
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	report, verifyErr := cudacheck.Verify(ctx, cudacheck.VerifyOptions{
		Model: model, Vocabulary: vocab, Golden: golden, Games: games, Backend: backend,
		Device: backend.Info(), Tolerance: configuration.tolerance,
	})
	if err := cudacheck.WriteJSON(configuration.report, report); err != nil {
		return fmt.Errorf("write verification report: %w", err)
	}
	printReport(stdout, report, configuration.report)
	if verifyErr != nil {
		return fmt.Errorf("verify CUDA model: %w", verifyErr)
	}
	if !report.Passed {
		return fmt.Errorf("CUDA verification failed; inspect %s", configuration.report)
	}
	return nil
}

func printReport(output io.Writer, report cudacheck.VerificationReport, path string) {
	fmt.Fprintf(output, "CUDA verification: passed=%t vectors=%d games=%d/%d report=%s\n", report.Passed, report.GoldenVectorCount, report.Games.GamesCompared, vocabulary.NumValidationSolutions, path)
	for _, pair := range []struct {
		name    string
		summary cudacheck.PairSummary
	}{
		{"reference<->portable", report.ReferencePortable},
		{"portable<->cuda", report.PortableCUDA},
		{"reference<->cuda", report.ReferenceCUDA},
	} {
		fmt.Fprintf(output, "%s: max=%.9g mean=%.9g worst=%s/%d top1=%d/%d top5-set=%d/%d selected=%d/%d tolerance-failures=%d\n", pair.name, pair.summary.MaximumAbsolute, pair.summary.MeanAbsolute, pair.summary.WorstVectorID, pair.summary.WorstActionID, pair.summary.Top1Agreement, pair.summary.Vectors, pair.summary.Top5SetAgreement, pair.summary.Vectors, pair.summary.SelectedActionAgreement, pair.summary.Vectors, pair.summary.ToleranceFailures)
	}
	if divergence := report.FirstDivergence; divergence != nil {
		fmt.Fprintf(output, "first action divergence: vector=%s pair=%s raw=%d(%.9g)->%d(%.9g) selected=%d->%d margins=%.9g->%.9g near-tie=%t\n", divergence.VectorID, divergence.Pair, divergence.ExpectedRawTop, divergence.ExpectedRawTopLogit, divergence.ActualRawTop, divergence.ActualRawTopLogit, divergence.ExpectedSelected, divergence.ActualSelected, divergence.ExpectedTopTwoMargin, divergence.ActualTopTwoMargin, divergence.NearTie)
	}
	if divergence := report.Games.FirstDivergence; divergence != nil {
		fmt.Fprintf(output, "first game divergence: solution=%s turn=%d field=%s expected=%v actual=%v\n", divergence.Solution, divergence.Turn, divergence.Field, divergence.Expected, divergence.Actual)
	}
	for _, failure := range report.Failures {
		fmt.Fprintf(output, "failure: %s\n", failure)
	}
}

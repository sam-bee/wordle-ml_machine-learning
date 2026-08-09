//go:build cuda_cgo

// Command cudafinal is the intentionally separate, one-shot final-test
// evaluator. It is confirmation-gated, creates an O_EXCL claim before opening
// the test list, and writes sanitized aggregate evidence only.
package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/sam-bee/wordle-ml_machine-learning/cudafinal"
	"github.com/sam-bee/wordle-ml_machine-learning/cudainfer"
	"github.com/sam-bee/wordle-ml_machine-learning/cudamodel"
	"github.com/sam-bee/wordle-ml_machine-learning/gameeval"
	"github.com/sam-bee/wordle-ml_machine-learning/vocabulary"
)

const (
	finalTestReportPath        = "/workspace/artifacts/cuda-cgo/final-test-evaluation.json"
	authorizedRunID            = "seed-replication-20260809-132505Z"
	authorizedCheckpoint       = "best"
	authorizedCheckpointUpdate = 2600
	authorizedTrainingCommit   = "2718164bb80460757592b90aa86b96eb6d596018"
	authorizedWeightsSHA256    = "b78dc980505998d9dd40551ef4d24788b8378be63e4d09fb90aa0a8be83c870d"
	authorizedFinalTestSHA256  = "978a25608a96370b3e26cc8621e9f2cc83ad2d581d07b4b23546b0b4ccdec130"
)

// evaluatorCommit is set only by the cudafinal Go build's -ldflags. An empty
// value makes direct, unproven builds fail before the sealed preflight or the
// one-shot claim can occur.
var evaluatorCommit string

func main() {
	if err := run(os.Args[1:], os.Stdout, os.Stderr); err != nil {
		fmt.Fprintf(os.Stderr, "cudafinal: %v\n", err)
		os.Exit(1)
	}
}

func run(args []string, stdout, stderr io.Writer) (resultErr error) {
	configuration, err := parseConfig(args, stderr)
	if err != nil {
		return err
	}
	commit, err := embeddedEvaluatorCommit()
	if err != nil {
		return err
	}
	// This preflight deliberately uses only the sealed loader. It validates the
	// CUDA artifact and device before the one-shot claim and final-test read.
	sealed, err := vocabulary.LoadWithoutFinalTest(configuration.dataDir)
	if err != nil {
		return fmt.Errorf("load sealed vocabulary for CUDA preflight: %w", err)
	}
	sealedHashes := sealed.Hashes()
	backend, manifest, err := cudainfer.Load(configuration.modelDir, cudamodel.VocabularyHashes{
		Solutions: sealedHashes.Solutions,
		Actions:   sealedHashes.Actions,
	})
	if err != nil {
		return fmt.Errorf("create CUDA backend before final-test claim: %w", err)
	}
	if err := validateAuthorizedModel(manifest); err != nil {
		if closeErr := backend.Close(); closeErr != nil {
			return errors.Join(err, fmt.Errorf("close CUDA backend: %w", closeErr))
		}
		return err
	}
	backendOpen := true
	claimMade := false
	var started cudafinal.Report
	defer func() {
		if !backendOpen {
			return
		}
		backendOpen = false
		if closeErr := backend.Close(); closeErr != nil {
			resultErr = errors.Join(resultErr, fmt.Errorf("close CUDA backend: %w", closeErr))
			if claimMade {
				if reportErr := replaceFailure(finalTestReportPath, started, cudafinal.FailureBackendClose); reportErr != nil {
					resultErr = errors.Join(resultErr, reportErr)
				}
			}
		}
	}()

	started, err = cudafinal.NewStarted(time.Now(), commit, manifest, backend.Info())
	if err != nil {
		return fmt.Errorf("create final-test evaluator claim: %w", err)
	}
	if err := cudafinal.Claim(finalTestReportPath, started); err != nil {
		return err
	}
	claimMade = true

	// This is the sole intentional final-test word-list read. Do not pass its
	// errors through verbatim: vocabulary or game errors can include a word.
	vocab, err := vocabulary.Load(configuration.dataDir)
	if err != nil {
		return fail(finalTestReportPath, started, cudafinal.FailureVocabularyLoad, errors.New("load final-test vocabulary failed"))
	}
	hashes := vocab.Hashes()
	if hashes.Solutions != sealedHashes.Solutions || hashes.Actions != sealedHashes.Actions || !authorizedFinalTest(hashes.Test, len(vocab.Test())) {
		return fail(finalTestReportPath, started, cudafinal.FailureVocabularyIdentity, errors.New("final-test vocabulary identity is incomplete or differs from CUDA preflight"))
	}
	evaluator, err := gameeval.New(gameeval.Config{
		Vocabulary: vocab,
		Score: func(ctx context.Context, position gameeval.Position) ([]float32, error) {
			return backend.Score(ctx, position.Inputs)
		},
		// CandidateSolutions intentionally remains empty: gameeval's fixed
		// default is the complete 2,309-solution candidate population.
	})
	if err != nil {
		return fail(finalTestReportPath, started, cudafinal.FailureEvaluatorSetup, errors.New("create final-test gameplay evaluator failed"))
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	evaluation, err := evaluator.Evaluate(ctx, vocab.Test())
	if err != nil {
		return fail(finalTestReportPath, started, cudafinal.FailureGameplay, errors.New("final-test gameplay evaluation failed"))
	}
	if len(evaluation.Games) != vocabulary.NumTestSolutions || evaluation.Summary.Games != vocabulary.NumTestSolutions {
		return fail(finalTestReportPath, started, cudafinal.FailureGameCount, errors.New("final-test gameplay did not evaluate exactly 100 solutions"))
	}
	report, err := cudafinal.Complete(started, time.Now(), hashes.Test, evaluation.Summary)
	if err != nil {
		return fail(finalTestReportPath, started, cudafinal.FailureReportValidation, errors.New("validate final-test aggregate report failed"))
	}
	if err := cudafinal.Replace(finalTestReportPath, report); err != nil {
		return fmt.Errorf("publish complete final-test report: %w", err)
	}

	// Close before reporting success, so a close failure is propagated and the
	// aggregate record is replaced with a safe failure state instead.
	backendOpen = false
	if err := backend.Close(); err != nil {
		if reportErr := replaceFailure(finalTestReportPath, started, cudafinal.FailureBackendClose); reportErr != nil {
			return errors.Join(fmt.Errorf("close CUDA backend: %w", err), reportErr)
		}
		return fmt.Errorf("close CUDA backend: %w", err)
	}
	printReport(stdout, report, finalTestReportPath)
	return nil
}

func embeddedEvaluatorCommit() (string, error) {
	if evaluatorCommit == "" {
		return "", errors.New("cudafinal was built without an embedded evaluator commit")
	}
	return evaluatorCommit, nil
}

func validateAuthorizedModel(manifest cudamodel.Manifest) error {
	if manifest.RunID != authorizedRunID || manifest.Checkpoint != authorizedCheckpoint || manifest.CheckpointUpdate != authorizedCheckpointUpdate ||
		manifest.TrainingCommit != authorizedTrainingCommit || manifest.WeightsSHA256 != authorizedWeightsSHA256 {
		return errors.New("CUDA model is not the one authorized for final-test evaluation")
	}
	return nil
}

func authorizedFinalTest(hash string, count int) bool {
	return hash == authorizedFinalTestSHA256 && count == vocabulary.NumTestSolutions
}

func fail(path string, started cudafinal.Report, code cudafinal.FailureCode, result error) error {
	if reportErr := replaceFailure(path, started, code); reportErr != nil {
		return errors.Join(result, reportErr)
	}
	return result
}

func replaceFailure(path string, started cudafinal.Report, code cudafinal.FailureCode) error {
	report, err := cudafinal.Failed(started, time.Now(), code)
	if err != nil {
		return fmt.Errorf("construct final-test failure report: %w", err)
	}
	if err := cudafinal.Replace(path, report); err != nil {
		return fmt.Errorf("publish final-test failure report: %w", err)
	}
	return nil
}

func printReport(output io.Writer, report cudafinal.Report, path string) {
	aggregate := report.Aggregate
	if aggregate == nil {
		return
	}
	fmt.Fprintf(output, "CUDA final test: games=%d solved=%d fraction=%.6f mean-guesses=%.6f failures=%d invalid=%d suppressed-raw-top=%d repeated=%d distribution=%v report=%s\n", aggregate.Games, aggregate.Solved, aggregate.SolvedFraction, aggregate.MeanGuesses, aggregate.Failures, aggregate.InvalidSelections, aggregate.SuppressedRawTopSelections, aggregate.RepeatedSelections, aggregate.GuessCountDistribution, path)
}

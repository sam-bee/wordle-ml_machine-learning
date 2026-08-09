// Package cudafinal owns the deliberately small, one-shot final-test report.
//
// It retains aggregate gameplay facts only. In particular it never stores a
// solution word, a guess, a trajectory, or a failed-solution list.
package cudafinal

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/sam-bee/wordle-ml_game-engine/game"
	"github.com/sam-bee/wordle-ml_machine-learning/cudainfer"
	"github.com/sam-bee/wordle-ml_machine-learning/cudamodel"
	"github.com/sam-bee/wordle-ml_machine-learning/gameeval"
	"github.com/sam-bee/wordle-ml_machine-learning/vocabulary"
)

const (
	// Format identifies the sanitized final-test report schema.
	Format = "wordle-cuda-final-test"
	// Version changes only when the JSON schema changes incompatibly.
	Version = 1
)

// Status records the one-shot evaluator lifecycle. A pre-existing started,
// failed, or complete report means the final-test word list must not be read
// again by this evaluator.
type Status string

const (
	StatusStarted  Status = "started"
	StatusFailed   Status = "failed"
	StatusComplete Status = "complete"
)

// FailureCode is one deliberately opaque, fixed reason for an interrupted
// one-shot evaluation. Its closed set prevents accidental persistence of a
// withheld word through an arbitrary error message.
type FailureCode string

const (
	FailureVocabularyLoad     FailureCode = "final_test_vocabulary_load_failed"
	FailureVocabularyIdentity FailureCode = "final_test_vocabulary_identity_failed"
	FailureEvaluatorSetup     FailureCode = "final_test_evaluator_setup_failed"
	FailureGameplay           FailureCode = "final_test_gameplay_failed"
	FailureGameCount          FailureCode = "final_test_game_count_failed"
	FailureReportValidation   FailureCode = "final_test_report_validation_failed"
	FailureBackendClose       FailureCode = "cuda_backend_close_failed"
)

// ModelIdentity is the model provenance safe to publish with aggregate test
// results. It intentionally omits the artifact directory and tensor values.
type ModelIdentity struct {
	Format           string `json:"format"`
	RunID            string `json:"run_id"`
	Checkpoint       string `json:"checkpoint"`
	CheckpointUpdate int    `json:"checkpoint_update"`
	TrainingCommit   string `json:"training_commit"`
	WeightsSHA256    string `json:"weights_sha256"`
	ParameterCount   int    `json:"parameter_count"`
}

// Aggregate contains only count-based outcomes from final-test gameplay.
// There is deliberately no FailedSolutions field.
type Aggregate struct {
	Games                      int                `json:"games"`
	Solved                     int                `json:"solved"`
	SolvedFraction             float64            `json:"solved_fraction"`
	MeanGuesses                float64            `json:"mean_guesses"`
	GuessCountDistribution     [game.MaxTurns]int `json:"guess_count_distribution"`
	Failures                   int                `json:"failures"`
	InvalidSelections          int                `json:"invalid_selections"`
	SuppressedRawTopSelections int                `json:"suppressed_raw_top_selections"`
	RepeatedSelections         int                `json:"repeated_selections"`
}

// Report is the sole on-disk record from the intentional final-test read.
// Failure is a fixed opaque code, never an error string, because evaluator
// error text may contain a withheld word.
type Report struct {
	Format          string         `json:"format"`
	Version         int            `json:"version"`
	Status          Status         `json:"status"`
	TimestampUTC    time.Time      `json:"timestamp_utc"`
	EvaluatorCommit string         `json:"evaluator_commit"`
	Model           ModelIdentity  `json:"model"`
	Device          cudainfer.Info `json:"device"`
	FinalTestSHA256 string         `json:"final_test_sha256,omitempty"`
	Aggregate       *Aggregate     `json:"aggregate,omitempty"`
	Failure         FailureCode    `json:"failure,omitempty"`
}

// NewStarted creates the durable pre-test claim record. The command must
// Claim it before calling vocabulary.Load, which is the only intentional read
// of the final-test list.
func NewStarted(timestamp time.Time, evaluatorCommit string, manifest cudamodel.Manifest, device cudainfer.Info) (Report, error) {
	identity, err := modelIdentity(manifest)
	if err != nil {
		return Report{}, err
	}
	if err := validateEvaluatorCommit(evaluatorCommit); err != nil {
		return Report{}, err
	}
	if strings.TrimSpace(device.DeviceName) == "" || strings.TrimSpace(device.ComputeCapability) == "" {
		return Report{}, errors.New("CUDA device identity is incomplete")
	}
	if timestamp.IsZero() {
		return Report{}, errors.New("report timestamp is required")
	}
	return Report{
		Format:          Format,
		Version:         Version,
		Status:          StatusStarted,
		TimestampUTC:    timestamp.UTC(),
		EvaluatorCommit: evaluatorCommit,
		Model:           identity,
		Device:          device,
	}, nil
}

// Complete turns a durable started claim into its sanitized aggregate result.
func Complete(started Report, timestamp time.Time, finalTestSHA256 string, summary gameeval.Summary) (Report, error) {
	if err := validateStarted(started); err != nil {
		return Report{}, err
	}
	if timestamp.IsZero() {
		return Report{}, errors.New("report timestamp is required")
	}
	if err := validateSHA256("final-test", finalTestSHA256); err != nil {
		return Report{}, err
	}
	aggregate, err := aggregateFromSummary(summary)
	if err != nil {
		return Report{}, err
	}
	result := started
	result.Status = StatusComplete
	result.TimestampUTC = timestamp.UTC()
	result.FinalTestSHA256 = finalTestSHA256
	result.Aggregate = &aggregate
	result.Failure = ""
	return result, nil
}

// Failed turns a started claim into a safe, non-retryable failure record. The
// caller supplies one of its own fixed failure codes, not an error message.
func Failed(started Report, timestamp time.Time, failure FailureCode) (Report, error) {
	if err := validateStarted(started); err != nil {
		return Report{}, err
	}
	if timestamp.IsZero() {
		return Report{}, errors.New("report timestamp is required")
	}
	if !validFailureCode(failure) {
		return Report{}, fmt.Errorf("unsupported final-test failure code %q", failure)
	}
	result := started
	result.Status = StatusFailed
	result.TimestampUTC = timestamp.UTC()
	result.Failure = failure
	return result, nil
}

// Claim writes a started report through O_CREATE|O_EXCL. A prior attempt is
// always a hard error, before the final-test vocabulary is opened.
func Claim(path string, started Report) error {
	if err := validateStarted(started); err != nil {
		return err
	}
	if strings.TrimSpace(path) == "" {
		return errors.New("final-test report path is required")
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("create final-test report directory: %w", err)
	}
	contents, err := encode(started)
	if err != nil {
		return err
	}
	file, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o644)
	if err != nil {
		if errors.Is(err, os.ErrExist) {
			return fmt.Errorf("final-test evaluator claim %q already exists; refusing another final-test read", path)
		}
		return fmt.Errorf("create final-test evaluator claim: %w", err)
	}
	if _, err := file.Write(contents); err != nil {
		_ = file.Close()
		return fmt.Errorf("write final-test evaluator claim: %w", err)
	}
	if err := file.Sync(); err != nil {
		_ = file.Close()
		return fmt.Errorf("sync final-test evaluator claim: %w", err)
	}
	if err := file.Close(); err != nil {
		return fmt.Errorf("close final-test evaluator claim: %w", err)
	}
	if err := syncDirectory(filepath.Dir(path)); err != nil {
		return fmt.Errorf("sync final-test report directory: %w", err)
	}
	return nil
}

// Replace atomically updates a previously claimed report with failed or
// complete state. It intentionally cannot create an unclaimed report.
func Replace(path string, report Report) error {
	if strings.TrimSpace(path) == "" {
		return errors.New("final-test report path is required")
	}
	if err := validateReplacement(report); err != nil {
		return err
	}
	if _, err := os.Lstat(path); err != nil {
		return fmt.Errorf("inspect final-test evaluator claim: %w", err)
	}
	contents, err := encode(report)
	if err != nil {
		return err
	}
	parent := filepath.Dir(path)
	temporary, err := os.CreateTemp(parent, ".final-test-*.tmp")
	if err != nil {
		return fmt.Errorf("create temporary final-test report: %w", err)
	}
	temporaryPath := temporary.Name()
	defer func() { _ = os.Remove(temporaryPath) }()
	if _, err := temporary.Write(contents); err != nil {
		_ = temporary.Close()
		return fmt.Errorf("write temporary final-test report: %w", err)
	}
	if err := temporary.Chmod(0o644); err != nil {
		_ = temporary.Close()
		return fmt.Errorf("set temporary final-test report permissions: %w", err)
	}
	if err := temporary.Sync(); err != nil {
		_ = temporary.Close()
		return fmt.Errorf("sync temporary final-test report: %w", err)
	}
	if err := temporary.Close(); err != nil {
		return fmt.Errorf("close temporary final-test report: %w", err)
	}
	if err := os.Rename(temporaryPath, path); err != nil {
		return fmt.Errorf("publish final-test report: %w", err)
	}
	return syncDirectory(parent)
}

func modelIdentity(manifest cudamodel.Manifest) (ModelIdentity, error) {
	if manifest.Format != cudamodel.Format || strings.TrimSpace(manifest.RunID) == "" || strings.TrimSpace(manifest.Checkpoint) == "" ||
		strings.TrimSpace(manifest.TrainingCommit) == "" || manifest.CheckpointUpdate < 0 || manifest.ParameterCount != cudamodel.ParameterCount {
		return ModelIdentity{}, errors.New("CUDA model manifest identity is incomplete or incompatible")
	}
	if err := validateSHA256("weights", manifest.WeightsSHA256); err != nil {
		return ModelIdentity{}, err
	}
	return ModelIdentity{
		Format:           manifest.Format,
		RunID:            manifest.RunID,
		Checkpoint:       manifest.Checkpoint,
		CheckpointUpdate: manifest.CheckpointUpdate,
		TrainingCommit:   manifest.TrainingCommit,
		WeightsSHA256:    manifest.WeightsSHA256,
		ParameterCount:   manifest.ParameterCount,
	}, nil
}

func aggregateFromSummary(summary gameeval.Summary) (Aggregate, error) {
	if summary.Games != vocabulary.NumTestSolutions || summary.Solved < 0 || summary.Solved > summary.Games || summary.Failures != summary.Games-summary.Solved ||
		math.IsNaN(summary.SolvedFraction) || math.IsInf(summary.SolvedFraction, 0) || math.IsNaN(summary.MeanGuesses) || math.IsInf(summary.MeanGuesses, 0) ||
		summary.SolvedFraction < 0 || summary.SolvedFraction > 1 || summary.MeanGuesses < 1 || summary.MeanGuesses > game.MaxTurns ||
		summary.InvalidSelections < 0 || summary.SuppressedRawTopSelections < 0 || summary.RepeatedSelections < 0 {
		return Aggregate{}, errors.New("final-test gameplay summary is invalid")
	}
	if math.Abs(summary.SolvedFraction-float64(summary.Solved)/float64(summary.Games)) > 1e-12 {
		return Aggregate{}, errors.New("final-test solved fraction differs from solved/game counts")
	}
	distributionTotal, guessTotal := 0, 0
	for turn, count := range summary.GuessCountDistribution {
		if count < 0 {
			return Aggregate{}, errors.New("final-test guess distribution contains a negative count")
		}
		distributionTotal += count
		guessTotal += (turn + 1) * count
	}
	if distributionTotal != summary.Games {
		return Aggregate{}, errors.New("final-test guess distribution does not total the game count")
	}
	if math.Abs(summary.MeanGuesses-float64(guessTotal)/float64(summary.Games)) > 1e-12 {
		return Aggregate{}, errors.New("final-test mean guesses differs from the distribution")
	}
	return Aggregate{
		Games:                      summary.Games,
		Solved:                     summary.Solved,
		SolvedFraction:             summary.SolvedFraction,
		MeanGuesses:                summary.MeanGuesses,
		GuessCountDistribution:     summary.GuessCountDistribution,
		Failures:                   summary.Failures,
		InvalidSelections:          summary.InvalidSelections,
		SuppressedRawTopSelections: summary.SuppressedRawTopSelections,
		RepeatedSelections:         summary.RepeatedSelections,
	}, nil
}

func validateStarted(report Report) error {
	if report.Format != Format || report.Version != Version || report.Status != StatusStarted || report.TimestampUTC.IsZero() ||
		report.TimestampUTC.Location() != time.UTC || report.FinalTestSHA256 != "" || report.Aggregate != nil || report.Failure != "" {
		return errors.New("final-test report is not a valid started claim")
	}
	if err := validateEvaluatorCommit(report.EvaluatorCommit); err != nil {
		return err
	}
	if report.Model.Format != cudamodel.Format || report.Model.RunID == "" || report.Model.Checkpoint == "" || report.Model.TrainingCommit == "" ||
		report.Model.CheckpointUpdate < 0 || report.Model.ParameterCount != cudamodel.ParameterCount {
		return errors.New("final-test report model identity is incomplete")
	}
	if err := validateSHA256("weights", report.Model.WeightsSHA256); err != nil {
		return err
	}
	if strings.TrimSpace(report.Device.DeviceName) == "" || strings.TrimSpace(report.Device.ComputeCapability) == "" {
		return errors.New("final-test report CUDA device identity is incomplete")
	}
	return nil
}

func validateReplacement(report Report) error {
	started := report
	started.Status = StatusStarted
	started.FinalTestSHA256 = ""
	started.Aggregate = nil
	started.Failure = ""
	if err := validateStarted(started); err != nil {
		return err
	}
	switch report.Status {
	case StatusFailed:
		if report.FinalTestSHA256 != "" || report.Aggregate != nil || !validFailureCode(report.Failure) {
			return errors.New("final-test failure report is not sanitized")
		}
	case StatusComplete:
		if report.Failure != "" || report.Aggregate == nil {
			return errors.New("final-test complete report is incomplete")
		}
		if err := validateSHA256("final-test", report.FinalTestSHA256); err != nil {
			return err
		}
		summary := gameeval.Summary{
			Games: report.Aggregate.Games, Solved: report.Aggregate.Solved, SolvedFraction: report.Aggregate.SolvedFraction,
			MeanGuesses: report.Aggregate.MeanGuesses, GuessCountDistribution: report.Aggregate.GuessCountDistribution,
			Failures: report.Aggregate.Failures, InvalidSelections: report.Aggregate.InvalidSelections,
			SuppressedRawTopSelections: report.Aggregate.SuppressedRawTopSelections, RepeatedSelections: report.Aggregate.RepeatedSelections,
		}
		if _, err := aggregateFromSummary(summary); err != nil {
			return err
		}
	default:
		return fmt.Errorf("replace final-test report with status %q", report.Status)
	}
	return nil
}

func validFailureCode(code FailureCode) bool {
	switch code {
	case FailureVocabularyLoad, FailureVocabularyIdentity, FailureEvaluatorSetup, FailureGameplay, FailureGameCount, FailureReportValidation, FailureBackendClose:
		return true
	default:
		return false
	}
}

func validateEvaluatorCommit(commit string) error {
	if len(commit) != 40 {
		return fmt.Errorf("evaluator commit has %d characters, want 40", len(commit))
	}
	if _, err := hex.DecodeString(commit); err != nil || commit != strings.ToLower(commit) {
		return errors.New("evaluator commit must be lowercase hexadecimal")
	}
	return nil
}

func validateSHA256(label, value string) error {
	if len(value) != sha256.Size*2 {
		return fmt.Errorf("%s SHA-256 has %d characters, want %d", label, len(value), sha256.Size*2)
	}
	if _, err := hex.DecodeString(value); err != nil || value != strings.ToLower(value) {
		return fmt.Errorf("%s SHA-256 must be lowercase hexadecimal", label)
	}
	return nil
}

func encode(report Report) ([]byte, error) {
	contents, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		return nil, fmt.Errorf("encode final-test report: %w", err)
	}
	return append(contents, '\n'), nil
}

func syncDirectory(path string) (resultErr error) {
	directory, err := os.Open(path)
	if err != nil {
		return err
	}
	defer func() {
		if closeErr := directory.Close(); closeErr != nil && resultErr == nil {
			resultErr = closeErr
		}
	}()
	return directory.Sync()
}

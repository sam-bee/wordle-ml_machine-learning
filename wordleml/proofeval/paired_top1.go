package proofeval

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"

	"github.com/gomlx/compute"
	"github.com/sam-bee/wordle-ml_machine-learning/imitationdata"
	"github.com/sam-bee/wordle-ml_machine-learning/modelstate"
	"github.com/sam-bee/wordle-ml_machine-learning/proofrun"
	"github.com/sam-bee/wordle-ml_machine-learning/runmetadata"
	"github.com/sam-bee/wordle-ml_machine-learning/runstate"
	"github.com/sam-bee/wordle-ml_machine-learning/supervised"
	"github.com/sam-bee/wordle-ml_machine-learning/vocabulary"
)

// PairedTop1ArtifactName is the immutable validation-state comparison written
// into the replication run after both best checkpoints are independently
// restored. It contains no final-test examples or predictions.
const PairedTop1ArtifactName = "best-paired-top1.json"

// PairedTop1Options identifies the predeclared original production run and its
// one fixed seed replication.
type PairedTop1Options struct {
	DataDir          string
	RunsDir          string
	OriginalRunID    string
	ReplicationRunID string
}

// PairedTop1Comparison counts ordered validation states by whether each best
// checkpoint selected the teacher top-1 action. The two hashes commit to the
// complete ordered boolean vectors without bloating the report artifact.
type PairedTop1Comparison struct {
	OriginalRunID         string `json:"original_run_id"`
	ReplicationRunID      string `json:"replication_run_id"`
	OriginalSeed          int64  `json:"original_seed"`
	ReplicationSeed       int64  `json:"replication_seed"`
	OriginalBestUpdate    int64  `json:"original_best_update"`
	ReplicationBestUpdate int64  `json:"replication_best_update"`
	ValidationSplitHash   string `json:"validation_split_hash"`
	Examples              int    `json:"examples"`
	BothTeacherTop1       int    `json:"both_teacher_top1"`
	OriginalOnly          int    `json:"original_only_teacher_top1"`
	ReplicationOnly       int    `json:"replication_only_teacher_top1"`
	NeitherTeacherTop1    int    `json:"neither_teacher_top1"`
	OriginalSelectionHash string `json:"original_teacher_top1_vector_sha256"`
	ReplicationHash       string `json:"replication_teacher_top1_vector_sha256"`
}

// CompareBestTeacherTop1 independently restores both best checkpoints, runs
// the frozen validation records in their on-disk order, and atomically writes
// their paired teacher-top-1 comparison into the replication run.
func CompareBestTeacherTop1(ctx context.Context, options PairedTop1Options) (PairedTop1Comparison, error) {
	if strings.TrimSpace(options.DataDir) == "" || strings.TrimSpace(options.RunsDir) == "" || strings.TrimSpace(options.OriginalRunID) == "" || strings.TrimSpace(options.ReplicationRunID) == "" {
		return PairedTop1Comparison{}, errors.New("data directory, runs directory, original run ID, and replication run ID are required")
	}
	if options.OriginalRunID == options.ReplicationRunID {
		return PairedTop1Comparison{}, errors.New("original and replication run IDs must differ")
	}
	originalLayout, err := runstate.Open(options.RunsDir, options.OriginalRunID)
	if err != nil {
		return PairedTop1Comparison{}, fmt.Errorf("open original run: %w", err)
	}
	replicationLayout, err := runstate.Open(options.RunsDir, options.ReplicationRunID)
	if err != nil {
		return PairedTop1Comparison{}, fmt.Errorf("open replication run: %w", err)
	}
	originalConfig, err := ReadConfig(originalLayout)
	if err != nil {
		return PairedTop1Comparison{}, fmt.Errorf("read original config: %w", err)
	}
	replicationConfig, err := ReadConfig(replicationLayout)
	if err != nil {
		return PairedTop1Comparison{}, fmt.Errorf("read replication config: %w", err)
	}
	if originalConfig.Stage != proofrun.Production || originalConfig.Seed != proofrun.Seed {
		return PairedTop1Comparison{}, errors.New("original run is not the fixed production configuration")
	}
	if replicationConfig.Stage != proofrun.SeedReplication || replicationConfig.Seed != proofrun.SeedReplicationSeed {
		return PairedTop1Comparison{}, errors.New("replication run is not the fixed seed-replication configuration")
	}
	normalized := replicationConfig
	normalized.Stage, normalized.Seed = originalConfig.Stage, originalConfig.Seed
	if normalized != originalConfig {
		return PairedTop1Comparison{}, errors.New("replication configuration differs from production beyond stage identity and seed")
	}
	if err := validateEvaluationTrainingComplete(originalLayout, originalConfig.Stage); err != nil {
		return PairedTop1Comparison{}, fmt.Errorf("original training completion: %w", err)
	}
	if err := validateEvaluationTrainingComplete(replicationLayout, replicationConfig.Stage); err != nil {
		return PairedTop1Comparison{}, fmt.Errorf("replication training completion: %w", err)
	}
	originalManifest, err := readPairedComparisonManifest(originalLayout, options.DataDir)
	if err != nil {
		return PairedTop1Comparison{}, fmt.Errorf("original metadata: %w", err)
	}
	replicationManifest, err := readPairedComparisonManifest(replicationLayout, options.DataDir)
	if err != nil {
		return PairedTop1Comparison{}, fmt.Errorf("replication metadata: %w", err)
	}
	if !samePairedInputs(originalManifest, replicationManifest) {
		return PairedTop1Comparison{}, errors.New("original and replication runs do not record identical frozen train/validation inputs")
	}
	if !replicationManifest.FinalTestSealed || len(replicationManifest.Splits.Test) != 0 {
		return PairedTop1Comparison{}, errors.New("replication metadata does not prove the final-test wordlist remained unopened")
	}
	originalFinal, err := readPairedFinal(originalLayout, proofrun.Production)
	if err != nil {
		return PairedTop1Comparison{}, fmt.Errorf("original final metrics: %w", err)
	}
	replicationFinal, err := readPairedFinal(replicationLayout, proofrun.SeedReplication)
	if err != nil {
		return PairedTop1Comparison{}, fmt.Errorf("replication final metrics: %w", err)
	}
	vocab, err := vocabulary.LoadWithoutFinalTest(options.DataDir)
	if err != nil {
		return PairedTop1Comparison{}, fmt.Errorf("load vocabulary: %w", err)
	}
	validation, err := imitationdata.Load(vocab, filepath.Join(options.DataDir, "imitation"), imitationdata.Validation)
	if err != nil {
		return PairedTop1Comparison{}, fmt.Errorf("load validation data: %w", err)
	}
	backend, err := compute.NewWithConfig("xla:cuda")
	if err != nil {
		return PairedTop1Comparison{}, fmt.Errorf("create xla:cuda backend: %w", err)
	}
	defer backend.Finalize()
	for name, manifest := range map[string]runmetadata.Manifest{"original": originalManifest, "replication": replicationManifest} {
		if err := runmetadata.VerifyEvaluationRuntime(manifest, backend.Name(), backend.Description()); err != nil {
			return PairedTop1Comparison{}, fmt.Errorf("verify %s evaluation runtime: %w", name, err)
		}
	}
	originalSelections, err := bestTeacherTop1Selections(ctx, backend, originalLayout, originalConfig, originalFinal, vocab, validation)
	if err != nil {
		return PairedTop1Comparison{}, fmt.Errorf("evaluate original best selections: %w", err)
	}
	replicationSelections, err := bestTeacherTop1Selections(ctx, backend, replicationLayout, replicationConfig, replicationFinal, vocab, validation)
	if err != nil {
		return PairedTop1Comparison{}, fmt.Errorf("evaluate replication best selections: %w", err)
	}
	if len(originalSelections) != len(replicationSelections) || len(originalSelections) != validation.Len() {
		return PairedTop1Comparison{}, errors.New("paired selection vectors do not cover the complete validation split")
	}
	result := PairedTop1Comparison{
		OriginalRunID: options.OriginalRunID, ReplicationRunID: options.ReplicationRunID,
		OriginalSeed: originalConfig.Seed, ReplicationSeed: replicationConfig.Seed,
		OriginalBestUpdate: originalFinal.BestValidationStep, ReplicationBestUpdate: replicationFinal.BestValidationStep,
		ValidationSplitHash: vocab.Hashes().Validation, Examples: len(originalSelections),
		OriginalSelectionHash: selectionHash(originalSelections), ReplicationHash: selectionHash(replicationSelections),
	}
	for index := range originalSelections {
		switch {
		case originalSelections[index] && replicationSelections[index]:
			result.BothTeacherTop1++
		case originalSelections[index]:
			result.OriginalOnly++
		case replicationSelections[index]:
			result.ReplicationOnly++
		default:
			result.NeitherTeacherTop1++
		}
	}
	if result.BothTeacherTop1+result.OriginalOnly != exactTop1Count(originalFinal.BestValidation.Top1, result.Examples) {
		return PairedTop1Comparison{}, errors.New("original paired selections do not reproduce saved top-1 accuracy")
	}
	if result.BothTeacherTop1+result.ReplicationOnly != exactTop1Count(replicationFinal.BestValidation.Top1, result.Examples) {
		return PairedTop1Comparison{}, errors.New("replication paired selections do not reproduce saved top-1 accuracy")
	}
	if err := replicationLayout.WriteEvaluationJSON("best", "paired-top1", result); err != nil {
		return PairedTop1Comparison{}, fmt.Errorf("write immutable paired top-1 artifact: %w", err)
	}
	return result, nil
}

func readPairedComparisonManifest(layout runstate.Layout, dataDir string) (runmetadata.Manifest, error) {
	contents, err := os.ReadFile(layout.MetadataPath)
	if err != nil {
		return runmetadata.Manifest{}, err
	}
	var manifest runmetadata.Manifest
	if err := json.Unmarshal(contents, &manifest); err != nil {
		return runmetadata.Manifest{}, err
	}
	if err := runmetadata.VerifyEvaluationInputs(manifest, dataDir); err != nil {
		return runmetadata.Manifest{}, err
	}
	return manifest, nil
}

func samePairedInputs(left, right runmetadata.Manifest) bool {
	return left.ModelParameterCount == 1_046_596 && right.ModelParameterCount == left.ModelParameterCount &&
		reflect.DeepEqual(left.Dataset, right.Dataset) && reflect.DeepEqual(left.Vocabulary, right.Vocabulary) &&
		reflect.DeepEqual(left.Splits.Training, right.Splits.Training) && reflect.DeepEqual(left.Splits.Validation, right.Splits.Validation)
}

func readPairedFinal(layout runstate.Layout, stage proofrun.Stage) (proofrun.Result, error) {
	contents, err := os.ReadFile(layout.FinalMetricsPath)
	if err != nil {
		return proofrun.Result{}, err
	}
	var result proofrun.Result
	if err := json.Unmarshal(contents, &result); err != nil {
		return proofrun.Result{}, err
	}
	if result.Stage != stage || !result.Passed || result.GlobalUpdate != 10_000 || result.BestValidationStep < 0 || result.BestValidationStep > result.GlobalUpdate {
		return proofrun.Result{}, errors.New("run lacks a passed fixed 10,000-update result")
	}
	return result, nil
}

func bestTeacherTop1Selections(ctx context.Context, backend compute.Backend, layout runstate.Layout, config proofrun.Config, final proofrun.Result, vocab *vocabulary.Vocabulary, validation *imitationdata.Data) ([]bool, error) {
	session, state, err := LoadSession(backend, layout, Best, config)
	if err != nil {
		return nil, err
	}
	defer session.Finalize()
	if state.GlobalUpdate != final.BestValidationStep || state.BestValidation == nil || state.BestValidation.Update != final.BestValidationStep || math.Abs(state.BestValidation.Value-final.BestValidation.Loss) > LossTolerance {
		return nil, errors.New("restored best checkpoint state differs from saved best metrics")
	}
	return teacherTop1Selections(ctx, session, validation)
}

func teacherTop1Selections(ctx context.Context, session *supervised.Session, validation *imitationdata.Data) ([]bool, error) {
	if session == nil || validation == nil || validation.Split() != imitationdata.Validation {
		return nil, errors.New("session and frozen validation data are required")
	}
	if err := warmValidation(session, validation, Normal, modelstate.Inputs{}); err != nil {
		return nil, err
	}
	before, err := proofrun.StoreFingerprint(session.Store)
	if err != nil {
		return nil, err
	}
	selections := make([]bool, 0, validation.Len())
	for start := 0; start < validation.Len(); start += validationBatchSize {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		end := min(start+validationBatchSize, validation.Len())
		examples := make([]imitationdata.Example, 0, end-start)
		for index := start; index < end; index++ {
			example, err := validation.Example(index)
			if err != nil {
				return nil, err
			}
			examples = append(examples, example)
		}
		rawRows, maskedRows, err := predictBatch(session, examples, Normal, modelstate.Inputs{})
		if err != nil {
			return nil, err
		}
		for offset, example := range examples {
			if err := validatePrediction(rawRows[offset], maskedRows[offset], example.AvailableActionMask); err != nil {
				return nil, fmt.Errorf("validation example %d: %w", start+offset, err)
			}
			prediction := argMax(maskedRows[offset])
			if prediction < 0 {
				return nil, fmt.Errorf("validation example %d has no legal predicted action", start+offset)
			}
			selections = append(selections, prediction == int(example.TeacherTopAction))
		}
	}
	after, err := proofrun.StoreFingerprint(session.Store)
	if err != nil {
		return nil, err
	}
	if before != after {
		return nil, fmt.Errorf("paired validation inference mutated Store: before %s, after %s", before, after)
	}
	return selections, nil
}

func selectionHash(selections []bool) string {
	encoded := make([]byte, len(selections))
	for index, selected := range selections {
		if selected {
			encoded[index] = 1
		}
	}
	sum := sha256.Sum256(encoded)
	return hex.EncodeToString(sum[:])
}

func exactTop1Count(accuracy float64, examples int) int {
	return int(math.Round(accuracy * float64(examples)))
}

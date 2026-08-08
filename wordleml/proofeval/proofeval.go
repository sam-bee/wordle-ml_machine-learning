// Package proofeval independently reloads proof checkpoints and evaluates
// them against the sealed validation split.  It intentionally has no training
// entry point and never opens the test split.
package proofeval

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"reflect"

	"github.com/gomlx/compute"
	_ "github.com/gomlx/gomlx/backends/default"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/sam-bee/wordle-ml_machine-learning/imitationdata"
	"github.com/sam-bee/wordle-ml_machine-learning/modelstate"
	"github.com/sam-bee/wordle-ml_machine-learning/policy"
	"github.com/sam-bee/wordle-ml_machine-learning/proofgames"
	"github.com/sam-bee/wordle-ml_machine-learning/proofmetrics"
	"github.com/sam-bee/wordle-ml_machine-learning/proofrun"
	"github.com/sam-bee/wordle-ml_machine-learning/runmetadata"
	"github.com/sam-bee/wordle-ml_machine-learning/runstate"
	"github.com/sam-bee/wordle-ml_machine-learning/supervised"
	"github.com/sam-bee/wordle-ml_machine-learning/tensorboard"
	"github.com/sam-bee/wordle-ml_machine-learning/vocabulary"
)

// Checkpoint selects an independently reloadable run checkpoint.
type Checkpoint string

const (
	Initial Checkpoint = "initial"
	Latest  Checkpoint = "latest"
	Best    Checkpoint = "best"
)

// Mode intentionally exposes only the fixed post-training checks in the
// proof plan.
type Mode string

const (
	Games10   Mode = "games10"
	Games100  Mode = "games100"
	Ablations Mode = "ablations"
)

// Options identifies one already-completed proof run. DataDir is the frozen
// data root; it is used only for vocabulary and the validation WDIT split.
type Options struct {
	DataDir    string
	RunsDir    string
	RunID      string
	Checkpoint Checkpoint
	Mode       Mode
}

// Result is JSON-safe output for the evaluation command. It is retained both
// as a standalone evaluation artifact and under final-metrics.json/evaluations
// after training has completed.
type Result struct {
	RunID               string                 `json:"run_id"`
	Stage               proofrun.Stage         `json:"stage"`
	Mode                Mode                   `json:"mode"`
	Checkpoint          Checkpoint             `json:"checkpoint"`
	CheckpointUpdate    int64                  `json:"checkpoint_update"`
	ValidationSplitHash string                 `json:"validation_split_hash"`
	Validation          proofmetrics.Result    `json:"validation,omitempty"`
	BestLoss            *LossReproduction      `json:"best_loss_reproduction,omitempty"`
	Games               *proofgames.Evaluation `json:"games,omitempty"`
	Ablations           *AblationResult        `json:"ablations,omitempty"`
}

// LossReproduction documents independent re-evaluation of the persisted best
// validation checkpoint, including all top-k metrics and its major groups.
type LossReproduction struct {
	Stored      proofrun.Metrics `json:"stored"`
	Measured    proofrun.Metrics `json:"measured"`
	Tolerance   float64          `json:"tolerance"`
	GroupsMatch bool             `json:"major_groups_match"`
}

// AblationResult reports no-retraining validation inference measurements.
// Effects are signed ablated-minus-normal values, so negative loss is an
// improvement whereas negative top-k agreement is a degradation.
type AblationResult struct {
	Normal                proofmetrics.Result `json:"normal"`
	OpeningCandidateState AblationMeasurement `json:"opening_candidate_state"`
	TurnZero              AblationMeasurement `json:"turn_zero"`
	NoCandidateBonus      AblationMeasurement `json:"no_candidate_bonus"`
}

type AblationMeasurement struct {
	Validation proofmetrics.Result `json:"validation"`
	Effect     MetricEffect        `json:"effect_vs_normal"`
}

type MetricEffect struct {
	Loss  float64 `json:"loss"`
	Top1  float64 `json:"top1_accuracy"`
	Top5  float64 `json:"top5_accuracy"`
	Top16 float64 `json:"top16_accuracy"`
}

const LossTolerance = 2e-4

// validationBatchSize keeps independent validation inference reasonably quick
// without changing the fixed record order or mean-of-examples aggregation.
const validationBatchSize = 100

// Run loads a checkpoint into a newly-created CUDA session, evaluates it, and
// returns structured output. It never opens imitationdata.Test.
func Run(ctx context.Context, options Options) (Result, error) {
	if err := validateOptions(options); err != nil {
		return Result{}, err
	}
	layout, err := runstate.Open(options.RunsDir, options.RunID)
	if err != nil {
		return Result{}, fmt.Errorf("open run: %w", err)
	}
	config, err := readConfig(layout)
	if err != nil {
		return Result{}, err
	}
	if err := validateCombination(config.Stage, options.Checkpoint, options.Mode); err != nil {
		return Result{}, err
	}
	if err := validateEvaluationTrainingComplete(layout, config.Stage); err != nil {
		return Result{}, err
	}
	manifest, err := verifyEvaluationMetadata(layout, options.DataDir)
	if err != nil {
		return Result{}, err
	}
	vocab, err := vocabulary.Load(options.DataDir)
	if err != nil {
		return Result{}, fmt.Errorf("load vocabulary: %w", err)
	}
	validation, err := imitationdata.Load(vocab, filepath.Join(options.DataDir, "imitation"), imitationdata.Validation)
	if err != nil {
		return Result{}, fmt.Errorf("load validation data: %w", err)
	}
	backend, err := compute.NewWithConfig("xla:cuda")
	if err != nil {
		return Result{}, fmt.Errorf("create xla:cuda backend: %w", err)
	}
	defer backend.Finalize()
	if err := runmetadata.VerifyEvaluationRuntime(manifest, backend.Name(), backend.Description()); err != nil {
		return Result{}, fmt.Errorf("verify evaluation runtime identity: %w", err)
	}
	session, state, err := LoadSession(backend, layout, options.Checkpoint, config)
	if err != nil {
		return Result{}, err
	}
	defer session.Finalize()

	result := Result{RunID: options.RunID, Stage: config.Stage, Mode: options.Mode, Checkpoint: options.Checkpoint, CheckpointUpdate: state.GlobalUpdate, ValidationSplitHash: vocab.Hashes().Validation}
	normal, err := EvaluateValidation(session, vocab, validation, Normal)
	if err != nil {
		return Result{}, err
	}
	result.Validation = normal
	switch options.Mode {
	case Games10, Games100:
		count := 10
		if options.Mode == Games100 {
			count = vocabulary.NumValidationSolutions
		}
		beforeGames, err := proofrun.StoreFingerprint(session.Store)
		if err != nil {
			return Result{}, fmt.Errorf("fingerprint checkpoint before games: %w", err)
		}
		evaluation, err := EvaluateGames(ctx, session, vocab, vocab.Validation()[:count])
		if err != nil {
			return Result{}, err
		}
		afterGames, err := proofrun.StoreFingerprint(session.Store)
		if err != nil {
			return Result{}, fmt.Errorf("fingerprint checkpoint after games: %w", err)
		}
		if beforeGames != afterGames {
			return Result{}, fmt.Errorf("game inference mutated Store: before %s, after %s", beforeGames, afterGames)
		}
		gamesJSONL, err := proofgames.JSONL(evaluation.Games)
		if err != nil {
			return Result{}, err
		}
		if err := layout.WriteEvaluationGamesJSONL(string(options.Checkpoint), string(options.Mode), gamesJSONL); err != nil {
			return Result{}, err
		}
		if isCanonicalReportTrajectory(config.Stage, options.Checkpoint, options.Mode) {
			if err := layout.WriteValidationGamesJSONL(gamesJSONL); err != nil {
				return Result{}, err
			}
		}
		result.Games = &evaluation
		if err := writeGameScalars(layout.EventsDir, state.GlobalUpdate, evaluation); err != nil {
			return Result{}, err
		}
	case Ablations:
		ablations, err := evaluateAblationsFromNormal(session, vocab, validation, normal)
		if err != nil {
			return Result{}, err
		}
		result.Ablations = &ablations
	default:
		return Result{}, fmt.Errorf("unsupported evaluation mode %q", options.Mode)
	}
	if options.Checkpoint == Best {
		measurement, err := checkBestReproduction(normal, state, layout)
		if err != nil {
			return Result{}, err
		}
		result.BestLoss = &measurement
	}
	if err := layout.WriteEvaluationJSON(string(options.Checkpoint), string(options.Mode), result); err != nil {
		return Result{}, err
	}
	if err := persistEvaluation(layout, result); err != nil {
		return Result{}, err
	}
	return result, nil
}

// validateEvaluationTrainingComplete is intentionally the first evaluation
// preflight after config parsing: a failed mini/full gate must not load data,
// create a CUDA backend, or publish any evaluation artifact.
func validateEvaluationTrainingComplete(layout runstate.Layout, stage proofrun.Stage) error {
	if stage != proofrun.Mini && stage != proofrun.Full {
		return nil
	}
	contents, err := os.ReadFile(layout.FinalMetricsPath)
	if err != nil {
		return fmt.Errorf("read final metrics before evaluation: %w", err)
	}
	var training proofrun.Result
	if err := json.Unmarshal(contents, &training); err != nil {
		return fmt.Errorf("decode final metrics before evaluation: %w", err)
	}
	if training.Stage != stage {
		return fmt.Errorf("final metrics stage %q differs from requested evaluation stage %q", training.Stage, stage)
	}
	if !training.Passed {
		return fmt.Errorf("refusing evaluation before %s training has passed", stage)
	}
	return nil
}

func isCanonicalReportTrajectory(stage proofrun.Stage, checkpoint Checkpoint, mode Mode) bool {
	return (stage == proofrun.Mini && checkpoint == Latest && mode == Games10) ||
		(stage == proofrun.Full && checkpoint == Best && mode == Games100)
}

// verifyEvaluationMetadata runs before vocabulary.Load or imitationdata.Load,
// so a changed validation WDIT file or word list cannot reach inference.
func verifyEvaluationMetadata(layout runstate.Layout, dataDir string) (runmetadata.Manifest, error) {
	contents, err := os.ReadFile(layout.MetadataPath)
	if err != nil {
		return runmetadata.Manifest{}, fmt.Errorf("read immutable run metadata: %w", err)
	}
	var manifest runmetadata.Manifest
	if err := json.Unmarshal(contents, &manifest); err != nil {
		return runmetadata.Manifest{}, fmt.Errorf("decode immutable run metadata: %w", err)
	}
	if err := runmetadata.VerifyEvaluationInputs(manifest, dataDir); err != nil {
		return runmetadata.Manifest{}, fmt.Errorf("verify evaluation data identity before load: %w", err)
	}
	if err := runmetadata.VerifyEvaluationRepositories(manifest,
		os.Getenv("WORDLEML_MACHINE_LEARNING_REPO_DIR"),
		os.Getenv("WORDLEML_SYNTHETIC_REPO_DIR"),
		os.Getenv("WORDLEML_GAME_ENGINE_REPO_DIR"),
	); err != nil {
		return runmetadata.Manifest{}, fmt.Errorf("verify evaluation repository identity: %w", err)
	}
	return manifest, nil
}

// persistEvaluation adds a raw full evaluation result to final-metrics.json
// without decoding/re-encoding unrelated fields. The runner owns the typed
// Result structure; this post-training writer preserves both future fields and
// earlier evaluations while atomically replacing the enclosing JSON document.
func persistEvaluation(layout runstate.Layout, result Result) error {
	contents, err := os.ReadFile(layout.FinalMetricsPath)
	if err != nil {
		return fmt.Errorf("read final metrics for evaluation persistence: %w", err)
	}
	var training proofrun.Result
	if err := json.Unmarshal(contents, &training); err != nil {
		return fmt.Errorf("decode final metrics for evaluation persistence: %w", err)
	}
	if training.Stage != result.Stage {
		return fmt.Errorf("final metrics stage %q differs from evaluation stage %q", training.Stage, result.Stage)
	}
	if (training.Stage == proofrun.Mini || training.Stage == proofrun.Full) && !training.Passed {
		return fmt.Errorf("refusing to persist %s evaluation before %s training has passed", result.Mode, training.Stage)
	}
	var document map[string]json.RawMessage
	if err := json.Unmarshal(contents, &document); err != nil {
		return fmt.Errorf("decode final metrics JSON object: %w", err)
	}
	if document == nil {
		return errors.New("final metrics JSON must be an object")
	}
	evaluations := make(map[string]json.RawMessage)
	if prior, found := document["evaluations"]; found {
		if err := json.Unmarshal(prior, &evaluations); err != nil {
			return fmt.Errorf("decode existing final-metrics evaluations: %w", err)
		}
	}
	encoded, err := json.Marshal(result)
	if err != nil {
		return fmt.Errorf("encode evaluation result: %w", err)
	}
	key := evaluationKey(result.Checkpoint, result.Mode)
	if existing, found := evaluations[key]; found {
		same, err := sameJSON(existing, encoded)
		if err != nil {
			return fmt.Errorf("compare existing evaluation %q: %w", key, err)
		}
		if !same {
			return fmt.Errorf("immutable final-metrics evaluation %q already exists with different contents", key)
		}
		return nil
	}
	evaluations[key] = encoded
	document["evaluations"], err = json.Marshal(evaluations)
	if err != nil {
		return fmt.Errorf("encode final-metrics evaluations: %w", err)
	}
	if err := layout.WriteFinalMetricsJSON(document); err != nil {
		return fmt.Errorf("atomically persist evaluation in final metrics: %w", err)
	}
	return nil
}

// sameJSON accepts byte-identical retry payloads and semantically identical
// JSON representations (for example the existing pretty-printed artifact).
func sameJSON(first, second []byte) (bool, error) {
	if bytes.Equal(first, second) {
		return true, nil
	}
	var left, right any
	if err := json.Unmarshal(first, &left); err != nil {
		return false, err
	}
	if err := json.Unmarshal(second, &right); err != nil {
		return false, err
	}
	return reflect.DeepEqual(left, right), nil
}

func evaluationKey(checkpoint Checkpoint, mode Mode) string {
	return string(checkpoint) + "-" + string(mode)
}

func validateOptions(options Options) error {
	if options.DataDir == "" || options.RunsDir == "" || options.RunID == "" {
		return errors.New("data directory, runs directory, and run ID are required")
	}
	if options.Checkpoint != Initial && options.Checkpoint != Latest && options.Checkpoint != Best {
		return fmt.Errorf("checkpoint must be %q, %q, or %q", Initial, Latest, Best)
	}
	if options.Mode != Games10 && options.Mode != Games100 && options.Mode != Ablations {
		return fmt.Errorf("mode must be %q, %q, or %q", Games10, Games100, Ablations)
	}
	return nil
}

func validateCombination(stage proofrun.Stage, checkpoint Checkpoint, mode Mode) error {
	switch mode {
	case Games10:
		if stage == proofrun.Mini && checkpoint == Latest {
			return nil
		}
	case Games100, Ablations:
		if stage == proofrun.Full && checkpoint == Best {
			return nil
		}
		if mode == Games100 && stage == proofrun.Full && checkpoint == Initial {
			return nil
		}
	}
	return fmt.Errorf("evaluation mode %q is not allowed for stage %q checkpoint %q", mode, stage, checkpoint)
}

func readConfig(layout runstate.Layout) (proofrun.Config, error) {
	contents, err := os.ReadFile(layout.ConfigPath)
	if err != nil {
		return proofrun.Config{}, fmt.Errorf("read run config: %w", err)
	}
	var config proofrun.Config
	if err := json.Unmarshal(contents, &config); err != nil {
		return proofrun.Config{}, fmt.Errorf("decode run config: %w", err)
	}
	want, err := proofrun.ConfigFor(config.Stage)
	if err != nil || config != want {
		return proofrun.Config{}, errors.New("run config is not one of the fixed proof configurations")
	}
	return config, nil
}

// LoadSession creates a fresh policy store and restores precisely the selected
// checkpoint directory. It has no relationship with the training process's
// Session or Store.
func LoadSession(backend compute.Backend, layout runstate.Layout, which Checkpoint, config proofrun.Config) (*supervised.Session, runstate.State, error) {
	if which != Initial && which != Latest && which != Best {
		return nil, runstate.State{}, fmt.Errorf("unknown checkpoint %q", which)
	}
	dir := layout.LatestCheckpointDir
	if which == Initial {
		dir = layout.InitialCheckpointDir
	}
	if which == Best {
		dir = layout.BestCheckpointDir
	}
	session, err := supervised.New(supervised.Config{
		Policy:       policy.Config{NumSolutions: vocabulary.NumSolutions, NumActions: vocabulary.NumActions},
		LearningRate: config.LearningRate,
		Seed:         config.Seed,
	}, backend)
	if err != nil {
		return nil, runstate.State{}, fmt.Errorf("create fresh inference session: %w", err)
	}
	if _, err := supervised.NewCheckpoint(session.Store, dir); err != nil {
		session.Finalize()
		return nil, runstate.State{}, fmt.Errorf("load %s checkpoint: %w", which, err)
	}
	state, err := runstate.LoadCheckpointState(session.Store)
	if err != nil {
		session.Finalize()
		return nil, runstate.State{}, fmt.Errorf("read restored checkpoint state: %w", err)
	}
	if session.Trainer.GlobalStep() != state.GlobalUpdate {
		session.Finalize()
		return nil, runstate.State{}, fmt.Errorf("restored global step %d differs from run state %d", session.Trainer.GlobalStep(), state.GlobalUpdate)
	}
	return session, state, nil
}

func checkBestReproduction(metrics proofmetrics.Result, state runstate.State, layout runstate.Layout) (LossReproduction, error) {
	if state.BestValidation == nil {
		return LossReproduction{}, errors.New("best checkpoint has no recorded best validation loss")
	}
	stored := proofrun.Metrics{Loss: state.BestValidation.Value}
	var storedDetails proofmetrics.Result
	if layout.FinalMetricsPath != "" {
		contents, err := os.ReadFile(layout.FinalMetricsPath)
		if err != nil {
			return LossReproduction{}, fmt.Errorf("read saved final metrics: %w", err)
		}
		var final proofrun.Result
		if err := json.Unmarshal(contents, &final); err != nil {
			return LossReproduction{}, fmt.Errorf("decode saved final metrics: %w", err)
		}
		if final.BestValidationStep != state.BestValidation.Update {
			return LossReproduction{}, fmt.Errorf("saved best update %d differs from best checkpoint state %d", final.BestValidationStep, state.BestValidation.Update)
		}
		if !closeMetric(final.BestValidation.Loss, state.BestValidation.Value) {
			return LossReproduction{}, fmt.Errorf("saved best loss %g differs from checkpoint state %g", final.BestValidation.Loss, state.BestValidation.Value)
		}
		stored, storedDetails = final.BestValidation, final.BestValidationDetails.Details
	}
	measured := proofrun.Metrics{Loss: metrics.Loss, Top1: metrics.Top1Accuracy, Top5: metrics.Top5Accuracy, Top16: metrics.Top16Accuracy}
	if !closeMetric(stored.Loss, measured.Loss) || !closeMetric(stored.Top1, measured.Top1) || !closeMetric(stored.Top5, measured.Top5) || !closeMetric(stored.Top16, measured.Top16) {
		return LossReproduction{}, fmt.Errorf("best checkpoint metrics %+v differ from saved %+v (tolerance %g)", measured, stored, LossTolerance)
	}
	groupsMatch := true
	if layout.FinalMetricsPath != "" {
		if err := compareMajorGroups(storedDetails, metrics); err != nil {
			return LossReproduction{}, err
		}
	}
	return LossReproduction{Stored: stored, Measured: measured, Tolerance: LossTolerance, GroupsMatch: groupsMatch}, nil
}

func closeMetric(first, second float64) bool { return math.Abs(first-second) <= LossTolerance }

func compareMajorGroups(stored, measured proofmetrics.Result) error {
	for _, groupSet := range [][2][]proofmetrics.GroupResult{{stored.ByTurn, measured.ByTurn}, {stored.ByShortlistBucket, measured.ByShortlistBucket}} {
		if len(groupSet[0]) != len(groupSet[1]) || len(groupSet[0]) == 0 {
			return errors.New("saved best validation lacks major group details")
		}
		for index, expected := range groupSet[0] {
			actual := groupSet[1][index]
			if expected.Name != actual.Name || expected.Examples != actual.Examples || !closeMetric(expected.Loss, actual.Loss) || !closeMetric(expected.Top1Accuracy, actual.Top1Accuracy) || !closeMetric(expected.Top5Accuracy, actual.Top5Accuracy) || !closeMetric(expected.Top16Accuracy, actual.Top16Accuracy) {
				return fmt.Errorf("best checkpoint group %q differs from saved result: got %+v want %+v", actual.Name, actual, expected)
			}
		}
	}
	return nil
}

// Variant describes a host-side, no-retraining inference change.
type Variant string

const (
	Normal                Variant = "normal"
	OpeningCandidateState Variant = "opening_candidate_state"
	TurnZero              Variant = "turn_zero"
	NoCandidateBonus      Variant = "no_candidate_bonus"
)

// EvaluateAblations evaluates every validation record in its fixed on-disk
// order. The opening representation is generated by the production encoder,
// rather than hand-built, so it exactly means all possible solutions at turn 0.
func EvaluateAblations(session *supervised.Session, vocab *vocabulary.Vocabulary, validation *imitationdata.Data) (AblationResult, error) {
	normal, err := EvaluateValidation(session, vocab, validation, Normal)
	if err != nil {
		return AblationResult{}, err
	}
	return evaluateAblationsFromNormal(session, vocab, validation, normal)
}

func evaluateAblationsFromNormal(session *supervised.Session, vocab *vocabulary.Vocabulary, validation *imitationdata.Data, normal proofmetrics.Result) (AblationResult, error) {
	opening, err := EvaluateValidation(session, vocab, validation, OpeningCandidateState)
	if err != nil {
		return AblationResult{}, err
	}
	turn, err := EvaluateValidation(session, vocab, validation, TurnZero)
	if err != nil {
		return AblationResult{}, err
	}
	bonus, err := EvaluateValidation(session, vocab, validation, NoCandidateBonus)
	if err != nil {
		return AblationResult{}, err
	}
	return AblationResult{
		Normal:                normal,
		OpeningCandidateState: measurement(normal, opening),
		TurnZero:              measurement(normal, turn),
		NoCandidateBonus:      measurement(normal, bonus),
	}, nil
}

func measurement(normal, ablated proofmetrics.Result) AblationMeasurement {
	return AblationMeasurement{Validation: ablated, Effect: MetricEffect{
		Loss: ablated.Loss - normal.Loss, Top1: ablated.Top1Accuracy - normal.Top1Accuracy,
		Top5: ablated.Top5Accuracy - normal.Top5Accuracy, Top16: ablated.Top16Accuracy - normal.Top16Accuracy,
	}}
}

// EvaluateValidation performs exact, ordered host aggregation over all records
// in the frozen validation data. No Trainer API is used, which also makes it a
// hard guard that this post-checkpoint path cannot update model parameters.
func EvaluateValidation(session *supervised.Session, vocab *vocabulary.Vocabulary, validation *imitationdata.Data, variant Variant) (proofmetrics.Result, error) {
	if session == nil || vocab == nil || validation == nil {
		return proofmetrics.Result{}, errors.New("session, vocabulary, and validation data are required")
	}
	if validation.Split() != imitationdata.Validation {
		return proofmetrics.Result{}, fmt.Errorf("validation evaluation refuses split %q", validation.Split())
	}
	if variant != Normal && variant != OpeningCandidateState && variant != TurnZero && variant != NoCandidateBonus {
		return proofmetrics.Result{}, fmt.Errorf("unknown validation variant %q", variant)
	}
	var opening modelstate.Inputs
	if variant == OpeningCandidateState {
		encoded, err := openingInputs(vocab)
		if err != nil {
			return proofmetrics.Result{}, err
		}
		opening = encoded
	}
	if err := warmValidation(session, validation, variant, opening); err != nil {
		return proofmetrics.Result{}, fmt.Errorf("warm validation inference: %w", err)
	}
	before, err := proofrun.StoreFingerprint(session.Store)
	if err != nil {
		return proofmetrics.Result{}, fmt.Errorf("fingerprint before validation: %w", err)
	}
	collector := &proofmetrics.Collector{}
	for start := 0; start < validation.Len(); start += validationBatchSize {
		end := min(start+validationBatchSize, validation.Len())
		examples := make([]imitationdata.Example, 0, end-start)
		for index := start; index < end; index++ {
			example, err := validation.Example(index)
			if err != nil {
				return proofmetrics.Result{}, err
			}
			examples = append(examples, example)
		}
		rawRows, maskedRows, err := predictBatch(session, examples, variant, opening)
		if err != nil {
			return proofmetrics.Result{}, fmt.Errorf("validation examples %d..%d: %w", start, end-1, err)
		}
		for offset, example := range examples {
			index, raw, masked := start+offset, rawRows[offset], maskedRows[offset]
			if err := validatePrediction(raw, masked, example.AvailableActionMask); err != nil {
				return proofmetrics.Result{}, fmt.Errorf("validation example %d: %w", index, err)
			}
			loss, err := crossEntropy(masked, int(example.TeacherTopAction))
			if err != nil {
				return proofmetrics.Result{}, fmt.Errorf("validation example %d: %w", index, err)
			}
			prediction := argMax(masked)
			if prediction < 0 {
				return proofmetrics.Result{}, fmt.Errorf("validation example %d: no legal predicted action", index)
			}
			if err := collector.Add(proofmetrics.Sample{
				Loss: loss, Logits: raw, MaskedPrediction: &prediction, TeacherTopActions: example.TeacherTopActions,
				Turn: int(example.Turn), CandidateCount: countCandidates(example.CandidateMask), Availability: example.AvailableActionMask,
			}); err != nil {
				return proofmetrics.Result{}, fmt.Errorf("validation example %d: %w", index, err)
			}
		}
	}
	result := collector.Result()
	after, err := proofrun.StoreFingerprint(session.Store)
	if err != nil {
		return proofmetrics.Result{}, fmt.Errorf("fingerprint after validation: %w", err)
	}
	if before != after {
		return proofmetrics.Result{}, fmt.Errorf("validation inference mutated Store: before %s, after %s", before, after)
	}
	if result.Examples != validation.Len() || result.MaskedArgmaxViolations != 0 {
		return proofmetrics.Result{}, fmt.Errorf("validation masking sanity check failed: examples=%d/%d masked violations=%d", result.Examples, validation.Len(), result.MaskedArgmaxViolations)
	}
	return result, nil
}

func warmValidation(session *supervised.Session, validation *imitationdata.Data, variant Variant, opening modelstate.Inputs) error {
	end := min(validationBatchSize, validation.Len())
	examples := make([]imitationdata.Example, 0, end)
	for index := 0; index < end; index++ {
		example, err := validation.Example(index)
		if err != nil {
			return err
		}
		examples = append(examples, example)
	}
	_, _, err := predictBatch(session, examples, variant, opening)
	return err
}

func openingInputs(vocab *vocabulary.Vocabulary) (modelstate.Inputs, error) {
	if vocab == nil {
		return modelstate.Inputs{}, errors.New("vocabulary must not be nil")
	}
	bits := make([]byte, modelstate.CandidateBitsetBytes)
	for solutionID := 0; solutionID < vocabulary.NumSolutions; solutionID++ {
		bits[solutionID/8] |= 1 << uint(solutionID%8)
	}
	encoder, err := modelstate.NewEncoder(vocab)
	if err != nil {
		return modelstate.Inputs{}, err
	}
	return encoder.Encode(bits, 0)
}

// predictBatch preserves the caller's example order and makes one CUDA call
// per fixed batch. The output row slices are independent from GoMLX tensors.
func predictBatch(session *supervised.Session, examples []imitationdata.Example, variant Variant, opening modelstate.Inputs) (rawRows, maskedRows [][]float32, err error) {
	if len(examples) == 0 {
		return nil, nil, errors.New("cannot predict an empty validation batch")
	}
	candidateMasks := make([][]float32, len(examples))
	stats := make([][]float32, len(examples))
	turns := make([]int32, len(examples))
	remaining := make([][]float32, len(examples))
	available := make([][]float32, len(examples))
	for index, example := range examples {
		candidateMasks[index], stats[index], turns[index], remaining[index], available[index] = example.CandidateMask, example.CandidateStats, example.Turn, example.RemainingActionMask, example.AvailableActionMask
		if variant == OpeningCandidateState {
			candidateMasks[index], stats[index], remaining[index] = opening.CandidateMask, opening.CandidateStats, opening.RemainingActionMask
		}
		if variant == TurnZero {
			turns[index] = 0
		}
	}
	rawTensor, maskedTensor, betaTensor, err := session.PredictDiagnostics(candidateMasks, stats, turns, remaining, available)
	if err != nil {
		return nil, nil, err
	}
	defer func() { _ = rawTensor.FinalizeAll(); _ = maskedTensor.FinalizeAll(); _ = betaTensor.FinalizeAll() }()
	raw, err := tensors.CopyFlatData[float32](rawTensor)
	if err != nil {
		return nil, nil, err
	}
	masked, err := tensors.CopyFlatData[float32](maskedTensor)
	if err != nil {
		return nil, nil, err
	}
	betas, err := tensors.CopyFlatData[float32](betaTensor)
	if err != nil {
		return nil, nil, err
	}
	if len(raw) != len(examples)*vocabulary.NumActions || len(masked) != len(raw) || len(betas) != len(examples) {
		return nil, nil, errors.New("inference output shape does not match validation batch")
	}
	rawRows, maskedRows = make([][]float32, len(examples)), make([][]float32, len(examples))
	for index, example := range examples {
		start := index * vocabulary.NumActions
		rawRows[index], maskedRows[index] = raw[start:start+vocabulary.NumActions], masked[start:start+vocabulary.NumActions]
		if variant == NoCandidateBonus {
			for action := range rawRows[index] {
				rawRows[index][action] -= betas[index] * example.RemainingActionMask[action]
			}
			maskedRows[index] = applyHostMask(rawRows[index], example.AvailableActionMask)
		}
	}
	return rawRows, maskedRows, nil
}

func applyHostMask(raw, availability []float32) []float32 {
	masked := append([]float32(nil), raw...)
	for index := range masked {
		if availability[index] == 0 {
			masked[index] = float32(math.Inf(-1))
		}
	}
	return masked
}

func validatePrediction(raw, masked, availability []float32) error {
	if len(raw) != vocabulary.NumActions || len(masked) != vocabulary.NumActions || len(availability) != vocabulary.NumActions {
		return errors.New("prediction has incorrect vocabulary width")
	}
	for action := range raw {
		if !finite(raw[action]) {
			return fmt.Errorf("raw logit %d is non-finite", action)
		}
		if availability[action] == 0 {
			if !math.IsInf(float64(masked[action]), -1) {
				return fmt.Errorf("masked action %d is selectable", action)
			}
		} else if !finite(masked[action]) {
			return fmt.Errorf("legal action %d has non-finite masked logit", action)
		}
	}
	return nil
}

func crossEntropy(logits []float32, target int) (float64, error) {
	if target < 0 || target >= len(logits) || math.IsInf(float64(logits[target]), -1) {
		return 0, errors.New("teacher target is invalid or masked")
	}
	maximum := float32(math.Inf(-1))
	for _, value := range logits {
		if value > maximum {
			maximum = value
		}
	}
	var sum float64
	for _, value := range logits {
		if !math.IsInf(float64(value), -1) {
			sum += math.Exp(float64(value - maximum))
		}
	}
	if sum == 0 || math.IsInf(sum, 0) || math.IsNaN(sum) {
		return 0, errors.New("cross entropy normalization is non-finite")
	}
	loss := math.Log(sum) - float64(logits[target]-maximum)
	if math.IsNaN(loss) || math.IsInf(loss, 0) {
		return 0, errors.New("cross entropy is non-finite")
	}
	return loss, nil
}

func argMax(values []float32) int {
	best := -1
	for i, value := range values {
		if !finite(value) {
			continue
		}
		if best < 0 || value > values[best] {
			best = i
		}
	}
	return best
}
func countCandidates(mask []float32) int {
	count := 0
	for _, value := range mask {
		if value != 0 {
			count++
		}
	}
	return count
}
func finite(value float32) bool { return !math.IsNaN(float64(value)) && !math.IsInf(float64(value), 0) }

// EvaluateGames runs deterministic greedy evaluation through the shared
// proofgames package. The scorer returns raw finite logits; proofgames/gameeval
// owns the legal/repeated-guess mask and records suppressed raw preferences.
func EvaluateGames(ctx context.Context, session *supervised.Session, vocab *vocabulary.Vocabulary, solutions []string) (proofgames.Evaluation, error) {
	return proofgames.EvaluateSession(ctx, session, vocab, solutions)
}

func writeGameScalars(eventsDir string, step int64, evaluation proofgames.Evaluation) error {
	expected := proofgames.TensorBoardScalars(evaluation)
	present, err := matchingGameScalarsExist(eventsDir, step, expected)
	if err != nil {
		return err
	}
	if present {
		return nil
	}
	writer, err := tensorboard.New(eventsDir)
	if err != nil {
		return fmt.Errorf("open game TensorBoard events: %w", err)
	}
	if err := writer.WriteScalars(step, expected...); err != nil {
		_ = writer.Close()
		return fmt.Errorf("write game TensorBoard scalars: %w", err)
	}
	if err := writer.Close(); err != nil {
		return fmt.Errorf("close game TensorBoard events: %w", err)
	}
	present, err = matchingGameScalarsExist(eventsDir, step, expected)
	if err != nil {
		return err
	}
	if !present {
		return errors.New("game TensorBoard scalar verification failed after write")
	}
	return nil
}

// matchingGameScalarsExist returns true only when every expected game tag is
// present exactly once at step with its exact FP32 value. Other runner-owned
// tags at the same step are intentionally ignored.
func matchingGameScalarsExist(eventsDir string, step int64, expected []tensorboard.Scalar) (bool, error) {
	inspection, err := tensorboard.InspectDir(eventsDir)
	if errors.Is(err, tensorboard.ErrNoEventFiles) {
		return false, nil
	}
	if err != nil {
		return false, fmt.Errorf("inspect TensorBoard events before game telemetry: %w", err)
	}
	want := make(map[string]float32, len(expected))
	for _, scalar := range expected {
		want[scalar.Tag] = scalar.Value
	}
	found := make(map[string][]float32, len(expected))
	for _, scalar := range inspection.Scalars {
		if scalar.Step != step {
			continue
		}
		if _, required := want[scalar.Tag]; required {
			found[scalar.Tag] = append(found[scalar.Tag], scalar.Value)
		}
	}
	if len(found) == 0 {
		return false, nil
	}
	for tag, expectedValue := range want {
		values := found[tag]
		if len(values) != 1 {
			return false, fmt.Errorf("game TensorBoard tag %q at step %d has %d records, want exactly one", tag, step, len(values))
		}
		if math.Float32bits(values[0]) != math.Float32bits(expectedValue) {
			return false, fmt.Errorf("game TensorBoard tag %q at step %d has value %g, want %g", tag, step, values[0], expectedValue)
		}
	}
	return true, nil
}

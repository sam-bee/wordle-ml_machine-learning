// Package productionreport validates the fixed production-training artifact
// set and renders its concise comparison with the retained initial proof.
//
// It is intentionally host-only: it reads artifacts that were produced by the
// independently-reloaded evaluation command and never creates a CUDA backend
// or loads a model checkpoint itself.
package productionreport

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"

	"github.com/sam-bee/wordle-ml_machine-learning/proofrun"
	"github.com/sam-bee/wordle-ml_machine-learning/runmetadata"
	"github.com/sam-bee/wordle-ml_machine-learning/runstate"
)

const (
	productionUpdates = int64(10_000)
	proofUpdates      = int64(2_000)
	validationEvery   = int64(100)
	scalarEvery       = int64(10)
	metricTolerance   = 2e-4
)

// Options identifies the completed production run, the retained full proof,
// and a separately-owned report path. None of the source run artifacts are
// modified by this package.
type Options struct {
	RunsDir         string
	ProductionRunID string
	ProofRunID      string
	OutputPath      string
}

// Report is verified evidence for the two best checkpoints and their deltas.
type Report struct {
	Production Checkpoint `json:"production"`
	Proof      Checkpoint `json:"proof"`
	Delta      Delta      `json:"delta_production_minus_proof"`
}

// Checkpoint is the concise set of independently verified metrics rendered in
// the report.
type Checkpoint struct {
	RunID                 string  `json:"run_id"`
	RunDir                string  `json:"run_dir"`
	Updates               int64   `json:"updates"`
	BestUpdate            int64   `json:"best_update"`
	Loss                  float64 `json:"validation_loss"`
	Top1                  float64 `json:"validation_top1"`
	Top5                  float64 `json:"validation_top5"`
	Top16                 float64 `json:"validation_top16"`
	Solved                int     `json:"games_solved"`
	Games                 int     `json:"games"`
	SolvedRate            float64 `json:"games_solved_fraction"`
	MeanGuesses           float64 `json:"games_mean_guesses"`
	Failures              int     `json:"games_failures"`
	GuessCounts           [6]int  `json:"games_guess_count_distribution"`
	ValidationSplitHash   string  `json:"validation_split_hash"`
	ReproductionTolerance float64 `json:"best_reproduction_tolerance"`
	MachineLearningCommit string  `json:"machine_learning_commit"`
	SyntheticDataCommit   string  `json:"synthetic_data_commit"`
	GameEngineCommit      string  `json:"game_engine_commit"`
}

// Delta expresses production-minus-proof values. Lower loss, mean guesses,
// and failures are improvements; higher top-k and solved rate are improvements.
type Delta struct {
	Loss        float64 `json:"validation_loss"`
	Top1        float64 `json:"validation_top1"`
	Top5        float64 `json:"validation_top5"`
	Top16       float64 `json:"validation_top16"`
	Solved      int     `json:"games_solved"`
	SolvedRate  float64 `json:"games_solved_fraction"`
	MeanGuesses float64 `json:"games_mean_guesses"`
	Failures    int     `json:"games_failures"`
	GuessCounts [6]int  `json:"games_guess_count_distribution"`
}

// Validate reads and verifies every production and retained-proof artifact
// required for the comparison. It does not write anything.
func Validate(options Options) (Report, error) {
	if strings.TrimSpace(options.RunsDir) == "" {
		return Report{}, errors.New("runs directory is required")
	}
	if strings.TrimSpace(options.ProductionRunID) == "" || strings.TrimSpace(options.ProofRunID) == "" {
		return Report{}, errors.New("production and proof run IDs are required")
	}
	if options.ProductionRunID == options.ProofRunID {
		return Report{}, errors.New("production and proof run IDs must differ")
	}
	production, err := loadRun(options.RunsDir, options.ProductionRunID, "production", productionUpdates)
	if err != nil {
		return Report{}, fmt.Errorf("production run %q: %w", options.ProductionRunID, err)
	}
	proof, err := loadRun(options.RunsDir, options.ProofRunID, "full", proofUpdates)
	if err != nil {
		return Report{}, fmt.Errorf("proof run %q: %w", options.ProofRunID, err)
	}
	if !sameArtifacts(production.metadata.Splits.Validation, proof.metadata.Splits.Validation) {
		return Report{}, errors.New("production and proof metadata have different validation split hashes")
	}
	if production.evaluation.ValidationSplitHash == "" || production.evaluation.ValidationSplitHash != proof.evaluation.ValidationSplitHash {
		return Report{}, errors.New("production and proof best evaluations have different validation split hashes")
	}
	productionCheckpoint := production.checkpoint()
	proofCheckpoint := proof.checkpoint()
	return Report{
		Production: productionCheckpoint,
		Proof:      proofCheckpoint,
		Delta:      difference(productionCheckpoint, proofCheckpoint),
	}, nil
}

// Write validates first, then atomically publishes a complete standalone
// production report. A validation failure leaves OutputPath unchanged.
func Write(options Options) (Report, error) {
	if strings.TrimSpace(options.OutputPath) == "" {
		return Report{}, errors.New("output path is required")
	}
	report, err := Validate(options)
	if err != nil {
		return Report{}, err
	}
	if err := writeAtomically(options.OutputPath, []byte(report.Markdown())); err != nil {
		return Report{}, err
	}
	return report, nil
}

// Markdown renders the independently validated production/proof comparison.
func (report Report) Markdown() string {
	var b strings.Builder
	b.WriteString("# Production training report\n\n")
	b.WriteString("<!-- productionreport: complete -->\n\n")
	b.WriteString("This report validates the completed production run and independently reloaded best-checkpoint evaluation artifacts. It compares them with the retained 2,000-update initial proof; it does not rerun training or inference.\n\n")
	b.WriteString("| Checkpoint | Run | Training updates / best update | Validation loss / top-1 / top-5 / top-16 | Games solved / mean guesses / failures | Guess-count distribution (1..6) |\n")
	b.WriteString("| --- | --- | ---: | --- | --- | --- |\n")
	writeCheckpointRow(&b, "production best", report.Production)
	writeCheckpointRow(&b, "initial proof best", report.Proof)
	b.WriteString("| production − proof | — | — | ")
	fmt.Fprintf(&b, "%+.4f / %+.3f / %+.3f / %+.3f | %+d / %+.3f / %+d | [%+d, %+d, %+d, %+d, %+d, %+d] |\n",
		report.Delta.Loss, report.Delta.Top1, report.Delta.Top5, report.Delta.Top16,
		report.Delta.Solved, report.Delta.MeanGuesses, report.Delta.Failures,
		report.Delta.GuessCounts[0], report.Delta.GuessCounts[1], report.Delta.GuessCounts[2], report.Delta.GuessCounts[3], report.Delta.GuessCounts[4], report.Delta.GuessCounts[5])
	b.WriteString("\n## Verified artifacts\n\n")
	fmt.Fprintf(&b, "- Production: `%s` (TensorBoard: `tensorboard --logdir %s`)\n", report.Production.RunDir, filepath.Join(report.Production.RunDir, "events"))
	fmt.Fprintf(&b, "  - best checkpoint: `%s`\n", filepath.Join(report.Production.RunDir, "checkpoints", "best"))
	fmt.Fprintf(&b, "  - independent evaluation: `%s`\n", filepath.Join(report.Production.RunDir, "evaluations", "best-games100.json"))
	fmt.Fprintf(&b, "  - commits: machine-learning `%s`; synthetic-data `%s`; game-engine `%s`\n", report.Production.MachineLearningCommit, report.Production.SyntheticDataCommit, report.Production.GameEngineCommit)
	fmt.Fprintf(&b, "  - validation split hash: `%s`; best-metric reproduction tolerance: `%g`\n", report.Production.ValidationSplitHash, report.Production.ReproductionTolerance)
	fmt.Fprintf(&b, "- Initial proof reference: `%s`\n", report.Proof.RunDir)
	fmt.Fprintf(&b, "  - best checkpoint: `%s`\n", filepath.Join(report.Proof.RunDir, "checkpoints", "best"))
	fmt.Fprintf(&b, "  - independent evaluation: `%s`\n", filepath.Join(report.Proof.RunDir, "evaluations", "best-games100.json"))
	fmt.Fprintf(&b, "  - commits: machine-learning `%s`; synthetic-data `%s`; game-engine `%s`\n", report.Proof.MachineLearningCommit, report.Proof.SyntheticDataCommit, report.Proof.GameEngineCommit)
	fmt.Fprintf(&b, "  - validation split hash: `%s`; best-metric reproduction tolerance: `%g`\n", report.Proof.ValidationSplitHash, report.Proof.ReproductionTolerance)
	b.WriteString("\nThe sealed final-test split is not opened, evaluated, or represented by this report.\n")
	return b.String()
}

func writeCheckpointRow(b *strings.Builder, name string, checkpoint Checkpoint) {
	fmt.Fprintf(b, "| %s | `%s` | %d / %d | %.4f / %.3f / %.3f / %.3f | %d/%d (%.3f) / %.3f / %d | [%d, %d, %d, %d, %d, %d] |\n",
		name, checkpoint.RunID, checkpoint.Updates, checkpoint.BestUpdate,
		checkpoint.Loss, checkpoint.Top1, checkpoint.Top5, checkpoint.Top16,
		checkpoint.Solved, checkpoint.Games, checkpoint.SolvedRate, checkpoint.MeanGuesses, checkpoint.Failures,
		checkpoint.GuessCounts[0], checkpoint.GuessCounts[1], checkpoint.GuessCounts[2], checkpoint.GuessCounts[3], checkpoint.GuessCounts[4], checkpoint.GuessCounts[5])
}

type fixedConfig struct {
	Stage            string  `json:"stage"`
	BatchSize        int     `json:"batch_size"`
	LearningRate     float64 `json:"learning_rate"`
	TargetUpdates    int64   `json:"target_updates"`
	ValidationEvery  int64   `json:"validation_every"`
	CheckpointEvery  int64   `json:"checkpoint_every"`
	ScalarEvery      int64   `json:"scalar_every"`
	Seed             int64   `json:"seed"`
	Precision        string  `json:"precision"`
	Objective        string  `json:"objective"`
	Optimizer        string  `json:"optimizer"`
	LearningRateMode string  `json:"learning_rate_schedule"`
	WeightDecay      float64 `json:"weight_decay"`
	GradientClipNorm float64 `json:"gradient_clip_global_norm"`
}

func (config fixedConfig) validate(stage string, updates int64) error {
	want := fixedConfig{
		Stage: stage, BatchSize: 256, LearningRate: 3e-4, TargetUpdates: updates,
		ValidationEvery: validationEvery, CheckpointEvery: validationEvery, ScalarEvery: scalarEvery,
		Seed: 20260808, Precision: "float32", Objective: "masked_sparse_cross_entropy_teacher_top1",
		Optimizer: "Adam", LearningRateMode: "constant", WeightDecay: 0, GradientClipNorm: 5,
	}
	if config != want {
		return fmt.Errorf("config is not the fixed %s configuration: got %+v", stage, config)
	}
	return nil
}

type metrics struct {
	Loss  float64 `json:"loss"`
	Top1  float64 `json:"top1"`
	Top5  float64 `json:"top5"`
	Top16 float64 `json:"top16"`
}

func (metrics metrics) finite() bool {
	return finite(metrics.Loss) && finite(metrics.Top1) && finite(metrics.Top5) && finite(metrics.Top16)
}

type group struct {
	Name     string  `json:"name"`
	Examples int     `json:"examples"`
	Loss     float64 `json:"loss"`
	Top1     float64 `json:"top1_accuracy"`
	Top5     float64 `json:"top5_accuracy"`
	Top16    float64 `json:"top16_accuracy"`
}

type validation struct {
	Examples               int     `json:"examples"`
	Loss                   float64 `json:"loss"`
	Top1                   float64 `json:"top1_accuracy"`
	Top5                   float64 `json:"top5_accuracy"`
	Top16                  float64 `json:"top16_accuracy"`
	MaskedArgmaxViolations int     `json:"masked_argmax_violations"`
	ByTurn                 []group `json:"by_turn"`
	ByShortlistBucket      []group `json:"by_shortlist_bucket"`
}

func (validation validation) metrics() metrics {
	return metrics{Loss: validation.Loss, Top1: validation.Top1, Top5: validation.Top5, Top16: validation.Top16}
}

func (validation validation) valid() bool {
	return validation.Examples == 2500 && validation.MaskedArgmaxViolations == 0 && validation.metrics().finite() &&
		validGroups(validation.ByTurn) && validGroups(validation.ByShortlistBucket)
}

func validGroups(groups []group) bool {
	if len(groups) == 0 {
		return false
	}
	for _, group := range groups {
		if strings.TrimSpace(group.Name) == "" || group.Examples < 0 || !groupMetrics(group).finite() {
			return false
		}
	}
	return true
}

func groupMetrics(group group) metrics {
	return metrics{Loss: group.Loss, Top1: group.Top1, Top5: group.Top5, Top16: group.Top16}
}

type snapshot struct {
	Update  int64      `json:"update"`
	Metrics metrics    `json:"metrics"`
	Details validation `json:"details"`
}

type finalMetrics struct {
	Stage                    string                     `json:"stage"`
	GlobalUpdate             int64                      `json:"global_update"`
	Passed                   bool                       `json:"passed"`
	InitialValidation        metrics                    `json:"initial_validation"`
	FinalValidation          metrics                    `json:"final_validation"`
	BestValidation           metrics                    `json:"best_validation"`
	BestValidationStep       int64                      `json:"best_validation_step"`
	InitialTraining          metrics                    `json:"initial_training"`
	FinalTraining            metrics                    `json:"final_training"`
	InitialValidationDetails snapshot                   `json:"initial_validation_details"`
	FinalValidationDetails   snapshot                   `json:"final_validation_details"`
	BestValidationDetails    snapshot                   `json:"best_validation_details"`
	ValidationSnapshots      []snapshot                 `json:"validation_snapshots"`
	ProductionSafety         *productionSafety          `json:"production_safety,omitempty"`
	Evaluations              map[string]json.RawMessage `json:"evaluations"`
}

type productionSafety struct {
	LossFinite       bool  `json:"loss_finite"`
	GradientsFinite  bool  `json:"gradients_finite"`
	ParametersFinite bool  `json:"parameters_finite"`
	UpdatesChecked   int64 `json:"updates_checked"`
}

type reproduction struct {
	Stored      metrics `json:"stored"`
	Measured    metrics `json:"measured"`
	Tolerance   float64 `json:"tolerance"`
	GroupsMatch bool    `json:"major_groups_match"`
}

type evaluation struct {
	RunID               string          `json:"run_id"`
	Stage               string          `json:"stage"`
	Mode                string          `json:"mode"`
	Checkpoint          string          `json:"checkpoint"`
	CheckpointUpdate    int64           `json:"checkpoint_update"`
	ValidationSplitHash string          `json:"validation_split_hash"`
	Validation          validation      `json:"validation"`
	BestLoss            *reproduction   `json:"best_loss_reproduction"`
	Games               *gameEvaluation `json:"games"`
}

type gameEvaluation struct {
	Summary gameSummary `json:"summary"`
	Games   []game      `json:"games"`
}

type gameSummary struct {
	Games                      int      `json:"games"`
	Solved                     int      `json:"solved"`
	SolvedFraction             float64  `json:"solved_fraction"`
	MeanGuesses                float64  `json:"mean_guesses"`
	GuessCountDistribution     [6]int   `json:"guess_count_distribution"`
	Failures                   int      `json:"failures"`
	FailedSolutions            []string `json:"failed_solutions"`
	InvalidSelections          int      `json:"invalid_selections"`
	SuppressedRawTopSelections int      `json:"suppressed_raw_top_selections"`
	RepeatedSelections         int      `json:"repeated_selections"`
}

type game struct {
	Solution                   string `json:"solution"`
	Solved                     bool   `json:"solved"`
	Guesses                    int    `json:"guesses"`
	Failure                    string `json:"failure,omitempty"`
	InvalidSelections          int    `json:"invalid_selections"`
	SuppressedRawTopSelections int    `json:"suppressed_raw_top_selections"`
	RepeatedSelections         int    `json:"repeated_selections"`
	Turns                      []turn `json:"turns"`
}

type turn struct {
	Turn  int    `json:"turn"`
	Guess string `json:"guess"`
}

type loadedRun struct {
	layout     runstate.Layout
	config     fixedConfig
	metadata   runmetadata.Manifest
	final      finalMetrics
	evaluation evaluation
}

func loadRun(runsDir, runID, stage string, updates int64) (loadedRun, error) {
	layout, err := runstate.New(runsDir, runID)
	if err != nil {
		return loadedRun{}, err
	}
	for _, path := range []string{layout.ConfigPath, layout.MetadataPath, layout.FinalMetricsPath, layout.StatePath, layout.TrainingLogPath, layout.ValidationGamesPath} {
		if err := requireRegular(path); err != nil {
			return loadedRun{}, err
		}
	}
	for _, path := range []string{layout.EventsDir, layout.InitialCheckpointDir, layout.LatestCheckpointDir, layout.BestCheckpointDir, layout.EvaluationsDir} {
		if err := requireNonEmptyDirectory(path); err != nil {
			return loadedRun{}, err
		}
	}
	configContents, err := os.ReadFile(layout.ConfigPath)
	if err != nil {
		return loadedRun{}, err
	}
	var config fixedConfig
	if err := json.Unmarshal(configContents, &config); err != nil {
		return loadedRun{}, fmt.Errorf("decode config: %w", err)
	}
	if err := config.validate(stage, updates); err != nil {
		return loadedRun{}, err
	}
	metadataContents, err := os.ReadFile(layout.MetadataPath)
	if err != nil {
		return loadedRun{}, err
	}
	var metadata runmetadata.Manifest
	if err := json.Unmarshal(metadataContents, &metadata); err != nil {
		return loadedRun{}, fmt.Errorf("decode metadata: %w", err)
	}
	if err := metadata.Validate(); err != nil {
		return loadedRun{}, fmt.Errorf("validate immutable metadata: %w", err)
	}
	if metadata.Seed != config.Seed || metadata.Runtime.Backend != "xla:cuda" || metadata.ModelParameterCount != 1_046_596 {
		return loadedRun{}, errors.New("immutable metadata does not describe the fixed CUDA policy run")
	}
	if same, err := sameJSON(configContents, metadata.EffectiveConfig); err != nil || !same {
		if err != nil {
			return loadedRun{}, fmt.Errorf("compare config with immutable metadata: %w", err)
		}
		return loadedRun{}, errors.New("config differs from immutable metadata effective_config")
	}
	finalContents, err := os.ReadFile(layout.FinalMetricsPath)
	if err != nil {
		return loadedRun{}, err
	}
	var final finalMetrics
	if err := json.Unmarshal(finalContents, &final); err != nil {
		return loadedRun{}, fmt.Errorf("decode final metrics: %w", err)
	}
	if err := validateFinal(final, stage, updates); err != nil {
		return loadedRun{}, err
	}
	state, err := layout.LoadStateMirror()
	if err != nil {
		return loadedRun{}, fmt.Errorf("read state mirror: %w", err)
	}
	if state.GlobalUpdate != updates || state.BestValidation == nil || state.BestValidation.Update != final.BestValidationStep || !close(state.BestValidation.Value, final.BestValidation.Loss) {
		return loadedRun{}, errors.New("latest checkpoint state mirror does not match final metrics")
	}
	evaluationPath := filepath.Join(layout.EvaluationsDir, "best-games100.json")
	jsonlPath := filepath.Join(layout.EvaluationsDir, "best-games100.jsonl")
	for _, path := range []string{evaluationPath, jsonlPath} {
		if err := requireRegular(path); err != nil {
			return loadedRun{}, err
		}
	}
	evaluationContents, err := os.ReadFile(evaluationPath)
	if err != nil {
		return loadedRun{}, err
	}
	var evaluated evaluation
	if err := json.Unmarshal(evaluationContents, &evaluated); err != nil {
		return loadedRun{}, fmt.Errorf("decode best games evaluation: %w", err)
	}
	if err := validateEvaluation(evaluated, runID, stage, final); err != nil {
		return loadedRun{}, err
	}
	if _, err := proofrun.VerifyTensorBoardEvents(layout.EventsDir, proofrun.Stage(stage)); err != nil {
		return loadedRun{}, fmt.Errorf("verify TensorBoard telemetry: %w", err)
	}
	if err := proofrun.VerifyGameTensorBoardEvents(layout.EventsDir, evaluated.CheckpointUpdate); err != nil {
		return loadedRun{}, fmt.Errorf("verify TensorBoard game telemetry: %w", err)
	}
	if raw, found := final.Evaluations["best-games100"]; !found {
		return loadedRun{}, errors.New("final metrics lacks best-games100 evaluation")
	} else if same, err := sameJSON(raw, evaluationContents); err != nil || !same {
		if err != nil {
			return loadedRun{}, fmt.Errorf("compare embedded best-games100 evaluation: %w", err)
		}
		return loadedRun{}, errors.New("embedded best-games100 evaluation differs from immutable artifact")
	}
	jsonl, err := os.ReadFile(jsonlPath)
	if err != nil {
		return loadedRun{}, err
	}
	if err := verifyJSONL(jsonl, evaluated.Games.Games); err != nil {
		return loadedRun{}, fmt.Errorf("verify best-games100 trajectories: %w", err)
	}
	canonical, err := os.ReadFile(layout.ValidationGamesPath)
	if err != nil {
		return loadedRun{}, err
	}
	if !bytes.Equal(canonical, jsonl) {
		return loadedRun{}, errors.New("validation-games.jsonl differs from canonical best-games100 trajectories")
	}
	return loadedRun{layout: layout, config: config, metadata: metadata, final: final, evaluation: evaluated}, nil
}

func validateFinal(final finalMetrics, stage string, updates int64) error {
	if final.Stage != stage || final.GlobalUpdate != updates || !final.Passed {
		return fmt.Errorf("run did not complete the passed %s target (stage=%q updates=%d passed=%t)", stage, final.Stage, final.GlobalUpdate, final.Passed)
	}
	for _, check := range []struct {
		name  string
		value metrics
	}{
		{"initial training", final.InitialTraining}, {"final training", final.FinalTraining},
		{"initial validation", final.InitialValidation}, {"final validation", final.FinalValidation}, {"best validation", final.BestValidation},
	} {
		if !check.value.finite() {
			return fmt.Errorf("%s metrics are non-finite", check.name)
		}
	}
	if final.BestValidationStep < 0 || final.BestValidationStep > updates || final.BestValidationStep%validationEvery != 0 {
		return fmt.Errorf("best validation update %d is not a checkpoint boundary", final.BestValidationStep)
	}
	if stage == "production" {
		safety := final.ProductionSafety
		if safety == nil || !safety.LossFinite || !safety.GradientsFinite || !safety.ParametersFinite || safety.UpdatesChecked != updates {
			return errors.New("production run lacks complete finite loss, gradient, and parameter evidence")
		}
	}
	if len(final.ValidationSnapshots) != int(updates/validationEvery+1) {
		return fmt.Errorf("validation snapshots = %d, want %d", len(final.ValidationSnapshots), updates/validationEvery+1)
	}
	for index, snapshot := range final.ValidationSnapshots {
		wantUpdate := int64(index) * validationEvery
		if err := validateSnapshot(snapshot, wantUpdate); err != nil {
			return fmt.Errorf("validation snapshot %d: %w", index, err)
		}
	}
	if err := validateSnapshot(final.InitialValidationDetails, 0); err != nil {
		return fmt.Errorf("initial validation details: %w", err)
	}
	if err := validateSnapshot(final.FinalValidationDetails, updates); err != nil {
		return fmt.Errorf("final validation details: %w", err)
	}
	if err := validateSnapshot(final.BestValidationDetails, final.BestValidationStep); err != nil {
		return fmt.Errorf("best validation details: %w", err)
	}
	if !closeMetrics(final.InitialValidationDetails.Metrics, final.InitialValidation) || !closeMetrics(final.FinalValidationDetails.Metrics, final.FinalValidation) || !closeMetrics(final.BestValidationDetails.Metrics, final.BestValidation) {
		return errors.New("top-level validation metrics differ from their snapshots")
	}
	if !sameSnapshot(final.ValidationSnapshots[0], final.InitialValidationDetails) ||
		!sameSnapshot(final.ValidationSnapshots[len(final.ValidationSnapshots)-1], final.FinalValidationDetails) ||
		!sameSnapshot(final.ValidationSnapshots[final.BestValidationStep/validationEvery], final.BestValidationDetails) {
		return errors.New("validation snapshot history differs from initial, final, or best detail")
	}
	return nil
}

func validateSnapshot(snapshot snapshot, update int64) error {
	if snapshot.Update != update || !snapshot.Metrics.finite() || !snapshot.Details.valid() || !closeMetrics(snapshot.Metrics, snapshot.Details.metrics()) {
		return fmt.Errorf("invalid snapshot at update %d", update)
	}
	return nil
}

func sameSnapshot(left, right snapshot) bool {
	return left.Update == right.Update && closeMetrics(left.Metrics, right.Metrics) && closeMetrics(left.Details.metrics(), right.Details.metrics())
}

func validateEvaluation(evaluated evaluation, runID, stage string, final finalMetrics) error {
	if evaluated.RunID != runID || evaluated.Stage != stage || evaluated.Mode != "games100" || evaluated.Checkpoint != "best" || evaluated.CheckpointUpdate != final.BestValidationStep {
		return errors.New("best-games100 evaluation has mismatched identity")
	}
	if strings.TrimSpace(evaluated.ValidationSplitHash) == "" || !evaluated.Validation.valid() || !closeMetrics(evaluated.Validation.metrics(), final.BestValidation) {
		return errors.New("best-games100 validation does not match finite saved best validation")
	}
	reproduction := evaluated.BestLoss
	if reproduction == nil || !reproduction.Stored.finite() || !reproduction.Measured.finite() || !reproduction.GroupsMatch || math.Abs(reproduction.Tolerance-metricTolerance) > 1e-12 ||
		!closeMetrics(reproduction.Stored, final.BestValidation) || !closeMetrics(reproduction.Measured, evaluated.Validation.metrics()) {
		return errors.New("best-games100 lacks a matching independent best-loss reproduction")
	}
	if !sameGroups(final.BestValidationDetails.Details, evaluated.Validation) {
		return errors.New("best-games100 validation major groups differ from saved best validation")
	}
	if evaluated.Games == nil {
		return errors.New("best-games100 evaluation lacks games")
	}
	return validateGames(*evaluated.Games)
}

func sameGroups(left, right validation) bool {
	return sameGroupSlice(left.ByTurn, right.ByTurn) && sameGroupSlice(left.ByShortlistBucket, right.ByShortlistBucket)
}

func sameGroupSlice(left, right []group) bool {
	if len(left) != len(right) {
		return false
	}
	for index := range left {
		if left[index].Name != right[index].Name || left[index].Examples != right[index].Examples || !closeMetrics(groupMetrics(left[index]), groupMetrics(right[index])) {
			return false
		}
	}
	return true
}

func validateGames(evaluation gameEvaluation) error {
	if len(evaluation.Games) != 100 || evaluation.Summary.Games != 100 {
		return fmt.Errorf("games = %d/%d, want 100", len(evaluation.Games), evaluation.Summary.Games)
	}
	var actual gameSummary
	actual.Games = len(evaluation.Games)
	seenSolutions := make(map[string]struct{}, len(evaluation.Games))
	for index, game := range evaluation.Games {
		if strings.TrimSpace(game.Solution) == "" {
			return fmt.Errorf("game %d has no solution", index)
		}
		if _, duplicate := seenSolutions[game.Solution]; duplicate {
			return fmt.Errorf("game %q occurs more than once", game.Solution)
		}
		seenSolutions[game.Solution] = struct{}{}
		if game.Guesses < 1 || game.Guesses > 6 || len(game.Turns) != game.Guesses || game.InvalidSelections < 0 || game.SuppressedRawTopSelections < 0 || game.RepeatedSelections < 0 {
			return fmt.Errorf("game %q has invalid guesses, turns, or counters", game.Solution)
		}
		seenGuesses := make(map[string]struct{}, len(game.Turns))
		for turnIndex, turn := range game.Turns {
			if turn.Turn != turnIndex+1 || strings.TrimSpace(turn.Guess) == "" {
				return fmt.Errorf("game %q has an invalid turn %d", game.Solution, turnIndex)
			}
			if _, duplicate := seenGuesses[turn.Guess]; duplicate {
				return fmt.Errorf("game %q repeated accepted guess %q", game.Solution, turn.Guess)
			}
			seenGuesses[turn.Guess] = struct{}{}
		}
		actual.GuessCountDistribution[game.Guesses-1]++
		actual.MeanGuesses += float64(game.Guesses)
		actual.InvalidSelections += game.InvalidSelections
		actual.SuppressedRawTopSelections += game.SuppressedRawTopSelections
		actual.RepeatedSelections += game.RepeatedSelections
		if game.Solved {
			if game.Failure != "" {
				return fmt.Errorf("solved game %q has failure %q", game.Solution, game.Failure)
			}
			actual.Solved++
		} else {
			if game.Failure == "" {
				return fmt.Errorf("unsolved game %q lacks a failure", game.Solution)
			}
			actual.Failures++
			actual.FailedSolutions = append(actual.FailedSolutions, game.Solution)
		}
	}
	actual.SolvedFraction = float64(actual.Solved) / float64(actual.Games)
	actual.MeanGuesses /= float64(actual.Games)
	if actual.Solved != evaluation.Summary.Solved || actual.Failures != evaluation.Summary.Failures || !closeExact(actual.SolvedFraction, evaluation.Summary.SolvedFraction) || !closeExact(actual.MeanGuesses, evaluation.Summary.MeanGuesses) ||
		actual.GuessCountDistribution != evaluation.Summary.GuessCountDistribution || !reflect.DeepEqual(actual.FailedSolutions, evaluation.Summary.FailedSolutions) || actual.InvalidSelections != evaluation.Summary.InvalidSelections || actual.SuppressedRawTopSelections != evaluation.Summary.SuppressedRawTopSelections || actual.RepeatedSelections != evaluation.Summary.RepeatedSelections {
		return errors.New("game summary does not match the 100 trajectories")
	}
	return nil
}

func verifyJSONL(contents []byte, want []game) error {
	scanner := bufio.NewScanner(bytes.NewReader(contents))
	scanner.Buffer(make([]byte, 64*1024), 4*1024*1024)
	var got []game
	for scanner.Scan() {
		if len(bytes.TrimSpace(scanner.Bytes())) == 0 {
			return errors.New("trajectory JSONL contains an empty line")
		}
		var game game
		if err := json.Unmarshal(scanner.Bytes(), &game); err != nil {
			return fmt.Errorf("decode JSONL game %d: %w", len(got), err)
		}
		got = append(got, game)
	}
	if err := scanner.Err(); err != nil {
		return err
	}
	if !reflect.DeepEqual(got, want) {
		return fmt.Errorf("JSONL trajectories differ from evaluation games: got %d, want %d", len(got), len(want))
	}
	return nil
}

func (run loadedRun) checkpoint() Checkpoint {
	summary := run.evaluation.Games.Summary
	return Checkpoint{
		RunID: run.layout.ID, RunDir: run.layout.Dir, Updates: run.final.GlobalUpdate, BestUpdate: run.final.BestValidationStep,
		Loss: run.final.BestValidation.Loss, Top1: run.final.BestValidation.Top1, Top5: run.final.BestValidation.Top5, Top16: run.final.BestValidation.Top16,
		Solved: summary.Solved, Games: summary.Games, SolvedRate: summary.SolvedFraction, MeanGuesses: summary.MeanGuesses, Failures: summary.Failures, GuessCounts: summary.GuessCountDistribution,
		ValidationSplitHash: run.evaluation.ValidationSplitHash, ReproductionTolerance: run.evaluation.BestLoss.Tolerance,
		MachineLearningCommit: run.metadata.Repositories.MachineLearning.Commit, SyntheticDataCommit: run.metadata.Repositories.SyntheticData.Commit, GameEngineCommit: run.metadata.Repositories.GameEngine.Commit,
	}
}

func difference(production, proof Checkpoint) Delta {
	delta := Delta{
		Loss: production.Loss - proof.Loss, Top1: production.Top1 - proof.Top1, Top5: production.Top5 - proof.Top5, Top16: production.Top16 - proof.Top16,
		Solved: production.Solved - proof.Solved, SolvedRate: production.SolvedRate - proof.SolvedRate, MeanGuesses: production.MeanGuesses - proof.MeanGuesses, Failures: production.Failures - proof.Failures,
	}
	for index := range delta.GuessCounts {
		delta.GuessCounts[index] = production.GuessCounts[index] - proof.GuessCounts[index]
	}
	return delta
}

func requireRegular(path string) error {
	info, err := os.Stat(path)
	if err != nil {
		return fmt.Errorf("required artifact %q: %w", path, err)
	}
	if !info.Mode().IsRegular() || info.Size() == 0 {
		return fmt.Errorf("required artifact %q is not a non-empty regular file", path)
	}
	return nil
}

func requireNonEmptyDirectory(path string) error {
	entries, err := os.ReadDir(path)
	if err != nil {
		return fmt.Errorf("required artifact directory %q: %w", path, err)
	}
	if len(entries) == 0 {
		return fmt.Errorf("required artifact directory %q is empty", path)
	}
	return nil
}

func sameArtifacts(left, right []runmetadata.Artifact) bool {
	return reflect.DeepEqual(left, right)
}

func sameJSON(left, right []byte) (bool, error) {
	var first, second any
	if err := json.Unmarshal(left, &first); err != nil {
		return false, err
	}
	if err := json.Unmarshal(right, &second); err != nil {
		return false, err
	}
	return reflect.DeepEqual(first, second), nil
}

func closeMetrics(left, right metrics) bool {
	return close(left.Loss, right.Loss) && close(left.Top1, right.Top1) && close(left.Top5, right.Top5) && close(left.Top16, right.Top16)
}

func close(left, right float64) bool      { return math.Abs(left-right) <= metricTolerance }
func closeExact(left, right float64) bool { return math.Abs(left-right) <= 1e-12 }
func finite(value float64) bool           { return !math.IsNaN(value) && !math.IsInf(value, 0) }

func writeAtomically(path string, contents []byte) error {
	directory := filepath.Dir(path)
	if err := os.MkdirAll(directory, 0o755); err != nil {
		return fmt.Errorf("create report directory %q: %w", directory, err)
	}
	temporary, err := os.CreateTemp(directory, "."+filepath.Base(path)+".tmp-*")
	if err != nil {
		return fmt.Errorf("create temporary report: %w", err)
	}
	temporaryPath := temporary.Name()
	defer func() { _ = os.Remove(temporaryPath) }()
	if err := temporary.Chmod(0o644); err != nil {
		_ = temporary.Close()
		return err
	}
	if _, err := temporary.Write(contents); err != nil {
		_ = temporary.Close()
		return err
	}
	if err := temporary.Sync(); err != nil {
		_ = temporary.Close()
		return err
	}
	if err := temporary.Close(); err != nil {
		return err
	}
	if err := os.Rename(temporaryPath, path); err != nil {
		return fmt.Errorf("publish report atomically: %w", err)
	}
	directoryHandle, err := os.Open(directory)
	if err != nil {
		return fmt.Errorf("open report directory for sync: %w", err)
	}
	defer func() { _ = directoryHandle.Close() }()
	if err := directoryHandle.Sync(); err != nil {
		return fmt.Errorf("sync report directory: %w", err)
	}
	return nil
}

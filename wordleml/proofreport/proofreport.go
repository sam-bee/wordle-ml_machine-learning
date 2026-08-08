// Package proofreport validates and renders the small supervised-training
// proof. It is deliberately host-only: it reads completed JSON artifacts and
// never creates a backend or loads a checkpoint.
package proofreport

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"sort"
	"strings"
	"time"

	"github.com/sam-bee/wordle-ml_machine-learning/proofrun"
)

const (
	// DefaultOutput is the conventional checked-in location for a proof report.
	DefaultOutput             = "docs/ml/initial-training-proof-report.md"
	metricTolerance           = 2e-4
	majorGroupMinimumExamples = 25
	validationExamples        = 2500
)

// Options identifies the three completed run directories and the report to
// publish. All paths are host filesystem paths.
type Options struct {
	RunsDir      string
	OverfitRunID string
	MiniRunID    string
	FullRunID    string
	OutputPath   string
}

// Report is the validated, renderable evidence. Validation returns an error
// for any incomplete or failed gate, so a Report always represents completion.
type Report struct {
	RunsDir   string
	Stages    []Stage
	Warnings  []string
	Ablations []Ablation
}

// Stage is one validated run shown in the report table.
type Stage struct {
	Name       string
	RunID      string
	RunDir     string
	Updates    int64
	Train      *metrics
	Validation metrics
	Games      gameSummary
	Checkpoint string
	Commits    commits
}

// Ablation is one best-checkpoint, no-retraining diagnostic shown in the
// report. Candidate state is a hard gate; the other two are recorded only.
type Ablation struct {
	Name    string
	Normal  validationDetails
	Ablated validationDetails
	Effect  metricEffect
}

// Validate consumes the proof artifacts without writing anything. It refuses
// a report unless every proof gate required by the plan is evidenced.
func Validate(options Options) (Report, error) {
	if strings.TrimSpace(options.RunsDir) == "" {
		return Report{}, errors.New("runs directory is required")
	}
	if strings.TrimSpace(options.OverfitRunID) == "" || strings.TrimSpace(options.MiniRunID) == "" || strings.TrimSpace(options.FullRunID) == "" {
		return Report{}, errors.New("overfit, mini, and full run IDs are required")
	}
	if options.OverfitRunID == options.MiniRunID || options.OverfitRunID == options.FullRunID || options.MiniRunID == options.FullRunID {
		return Report{}, errors.New("proof run IDs must be distinct")
	}
	runsRoot, err := filepath.Abs(options.RunsDir)
	if err != nil {
		return Report{}, fmt.Errorf("make runs directory absolute: %w", err)
	}
	overfit, err := loadRun(runsRoot, options.OverfitRunID, "overfit")
	if err != nil {
		return Report{}, err
	}
	mini, err := loadRun(runsRoot, options.MiniRunID, "mini")
	if err != nil {
		return Report{}, err
	}
	full, err := loadRun(runsRoot, options.FullRunID, "full")
	if err != nil {
		return Report{}, err
	}
	if !overfit.metadata.CollectedAt.Before(mini.metadata.CollectedAt) || !mini.metadata.CollectedAt.Before(full.metadata.CollectedAt) {
		return Report{}, fmt.Errorf("proof stages are not ordered by metadata collection time: overfit=%s mini=%s full=%s", overfit.metadata.CollectedAt.Format(time.RFC3339Nano), mini.metadata.CollectedAt.Format(time.RFC3339Nano), full.metadata.CollectedAt.Format(time.RFC3339Nano))
	}
	if err := verifyOverlapAudits(overfit.result.DataOverlapAudit, mini.result.DataOverlapAudit, full.result.DataOverlapAudit, overfit.result.Warnings, mini.result.Warnings, full.result.Warnings); err != nil {
		return Report{}, err
	}
	if _, err := proofrun.VerifyTensorBoardEvents(filepath.Join(overfit.dir, "events"), proofrun.Overfit); err != nil {
		return Report{}, fmt.Errorf("overfit run %q TensorBoard proof: %w", overfit.id, err)
	}
	if _, err := proofrun.VerifyTensorBoardEvents(filepath.Join(mini.dir, "events"), proofrun.Mini); err != nil {
		return Report{}, fmt.Errorf("mini run %q TensorBoard proof: %w", mini.id, err)
	}
	if _, err := proofrun.VerifyTensorBoardEvents(filepath.Join(full.dir, "events"), proofrun.Full); err != nil {
		return Report{}, fmt.Errorf("full run %q TensorBoard proof: %w", full.id, err)
	}

	initial, err := overfit.requireEvaluation("initial-games10", "initial", "games10", 10)
	if err != nil {
		return Report{}, err
	}
	latest, err := mini.requireEvaluation("latest-games10", "latest", "games10", 10)
	if err != nil {
		return Report{}, err
	}
	bestGames, err := full.requireEvaluation("best-games100", "best", "games100", 100)
	if err != nil {
		return Report{}, err
	}
	fullInitial, err := full.requireEvaluation("initial-games100", "initial", "games100", 100)
	if err != nil {
		return Report{}, err
	}
	bestAblations, err := full.requireEvaluation("best-ablations", "best", "ablations", 0)
	if err != nil {
		return Report{}, err
	}
	if err := verifyResume(mini.result.ResumeProof); err != nil {
		return Report{}, fmt.Errorf("mini run %q: %w", mini.id, err)
	}
	if err := verifyMiniTelemetry(filepath.Join(mini.dir, "events"), mini.result.TelemetryProof); err != nil {
		return Report{}, fmt.Errorf("mini run %q: %w", mini.id, err)
	}
	if err := verifyBestReproduction(full, bestGames); err != nil {
		return Report{}, err
	}
	if err := verifyBestReproduction(full, bestAblations); err != nil {
		return Report{}, err
	}
	if err := verifyAblation(bestAblations); err != nil {
		return Report{}, fmt.Errorf("full run %q: %w", full.id, err)
	}
	if initial.ValidationSplitHash == "" || initial.ValidationSplitHash != latest.ValidationSplitHash || initial.ValidationSplitHash != fullInitial.ValidationSplitHash || initial.ValidationSplitHash != bestGames.ValidationSplitHash {
		return Report{}, errors.New("game evaluations do not use one matching validation split")
	}
	if err := verifyImprovedGames(fullInitial.Games, bestGames.Games); err != nil {
		return Report{}, fmt.Errorf("full run %q: %w", full.id, err)
	}
	if full.result.BestValidationStep <= 0 {
		return Report{}, fmt.Errorf("full run %q best checkpoint is not post-initialization", full.id)
	}
	for _, check := range []struct {
		run  loadedRun
		step int64
		name string
	}{
		{overfit, 0, "initial games10"},
		{mini, mini.result.GlobalUpdate, "latest games10"},
		{full, 0, "initial games100"},
		{full, full.result.BestValidationStep, "best games100"},
	} {
		if err := proofrun.VerifyGameTensorBoardEvents(filepath.Join(check.run.dir, "events"), check.step); err != nil {
			return Report{}, fmt.Errorf("%s run %q TensorBoard game telemetry: %w", check.name, check.run.id, err)
		}
	}
	if err := verifyJSONL(filepath.Join(overfit.dir, "validation-games.jsonl"), initial.Games.Games); err != nil {
		return Report{}, fmt.Errorf("overfit run %q: %w", overfit.id, err)
	}
	if err := verifyJSONL(filepath.Join(mini.dir, "validation-games.jsonl"), latest.Games.Games); err != nil {
		return Report{}, fmt.Errorf("mini run %q: %w", mini.id, err)
	}
	if err := verifyJSONL(filepath.Join(full.dir, "validation-games.jsonl"), bestGames.Games.Games); err != nil {
		return Report{}, fmt.Errorf("full run %q: %w", full.id, err)
	}

	warnings := append([]string{}, overfit.result.Warnings...)
	warnings = append(warnings, mini.result.Warnings...)
	warnings = append(warnings, full.result.Warnings...)
	warnings = uniqueSorted(warnings)
	return Report{RunsDir: runsRoot, Stages: []Stage{
		baselineStage(overfit, initial),
		overfitStage(overfit),
		stageFor(mini, latest, "latest"),
		stageFor(full, bestGames, "best"),
	}, Warnings: warnings, Ablations: ablationRows(*bestAblations.Ablations)}, nil
}

// Write validates first, then atomically replaces OutputPath with a complete
// report. On validation failure it publishes an explicitly incomplete report
// only when doing so cannot replace an existing successful report.
func Write(options Options) (Report, error) {
	if strings.TrimSpace(options.OutputPath) == "" {
		return Report{}, errors.New("output path is required")
	}
	report, err := Validate(options)
	if err != nil {
		if writeErr := writeIncompleteReport(options, err); writeErr != nil {
			return Report{}, fmt.Errorf("%w; also write incomplete report: %v", err, writeErr)
		}
		return Report{}, err
	}
	contents := []byte(report.Markdown())
	if err := writeAtomically(options.OutputPath, contents); err != nil {
		return Report{}, err
	}
	return report, nil
}

// Markdown renders a compact human-readable proof summary.
func (report Report) Markdown() string {
	var b strings.Builder
	b.WriteString("# Initial training proof report\n\n")
	b.WriteString("<!-- proofreport: complete -->\n\n")
	b.WriteString("All required proof artifacts validated successfully. This report is generated from the immutable run artifacts; it does not rerun training or evaluation.\n\n")
	b.WriteString("| Stage | Run | Updates | Train loss / top-1 / top-5 / top-16 | Validation loss / top-1 / top-5 / top-16 | Games win rate / mean guesses | Checkpoint | Pass |\n")
	b.WriteString("| --- | --- | ---: | --- | --- | --- | --- | --- |\n")
	for _, stage := range report.Stages {
		fmt.Fprintf(&b, "| %s | `%s` | %d | %s | %s | %s | `%s` | passed |\n", stage.Name, stage.RunID, stage.Updates, formatMetrics(stage.Train), formatMetrics(&stage.Validation), formatGames(stage.Games), stage.Checkpoint)
	}
	b.WriteString("\n## Artifacts\n\n")
	for _, stage := range report.Stages {
		fmt.Fprintf(&b, "- %s: `%s` (TensorBoard: `tensorboard --logdir %s`)\n", stage.Name, stage.RunDir, filepath.Join(stage.RunDir, "events"))
		fmt.Fprintf(&b, "  - checkpoint: `%s`\n", filepath.Join(stage.RunDir, "checkpoints", stage.Checkpoint))
		fmt.Fprintf(&b, "  - commits: machine-learning `%s`; synthetic-data `%s`; game-engine `%s`\n", stage.Commits.MachineLearning, stage.Commits.SyntheticData, stage.Commits.GameEngine)
	}
	b.WriteString("\nThe full run includes independently reproduced best-checkpoint metrics, initial and best 100-game validation trajectories, and best-checkpoint ablations. The candidate-state ablation materially degrades validation performance; the best model improves that same 100-game population over the immutable initial checkpoint.\n")
	b.WriteString("\n## Best-checkpoint ablations\n\n")
	b.WriteString("| Ablation | Normal loss / top-1 / top-5 / top-16 | Ablated loss / top-1 / top-5 / top-16 | Effect (loss / top-1 / top-5 / top-16) |\n")
	b.WriteString("| --- | --- | --- | --- |\n")
	for _, ablation := range report.Ablations {
		fmt.Fprintf(&b, "| %s | %s | %s | %s |\n", ablation.Name, formatValidation(ablation.Normal), formatValidation(ablation.Ablated), formatEffect(ablation.Effect))
	}
	b.WriteString("\n## Deviations and warnings\n\n")
	if len(report.Warnings) == 0 {
		b.WriteString("None recorded.\n")
	} else {
		for _, warning := range report.Warnings {
			fmt.Fprintf(&b, "- %s\n", warning)
		}
	}
	return b.String()
}

func writeIncompleteReport(options Options, gateErr error) error {
	if existing, err := os.ReadFile(options.OutputPath); err == nil && isCompleteReport(existing) {
		return nil
	} else if err != nil && !errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("inspect existing report: %w", err)
	}
	return writeAtomically(options.OutputPath, []byte(incompleteMarkdown(options, gateErr)))
}

func isCompleteReport(contents []byte) bool {
	return bytes.Contains(contents, []byte("<!-- proofreport: complete -->")) || bytes.Contains(contents, []byte("All required proof artifacts validated successfully."))
}

func incompleteMarkdown(options Options, gateErr error) string {
	var b strings.Builder
	b.WriteString("# Initial training proof report — INCOMPLETE\n\n")
	b.WriteString("<!-- proofreport: incomplete -->\n\n")
	b.WriteString("**Validation failed. This document is not proof completion and must not be used as a successful proof report.**\n\n")
	b.WriteString("| Expected stage | Run ID | Discovered stage | Discovered update |\n")
	b.WriteString("| --- | --- | --- | ---: |\n")
	for _, stage := range discoverFailureStages(options) {
		fmt.Fprintf(&b, "| %s | `%s` | %s | %s |\n", stage.expected, stage.id, stage.stage, stage.update)
	}
	b.WriteString("\n## Blocking gate\n\n")
	fmt.Fprintf(&b, "`%s`\n", strings.ReplaceAll(gateErr.Error(), "`", "'"))
	return b.String()
}

type failureStage struct{ expected, id, stage, update string }

func discoverFailureStages(options Options) []failureStage {
	stages := []failureStage{{expected: "overfit", id: options.OverfitRunID, stage: "not discovered", update: "—"}, {expected: "mini", id: options.MiniRunID, stage: "not discovered", update: "—"}, {expected: "full", id: options.FullRunID, stage: "not discovered", update: "—"}}
	root, err := filepath.Abs(options.RunsDir)
	if err != nil {
		return stages
	}
	for index := range stages {
		path := filepath.Join(root, stages[index].id, "final-metrics.json")
		relative, err := filepath.Rel(root, path)
		if err != nil || strings.HasPrefix(relative, ".."+string(filepath.Separator)) || relative == ".." {
			continue
		}
		var summary struct {
			Stage        string `json:"stage"`
			GlobalUpdate int64  `json:"global_update"`
		}
		if contents, err := os.ReadFile(path); err == nil && json.Unmarshal(contents, &summary) == nil {
			if summary.Stage != "" {
				stages[index].stage = summary.Stage
			}
			stages[index].update = fmt.Sprintf("%d", summary.GlobalUpdate)
		}
	}
	return stages
}

type loadedRun struct {
	id          string
	dir         string
	result      runResult
	metadata    metadata
	evaluations map[string]json.RawMessage
}

type config struct {
	Stage         string `json:"stage"`
	TargetUpdates int64  `json:"target_updates"`
}
type metrics struct {
	Loss  float64 `json:"loss"`
	Top1  float64 `json:"top1"`
	Top5  float64 `json:"top5"`
	Top16 float64 `json:"top16"`
}
type resumeProof struct {
	CheckpointNextRecordIDs             []int `json:"checkpoint_next_record_ids"`
	UninterruptedReferenceNextRecordIDs []int `json:"uninterrupted_reference_next_record_ids"`
	ResumeFromUpdate                    int64 `json:"resume_from_update"`
	FirstResumedScalarUpdate            int64 `json:"first_resumed_scalar_update"`
	Completed                           bool  `json:"completed"`
}
type telemetryProof struct {
	TrainingSteps       []int64            `json:"training_steps"`
	ValidationSteps     []int64            `json:"validation_steps"`
	HistogramStepsByTag map[string][]int64 `json:"histogram_steps_by_tag"`
}
type dataOverlapAudit struct {
	TrainingRecords         int `json:"training_records"`
	TrainingUniqueStates    int `json:"training_unique_states"`
	ValidationRecords       int `json:"validation_records"`
	ValidationUniqueStates  int `json:"validation_unique_states"`
	OverlappingUniqueStates int `json:"overlapping_unique_states"`
}
type runResult struct {
	Stage                    string                     `json:"stage"`
	GlobalUpdate             int64                      `json:"global_update"`
	Passed                   bool                       `json:"passed"`
	Warnings                 []string                   `json:"warnings"`
	InitialValidation        metrics                    `json:"initial_validation"`
	FinalValidation          metrics                    `json:"final_validation"`
	BestValidation           metrics                    `json:"best_validation"`
	InitialTraining          metrics                    `json:"initial_training"`
	FinalTraining            metrics                    `json:"final_training"`
	ValidationImprovements   int                        `json:"validation_improvements"`
	MajorGroupLearning       majorGroupLearning         `json:"major_group_learning"`
	InitialValidationDetails validationSnapshot         `json:"initial_validation_details"`
	BestValidationDetails    validationSnapshot         `json:"best_validation_details"`
	OverfitProof             *overfitProof              `json:"overfit_proof"`
	BestValidationStep       int64                      `json:"best_validation_step"`
	ResumeProof              *resumeProof               `json:"resume_proof"`
	TelemetryProof           *telemetryProof            `json:"telemetry_proof"`
	DataOverlapAudit         dataOverlapAudit           `json:"data_overlap_audit"`
	Evaluations              map[string]json.RawMessage `json:"evaluations"`
}

type majorGroupLearning struct {
	MinimumExamples int      `json:"minimum_examples"`
	TurnCount       int      `json:"turn_count"`
	TurnGroups      []string `json:"turn_groups"`
	ShortlistCount  int      `json:"shortlist_count"`
	ShortlistGroups []string `json:"shortlist_groups"`
	Count           int      `json:"count"`
	Groups          []string `json:"groups"`
}

// validationSnapshot retains only the host-side group evidence used to
// independently replay the broad-learning predicate.
type validationSnapshot struct {
	Details validationGroups `json:"details"`
}

type validationGroups struct {
	Examples          int               `json:"examples"`
	ByTurn            []validationGroup `json:"by_turn"`
	ByShortlistBucket []validationGroup `json:"by_shortlist_bucket"`
}

type validationGroup struct {
	Name     string  `json:"name"`
	Examples int     `json:"examples"`
	Loss     float64 `json:"loss"`
}

type overfitProof struct {
	InitialFixedBatch               metrics `json:"initial_fixed_batch"`
	FinalFixedBatch                 metrics `json:"final_fixed_batch"`
	LossReducedAtLeastNinetyPercent bool    `json:"loss_reduced_at_least_90_percent"`
	Top1AtLeastNinetyFivePercent    bool    `json:"top1_at_least_95_percent"`
	DiagnosticsFinite               bool    `json:"diagnostics_finite"`
	ParametersFinite                bool    `json:"parameters_finite"`
	NonBiasWeightChanged            bool    `json:"non_bias_weight_changed"`
	CheckpointPredictionsReproduced bool    `json:"checkpoint_predictions_reproduced"`
}
type commits struct {
	MachineLearning string
	SyntheticData   string
	GameEngine      string
}
type metadata struct {
	CollectedAt  time.Time `json:"collected_at"`
	Repositories struct {
		MachineLearning struct {
			Commit string `json:"commit"`
		} `json:"machine_learning"`
		SyntheticData struct {
			Commit string `json:"commit"`
		} `json:"synthetic_data"`
		GameEngine struct {
			Commit string `json:"commit"`
		} `json:"game_engine"`
	} `json:"repositories"`
}
type evaluation struct {
	RunID               string            `json:"run_id"`
	Stage               string            `json:"stage"`
	Mode                string            `json:"mode"`
	Checkpoint          string            `json:"checkpoint"`
	CheckpointUpdate    int64             `json:"checkpoint_update"`
	ValidationSplitHash string            `json:"validation_split_hash"`
	Validation          validationDetails `json:"validation"`
	BestLoss            *lossReproduction `json:"best_loss_reproduction"`
	Games               *games            `json:"games"`
	Ablations           *ablations        `json:"ablations"`
}
type validationDetails struct {
	Loss                   float64 `json:"loss"`
	Top1                   float64 `json:"top1_accuracy"`
	Top5                   float64 `json:"top5_accuracy"`
	Top16                  float64 `json:"top16_accuracy"`
	MaskedArgmaxViolations int     `json:"masked_argmax_violations"`
}

// UnmarshalJSON accepts the normal proofeval schema and the earlier
// InitialGamesEvaluation schema. The latter serializes proofrun.Metrics and
// names the top-k fields top1/top5/top16 rather than *_accuracy. Presence
// pointers distinguish a genuine zero agreement from a missing metric.
func (details *validationDetails) UnmarshalJSON(contents []byte) error {
	type encoded struct {
		Loss                   float64  `json:"loss"`
		Top1Accuracy           *float64 `json:"top1_accuracy"`
		Top5Accuracy           *float64 `json:"top5_accuracy"`
		Top16Accuracy          *float64 `json:"top16_accuracy"`
		Top1                   *float64 `json:"top1"`
		Top5                   *float64 `json:"top5"`
		Top16                  *float64 `json:"top16"`
		MaskedArgmaxViolations int      `json:"masked_argmax_violations"`
	}
	var value encoded
	if err := json.Unmarshal(contents, &value); err != nil {
		return err
	}
	choose := func(accuracy, legacy *float64, name string) (float64, error) {
		if accuracy != nil {
			return *accuracy, nil
		}
		if legacy != nil {
			return *legacy, nil
		}
		return 0, fmt.Errorf("validation lacks %s", name)
	}
	var err error
	details.Loss = value.Loss
	if details.Top1, err = choose(value.Top1Accuracy, value.Top1, "top1"); err != nil {
		return err
	}
	if details.Top5, err = choose(value.Top5Accuracy, value.Top5, "top5"); err != nil {
		return err
	}
	if details.Top16, err = choose(value.Top16Accuracy, value.Top16, "top16"); err != nil {
		return err
	}
	details.MaskedArgmaxViolations = value.MaskedArgmaxViolations
	return nil
}

type lossReproduction struct {
	Stored      metrics `json:"stored"`
	Measured    metrics `json:"measured"`
	Tolerance   float64 `json:"tolerance"`
	GroupsMatch bool    `json:"major_groups_match"`
}
type ablations struct {
	Normal                validationDetails   `json:"normal"`
	OpeningCandidateState ablationMeasurement `json:"opening_candidate_state"`
	TurnZero              ablationMeasurement `json:"turn_zero"`
	NoCandidateBonus      ablationMeasurement `json:"no_candidate_bonus"`
}
type ablationMeasurement struct {
	Validation validationDetails `json:"validation"`
	Effect     metricEffect      `json:"effect_vs_normal"`
}
type metricEffect struct {
	Loss  float64 `json:"loss"`
	Top1  float64 `json:"top1_accuracy"`
	Top5  float64 `json:"top5_accuracy"`
	Top16 float64 `json:"top16_accuracy"`
}
type games struct {
	Summary gameSummary `json:"summary"`
	Games   []game      `json:"games"`
}
type gameSummary struct {
	Games                      int     `json:"games"`
	Solved                     int     `json:"solved"`
	SolvedFraction             float64 `json:"solved_fraction"`
	MeanGuesses                float64 `json:"mean_guesses"`
	Failures                   int     `json:"failures"`
	InvalidSelections          int     `json:"invalid_selections"`
	RepeatedSelections         int     `json:"repeated_selections"`
	SuppressedRawTopSelections int     `json:"suppressed_raw_top_selections"`
}
type game struct {
	Solution                   string `json:"solution"`
	Solved                     bool   `json:"solved"`
	Guesses                    int    `json:"guesses"`
	Failure                    string `json:"failure,omitempty"`
	InvalidSelections          int    `json:"invalid_selections"`
	RepeatedSelections         int    `json:"repeated_selections"`
	SuppressedRawTopSelections int    `json:"suppressed_raw_top_selections"`
	Turns                      []turn `json:"turns"`
}
type turn struct {
	Turn                int    `json:"turn"`
	RawTopActionID      int    `json:"raw_top_action_id"`
	RawTopGuess         string `json:"raw_top_guess"`
	Guess               string `json:"guess"`
	Feedback            string `json:"feedback"`
	ShortlistSizeBefore int    `json:"shortlist_size_before"`
	ShortlistSizeAfter  int    `json:"shortlist_size_after"`
}

func loadRun(root, id, expectedStage string) (loadedRun, error) {
	dir := filepath.Join(root, id)
	if relative, err := filepath.Rel(root, dir); err != nil || relative == ".." || strings.HasPrefix(relative, ".."+string(filepath.Separator)) {
		return loadedRun{}, fmt.Errorf("unsafe run ID %q", id)
	}
	// Load the runner-owned evidence before requiring post-training evaluation
	// artifacts. A failed stage intentionally has no validation-games.jsonl;
	// reporting that missing downstream file would hide the gate that stopped
	// the sequence.
	for _, path := range []string{"config.json", "metadata.json", "final-metrics.json", "run-state.json", "training.log"} {
		if err := requireFile(filepath.Join(dir, path)); err != nil {
			return loadedRun{}, fmt.Errorf("run %q: %w", id, err)
		}
	}
	var cfg config
	if err := decodeFile(filepath.Join(dir, "config.json"), &cfg); err != nil {
		return loadedRun{}, fmt.Errorf("run %q: decode config: %w", id, err)
	}
	if cfg.Stage != expectedStage {
		return loadedRun{}, fmt.Errorf("run %q has stage %q, want %q", id, cfg.Stage, expectedStage)
	}
	if cfg.TargetUpdates != expectedUpdates(expectedStage) {
		return loadedRun{}, fmt.Errorf("run %q config target updates %d, want %d", id, cfg.TargetUpdates, expectedUpdates(expectedStage))
	}
	var result runResult
	if err := decodeFile(filepath.Join(dir, "final-metrics.json"), &result); err != nil {
		return loadedRun{}, fmt.Errorf("run %q: decode final metrics: %w", id, err)
	}
	if result.Stage != expectedStage || result.GlobalUpdate != expectedUpdates(expectedStage) {
		return loadedRun{}, fmt.Errorf("run %q did not complete and pass %s (stage=%q updates=%d passed=%t)", id, expectedStage, result.Stage, result.GlobalUpdate, result.Passed)
	}
	if !finiteMetrics(result.InitialValidation) || !finiteMetrics(result.FinalValidation) || !finiteMetrics(result.BestValidation) || !finiteMetrics(result.InitialTraining) || !finiteMetrics(result.FinalTraining) {
		return loadedRun{}, fmt.Errorf("run %q has non-finite recorded metrics", id)
	}
	if err := verifyStageGate(result); err != nil {
		return loadedRun{}, fmt.Errorf("run %q: recorded %s gate evidence is invalid: %w", id, expectedStage, err)
	}
	if !result.Passed {
		return loadedRun{}, fmt.Errorf("run %q completed %s but did not record a passing result", id, expectedStage)
	}
	if err := requireFile(filepath.Join(dir, "validation-games.jsonl")); err != nil {
		return loadedRun{}, fmt.Errorf("run %q: %w", id, err)
	}
	for _, path := range []string{"events", "checkpoints/initial", "checkpoints/latest", "checkpoints/best", "evaluations"} {
		if err := requireNonEmptyDirectory(filepath.Join(dir, path)); err != nil {
			return loadedRun{}, fmt.Errorf("run %q: %w", id, err)
		}
	}
	rawMetadata, err := os.ReadFile(filepath.Join(dir, "metadata.json"))
	if err != nil {
		return loadedRun{}, err
	}
	var meta metadata
	if err := json.Unmarshal(rawMetadata, &meta); err != nil {
		return loadedRun{}, fmt.Errorf("run %q: decode metadata: %w", id, err)
	}
	if meta.CollectedAt.IsZero() || strings.TrimSpace(meta.Repositories.MachineLearning.Commit) == "" || strings.TrimSpace(meta.Repositories.SyntheticData.Commit) == "" || strings.TrimSpace(meta.Repositories.GameEngine.Commit) == "" {
		return loadedRun{}, fmt.Errorf("run %q metadata lacks collection time or required repository commits", id)
	}
	return loadedRun{id: id, dir: dir, result: result, metadata: meta, evaluations: result.Evaluations}, nil
}

func (run loadedRun) requireEvaluation(key, checkpoint, mode string, count int) (evaluation, error) {
	raw, found := run.evaluations[key]
	if !found {
		return evaluation{}, fmt.Errorf("run %q is missing embedded evaluation %q", run.id, key)
	}
	var value evaluation
	if err := json.Unmarshal(raw, &value); err != nil {
		return evaluation{}, fmt.Errorf("run %q: decode embedded %s: %w", run.id, key, err)
	}
	path := filepath.Join(run.dir, "evaluations", checkpoint+"-"+mode+".json")
	contents, err := os.ReadFile(path)
	if err != nil {
		return evaluation{}, fmt.Errorf("run %q: read evaluation artifact %q: %w", run.id, filepath.Base(path), err)
	}
	var onDisk evaluation
	if err := json.Unmarshal(contents, &onDisk); err != nil {
		return evaluation{}, fmt.Errorf("run %q: decode evaluation artifact %q: %w", run.id, filepath.Base(path), err)
	}
	if !sameJSON(raw, contents) {
		return evaluation{}, fmt.Errorf("run %q: embedded evaluation %q does not match its immutable artifact", run.id, key)
	}
	if value.RunID != run.id || value.Stage != run.result.Stage || value.Mode != mode || value.Checkpoint != checkpoint || value.ValidationSplitHash == "" {
		return evaluation{}, fmt.Errorf("run %q: evaluation %q has mismatched identity", run.id, key)
	}
	wantUpdate := run.result.GlobalUpdate
	if checkpoint == "initial" {
		wantUpdate = 0
	}
	if checkpoint == "best" {
		wantUpdate = run.result.BestValidationStep
	}
	if value.CheckpointUpdate != wantUpdate {
		return evaluation{}, fmt.Errorf("run %q: evaluation %q checkpoint update %d, want %d", run.id, key, value.CheckpointUpdate, wantUpdate)
	}
	if !finiteValidation(value.Validation) || value.Validation.MaskedArgmaxViolations != 0 {
		return evaluation{}, fmt.Errorf("run %q: evaluation %q has invalid validation metrics or selected masked actions", run.id, key)
	}
	if (key == "initial-games10" || key == "initial-games100") && !closeMetrics(metrics{Loss: value.Validation.Loss, Top1: value.Validation.Top1, Top5: value.Validation.Top5, Top16: value.Validation.Top16}, run.result.InitialValidation, metricTolerance) {
		return evaluation{}, fmt.Errorf("run %q: run-zero game evaluation validation metrics do not match initial validation", run.id)
	}
	if count > 0 {
		if value.Games == nil {
			return evaluation{}, fmt.Errorf("run %q: evaluation %q lacks game results", run.id, key)
		}
		if err := verifyGames(*value.Games, count); err != nil {
			return evaluation{}, fmt.Errorf("run %q: evaluation %q: %w", run.id, key, err)
		}
		if err := verifyJSONL(filepath.Join(run.dir, "evaluations", checkpoint+"-"+mode+".jsonl"), value.Games.Games); err != nil {
			return evaluation{}, fmt.Errorf("run %q: %w", run.id, err)
		}
	}
	if mode == "ablations" && value.Ablations == nil {
		return evaluation{}, fmt.Errorf("run %q: evaluation %q lacks ablations", run.id, key)
	}
	return onDisk, nil
}

func expectedUpdates(stage string) int64 {
	if stage == "overfit" {
		return 400
	}
	if stage == "mini" {
		return 1000
	}
	return 2000
}
func finiteMetrics(m metrics) bool {
	return finite(m.Loss) && finite(m.Top1) && finite(m.Top5) && finite(m.Top16)
}
func finiteValidation(m validationDetails) bool {
	return finite(m.Loss) && finite(m.Top1) && finite(m.Top5) && finite(m.Top16)
}
func finite(value float64) bool { return !math.IsNaN(value) && !math.IsInf(value, 0) }

// verifyStageGate repeats the runner's pure, persisted gate predicates. A
// report must not turn an unverified `passed: true` bit into proof evidence.
func verifyStageGate(result runResult) error {
	switch result.Stage {
	case "overfit":
		proof := result.OverfitProof
		if proof == nil || !finiteMetrics(proof.InitialFixedBatch) || !finiteMetrics(proof.FinalFixedBatch) {
			return errors.New("missing finite fixed-batch overfit proof")
		}
		if proof.FinalFixedBatch.Loss > proof.InitialFixedBatch.Loss*0.1 || !proof.LossReducedAtLeastNinetyPercent {
			return errors.New("fixed-batch loss did not prove at least a 90% reduction")
		}
		if proof.FinalFixedBatch.Top1 < .95 || !proof.Top1AtLeastNinetyFivePercent {
			return errors.New("fixed-batch top-1 did not prove at least 95% agreement")
		}
		if !proof.DiagnosticsFinite || !proof.ParametersFinite || !proof.NonBiasWeightChanged || !proof.CheckpointPredictionsReproduced {
			return errors.New("overfit safety, weight-change, or checkpoint-reload proof is incomplete")
		}
	case "mini":
		if result.FinalTraining.Loss >= result.InitialTraining.Loss*.5 ||
			result.FinalTraining.Top1 < result.InitialTraining.Top1+.1 ||
			result.FinalTraining.Top16 < result.InitialTraining.Top16+.1 ||
			result.TelemetryProof == nil {
			return errors.New("mini learning or telemetry gate did not pass")
		}
	case "full":
		if result.FinalTraining.Loss >= result.InitialTraining.Loss ||
			result.BestValidation.Loss > result.InitialValidation.Loss*.95 ||
			result.BestValidation.Top1 <= result.InitialValidation.Top1 ||
			result.BestValidation.Top5 <= result.InitialValidation.Top5 ||
			result.BestValidation.Top16 <= result.InitialValidation.Top16 ||
			result.ValidationImprovements < 2 {
			return errors.New("full optimization, validation, or broad-group gate did not pass")
		}
		if err := verifyMajorGroupLearning(result); err != nil {
			return fmt.Errorf("full broad-group learning evidence is invalid: %w", err)
		}
	default:
		return fmt.Errorf("unknown stage %q", result.Stage)
	}
	return nil
}

// verifyMajorGroupLearning recomputes the runner's group predicate from the
// persisted initial and best validation details. Counts alone cannot prove
// broad learning because they can be forged independently of the groups.
func verifyMajorGroupLearning(result runResult) error {
	stored := result.MajorGroupLearning
	if stored.MinimumExamples != majorGroupMinimumExamples {
		return fmt.Errorf("minimum examples %d, want %d", stored.MinimumExamples, majorGroupMinimumExamples)
	}
	initialDetails, bestDetails := result.InitialValidationDetails.Details, result.BestValidationDetails.Details
	if initialDetails.Examples != validationExamples || bestDetails.Examples != validationExamples {
		return fmt.Errorf("initial/best validation examples are %d/%d, want %d", initialDetails.Examples, bestDetails.Examples, validationExamples)
	}
	turnNames := []string{"turn_1", "turn_2", "turn_3", "turn_4", "turn_5", "turn_6"}
	turns, err := improvedGroupNames(initialDetails.ByTurn, bestDetails.ByTurn, turnNames, validationExamples, stored.MinimumExamples)
	if err != nil {
		return fmt.Errorf("turn groups: %w", err)
	}
	shortlistNames := []string{"1", "2-5", "6-20", "21-100", ">100"}
	shortlists, err := improvedGroupNames(initialDetails.ByShortlistBucket, bestDetails.ByShortlistBucket, shortlistNames, validationExamples, stored.MinimumExamples)
	if err != nil {
		return fmt.Errorf("shortlist groups: %w", err)
	}
	groups := make([]string, 0, len(turns)+len(shortlists))
	for _, group := range turns {
		groups = append(groups, "turn/"+group)
	}
	for _, group := range shortlists {
		groups = append(groups, "shortlist/"+group)
	}
	if stored.TurnCount != len(turns) || !slices.Equal(stored.TurnGroups, turns) ||
		stored.ShortlistCount != len(shortlists) || !slices.Equal(stored.ShortlistGroups, shortlists) ||
		stored.Count != len(groups) || !slices.Equal(stored.Groups, groups) {
		return fmt.Errorf("stored counts/groups do not match recomputed turn=%v shortlist=%v", turns, shortlists)
	}
	if len(turns) < 2 || len(shortlists) < 2 {
		return fmt.Errorf("need at least two improved turn and shortlist groups, got %d and %d", len(turns), len(shortlists))
	}
	return nil
}

func improvedGroupNames(initial, best []validationGroup, names []string, expectedExamples, minimum int) ([]string, error) {
	if len(initial) != len(names) || len(best) != len(names) {
		return nil, fmt.Errorf("initial/best group counts are %d/%d, want %d", len(initial), len(best), len(names))
	}
	groups := make([]string, 0, len(initial))
	initialExamples, bestExamples := 0, 0
	for index, before := range initial {
		after := best[index]
		if before.Name != names[index] || after.Name != names[index] || before.Examples < 0 || after.Examples < 0 || !finite(before.Loss) || !finite(after.Loss) {
			return nil, fmt.Errorf("invalid group at index %d", index)
		}
		initialExamples += before.Examples
		bestExamples += after.Examples
		if before.Examples >= minimum && after.Examples >= minimum && after.Loss < before.Loss {
			groups = append(groups, before.Name)
		}
	}
	if initialExamples != expectedExamples || bestExamples != expectedExamples {
		return nil, fmt.Errorf("initial/best group examples are %d/%d, want %d", initialExamples, bestExamples, expectedExamples)
	}
	return groups, nil
}

func verifyResume(proof *resumeProof) error {
	if proof == nil || !proof.Completed || proof.ResumeFromUpdate != 500 || proof.FirstResumedScalarUpdate != 510 {
		return errors.New("missing completed 500→1000 resume proof")
	}
	if len(proof.CheckpointNextRecordIDs) == 0 || !sameInts(proof.CheckpointNextRecordIDs, proof.UninterruptedReferenceNextRecordIDs) {
		return errors.New("resume sampler records do not match the uninterrupted reference")
	}
	return nil
}

func verifyMiniTelemetry(eventsDir string, proof *telemetryProof) error {
	if proof == nil {
		return errors.New("missing continuous TensorBoard telemetry proof")
	}
	derived, err := proofrun.VerifyMiniTensorBoardEvents(eventsDir)
	if err != nil {
		return fmt.Errorf("verify event files: %w", err)
	}
	if !sameInt64s(proof.TrainingSteps, derived.TrainingSteps) || !sameInt64s(proof.ValidationSteps, derived.ValidationSteps) || !sameStepMaps(proof.HistogramStepsByTag, derived.HistogramStepsByTag) {
		return errors.New("embedded TensorBoard proof does not match event files")
	}
	return nil
}

func expectedSteps(first, cadence, last int64) []int64 {
	steps := make([]int64, 0, (last-first)/cadence+1)
	for step := first; step <= last; step += cadence {
		steps = append(steps, step)
	}
	return steps
}

func verifyOverlapAudits(overfit, mini, full dataOverlapAudit, warningSets ...[]string) error {
	if err := validateOverlapAudit(overfit); err != nil {
		return fmt.Errorf("overfit run: %w", err)
	}
	if err := validateOverlapAudit(mini); err != nil {
		return fmt.Errorf("mini run: %w", err)
	}
	if err := validateOverlapAudit(full); err != nil {
		return fmt.Errorf("full run: %w", err)
	}
	if overfit != mini || overfit != full {
		return errors.New("train/validation state-overlap evidence differs between proof stages")
	}
	if overfit.OverlappingUniqueStates == 0 {
		return nil
	}
	for index, warnings := range warningSets {
		found := false
		for _, warning := range warnings {
			text := strings.ToLower(warning)
			if strings.Contains(text, "state-distribution overlap") && strings.Contains(text, "teacher top-1 labels agree") {
				found = true
				break
			}
		}
		if !found {
			return fmt.Errorf("proof stage %d has state overlap but lacks its agreeing-label warning", index+1)
		}
	}
	return nil
}

func validateOverlapAudit(audit dataOverlapAudit) error {
	if audit.TrainingRecords <= 0 || audit.ValidationRecords <= 0 || audit.TrainingUniqueStates <= 0 || audit.ValidationUniqueStates <= 0 {
		return errors.New("missing train/validation state-overlap evidence")
	}
	if audit.TrainingUniqueStates > audit.TrainingRecords || audit.ValidationUniqueStates > audit.ValidationRecords || audit.OverlappingUniqueStates < 0 || audit.OverlappingUniqueStates > audit.TrainingUniqueStates || audit.OverlappingUniqueStates > audit.ValidationUniqueStates {
		return errors.New("invalid train/validation state-overlap evidence")
	}
	return nil
}

func verifyBestReproduction(run loadedRun, evaluation evaluation) error {
	proof := evaluation.BestLoss
	if proof == nil || !proof.GroupsMatch || !finite(proof.Tolerance) || proof.Tolerance <= 0 {
		return fmt.Errorf("full run %q lacks a matching best-checkpoint reproduction", run.id)
	}
	if !closeMetrics(proof.Stored, run.result.BestValidation, proof.Tolerance) || !closeMetrics(proof.Measured, run.result.BestValidation, proof.Tolerance) {
		return fmt.Errorf("full run %q best-checkpoint reproduction does not match saved best validation", run.id)
	}
	if !close(proof.Measured.Loss, evaluation.Validation.Loss, proof.Tolerance) || !close(proof.Measured.Top1, evaluation.Validation.Top1, proof.Tolerance) || !close(proof.Measured.Top5, evaluation.Validation.Top5, proof.Tolerance) || !close(proof.Measured.Top16, evaluation.Validation.Top16, proof.Tolerance) {
		return fmt.Errorf("full run %q best-checkpoint reproduction does not match independent evaluation", run.id)
	}
	return nil
}

func verifyAblation(evaluation evaluation) error {
	a := evaluation.Ablations
	if a == nil || !finiteValidation(a.Normal) || !finiteValidation(a.OpeningCandidateState.Validation) || !finiteValidation(a.TurnZero.Validation) || !finiteValidation(a.NoCandidateBonus.Validation) {
		return errors.New("candidate-state ablation is incomplete")
	}
	for _, measurement := range []ablationMeasurement{a.OpeningCandidateState, a.TurnZero, a.NoCandidateBonus} {
		effect := measurement.Effect
		if !close(effect.Loss, measurement.Validation.Loss-a.Normal.Loss, metricTolerance) || !close(effect.Top1, measurement.Validation.Top1-a.Normal.Top1, metricTolerance) || !close(effect.Top5, measurement.Validation.Top5-a.Normal.Top5, metricTolerance) || !close(effect.Top16, measurement.Validation.Top16-a.Normal.Top16, metricTolerance) {
			return errors.New("ablation effect does not match recorded metrics")
		}
	}
	effect := a.OpeningCandidateState.Effect
	if effect.Loss < a.Normal.Loss*0.01 && effect.Top1 > -0.01 && effect.Top5 > -0.01 && effect.Top16 > -0.01 {
		return errors.New("candidate-state ablation did not materially worsen loss or a top-k metric")
	}
	return nil
}

func verifyImprovedGames(initial, full *games) error {
	if initial == nil || full == nil || len(initial.Games) != 100 || len(full.Games) != 100 {
		return errors.New("missing comparable 100-game initial or best trajectories")
	}
	for i := range initial.Games {
		if initial.Games[i].Solution != full.Games[i].Solution {
			return fmt.Errorf("validation game %d differs between initial and best evaluation", i+1)
		}
	}
	base, best := summarize(initial.Games), summarize(full.Games)
	if best.SolvedFraction <= base.SolvedFraction && best.MeanGuesses >= base.MeanGuesses {
		return fmt.Errorf("best checkpoint did not improve the same 100 validation games (initial %.3f/%.3f, best %.3f/%.3f)", base.SolvedFraction, base.MeanGuesses, best.SolvedFraction, best.MeanGuesses)
	}
	return nil
}

func verifyGames(value games, expected int) error {
	if len(value.Games) != expected || value.Summary.Games != expected {
		return fmt.Errorf("has %d games (summary %d), want %d", len(value.Games), value.Summary.Games, expected)
	}
	for _, game := range value.Games {
		if game.Solution == "" || game.Guesses < 1 || game.Guesses > 6 || len(game.Turns) != game.Guesses {
			return fmt.Errorf("game %q has invalid accepted trajectory", game.Solution)
		}
		seen := map[string]bool{}
		for _, turn := range game.Turns {
			if turn.Guess == "" || seen[turn.Guess] {
				return fmt.Errorf("game %q has a repeated or empty accepted guess", game.Solution)
			}
			seen[turn.Guess] = true
		}
	}
	want := summarize(value.Games)
	if value.Summary.Solved != want.Solved || value.Summary.Failures != want.Failures || value.Summary.InvalidSelections != want.InvalidSelections || value.Summary.RepeatedSelections != want.RepeatedSelections || value.Summary.SuppressedRawTopSelections != want.SuppressedRawTopSelections || !close(value.Summary.SolvedFraction, want.SolvedFraction, metricTolerance) || !close(value.Summary.MeanGuesses, want.MeanGuesses, metricTolerance) {
		return errors.New("game summary does not match its trajectories")
	}
	return nil
}

func summarize(values []game) gameSummary {
	result := gameSummary{Games: len(values)}
	var total int
	for _, game := range values {
		total += game.Guesses
		if game.Solved {
			result.Solved++
		} else {
			result.Failures++
		}
		result.InvalidSelections += game.InvalidSelections
		result.RepeatedSelections += game.RepeatedSelections
		result.SuppressedRawTopSelections += game.SuppressedRawTopSelections
	}
	if len(values) > 0 {
		result.SolvedFraction = float64(result.Solved) / float64(len(values))
		result.MeanGuesses = float64(total) / float64(len(values))
	}
	return result
}
func closeMetrics(a, b metrics, tolerance float64) bool {
	return close(a.Loss, b.Loss, tolerance) && close(a.Top1, b.Top1, tolerance) && close(a.Top5, b.Top5, tolerance) && close(a.Top16, b.Top16, tolerance)
}
func close(a, b, tolerance float64) bool { return math.Abs(a-b) <= tolerance }
func sameInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
func sameInt64s(a, b []int64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
func sameStepMaps(a, b map[string][]int64) bool {
	if len(a) != len(b) {
		return false
	}
	for tag, steps := range a {
		if !sameInt64s(steps, b[tag]) {
			return false
		}
	}
	return true
}

func verifyJSONL(path string, want []game) error {
	contents, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("read game trajectory artifact %q: %w", filepath.Base(path), err)
	}
	lines := bytes.Split(bytes.TrimSpace(contents), []byte{'\n'})
	if len(lines) != len(want) {
		return fmt.Errorf("game trajectory artifact %q has %d records, want %d", filepath.Base(path), len(lines), len(want))
	}
	for i, line := range lines {
		var got game
		if err := json.Unmarshal(line, &got); err != nil || !sameJSON(line, mustJSON(want[i])) {
			return fmt.Errorf("game trajectory artifact %q differs at record %d", filepath.Base(path), i+1)
		}
	}
	return nil
}
func mustJSON(value any) []byte { contents, _ := json.Marshal(value); return contents }
func sameJSON(a, b []byte) bool {
	var left, right any
	return json.Unmarshal(a, &left) == nil && json.Unmarshal(b, &right) == nil && reflect.DeepEqual(left, right)
}

func stageFor(run loadedRun, evaluation evaluation, checkpoint string) Stage {
	games := gameSummary{}
	if evaluation.Games != nil {
		games = evaluation.Games.Summary
	}
	train := run.result.FinalTraining
	return Stage{Name: run.result.Stage, RunID: run.id, RunDir: run.dir, Updates: run.result.GlobalUpdate, Train: &train, Validation: metrics{Loss: evaluation.Validation.Loss, Top1: evaluation.Validation.Top1, Top5: evaluation.Validation.Top5, Top16: evaluation.Validation.Top16}, Games: games, Checkpoint: checkpoint, Commits: commits{MachineLearning: run.metadata.Repositories.MachineLearning.Commit, SyntheticData: run.metadata.Repositories.SyntheticData.Commit, GameEngine: run.metadata.Repositories.GameEngine.Commit}}
}
func baselineStage(run loadedRun, evaluation evaluation) Stage {
	games := gameSummary{}
	if evaluation.Games != nil {
		games = evaluation.Games.Summary
	}
	return Stage{Name: "untrained baseline", RunID: run.id, RunDir: run.dir, Updates: 0, Validation: run.result.InitialValidation, Games: games, Checkpoint: "initial", Commits: commits{MachineLearning: run.metadata.Repositories.MachineLearning.Commit, SyntheticData: run.metadata.Repositories.SyntheticData.Commit, GameEngine: run.metadata.Repositories.GameEngine.Commit}}
}
func overfitStage(run loadedRun) Stage {
	train := run.result.FinalTraining
	return Stage{Name: "one-batch overfit", RunID: run.id, RunDir: run.dir, Updates: run.result.GlobalUpdate, Train: &train, Validation: run.result.FinalValidation, Checkpoint: "latest", Commits: commits{MachineLearning: run.metadata.Repositories.MachineLearning.Commit, SyntheticData: run.metadata.Repositories.SyntheticData.Commit, GameEngine: run.metadata.Repositories.GameEngine.Commit}}
}
func ablationRows(value ablations) []Ablation {
	return []Ablation{
		{Name: "candidate state", Normal: value.Normal, Ablated: value.OpeningCandidateState.Validation, Effect: value.OpeningCandidateState.Effect},
		{Name: "fixed turn", Normal: value.Normal, Ablated: value.TurnZero.Validation, Effect: value.TurnZero.Effect},
		{Name: "no candidate bonus", Normal: value.Normal, Ablated: value.NoCandidateBonus.Validation, Effect: value.NoCandidateBonus.Effect},
	}
}
func formatMetrics(value *metrics) string {
	if value == nil {
		return "—"
	}
	return fmt.Sprintf("%.4f / %.3f / %.3f / %.3f", value.Loss, value.Top1, value.Top5, value.Top16)
}
func formatValidation(value validationDetails) string {
	return fmt.Sprintf("%.4f / %.3f / %.3f / %.3f", value.Loss, value.Top1, value.Top5, value.Top16)
}
func formatEffect(value metricEffect) string {
	return fmt.Sprintf("%+.4f / %+.3f / %+.3f / %+.3f", value.Loss, value.Top1, value.Top5, value.Top16)
}
func formatGames(value gameSummary) string {
	if value.Games == 0 {
		return "—"
	}
	return fmt.Sprintf("%.3f / %.3f (suppressed raw I/R %d/%d)", value.SolvedFraction, value.MeanGuesses, value.InvalidSelections, value.RepeatedSelections)
}

func uniqueSorted(values []string) []string {
	set := map[string]bool{}
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value != "" {
			set[value] = true
		}
	}
	result := make([]string, 0, len(set))
	for value := range set {
		result = append(result, value)
	}
	sort.Strings(result)
	return result
}

func requireFile(path string) error {
	info, err := os.Stat(path)
	if err != nil {
		return fmt.Errorf("missing required artifact %q: %w", path, err)
	}
	if info.IsDir() || info.Size() == 0 {
		return fmt.Errorf("required artifact %q is empty or not a file", path)
	}
	return nil
}
func requireNonEmptyDirectory(path string) error {
	entries, err := os.ReadDir(path)
	if err != nil {
		return fmt.Errorf("missing required artifact directory %q: %w", path, err)
	}
	if len(entries) == 0 {
		return fmt.Errorf("required artifact directory %q is empty", path)
	}
	return nil
}
func decodeFile(path string, value any) error {
	contents, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	return json.Unmarshal(contents, value)
}
func writeAtomically(path string, contents []byte) error {
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("create report directory: %w", err)
	}
	tmp, err := os.CreateTemp(dir, "."+filepath.Base(path)+".tmp-*")
	if err != nil {
		return err
	}
	name := tmp.Name()
	defer os.Remove(name)
	if _, err = tmp.Write(contents); err != nil {
		tmp.Close()
		return err
	}
	if err = tmp.Sync(); err != nil {
		tmp.Close()
		return err
	}
	if err = tmp.Close(); err != nil {
		return err
	}
	if err = os.Rename(name, path); err != nil {
		return fmt.Errorf("publish report atomically: %w", err)
	}
	return nil
}

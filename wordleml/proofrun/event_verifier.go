package proofrun

import (
	"fmt"
	"math"
	"slices"

	"github.com/sam-bee/wordle-ml_machine-learning/tensorboard"
)

const miniResumeCheckpointUpdate = int64(500)

var trainingScalarTags = []string{
	"train/loss",
	"train/top1_accuracy",
	"train/top5_accuracy",
	"train/top16_accuracy",
	"optimizer/learning_rate",
	"optimizer/global_gradient_norm",
	"optimizer/applied_global_gradient_norm",
	"optimizer/parameter_norm",
	"optimizer/update_to_parameter_norm",
	"data/epoch",
	"data/examples_consumed",
	"data/shortlist_size_mean",
	"data/shortlist_size_min",
	"data/shortlist_size_max",
	"performance/examples_per_second",
	"performance/batch_duration",
	"performance/input_wait_duration",
}

var validationScalarTags = appendValidationGroupTags([]string{
	"validation/loss",
	"validation/top1_accuracy",
	"validation/top5_accuracy",
	"validation/top16_accuracy",
	"validation/output_entropy",
	"validation/raw_argmax_unavailable",
	"validation/masked_argmax_violations",
	"model/output_entropy",
	"model/beta_mean",
	"model/beta_min",
	"model/beta_max",
	"performance/validation_duration",
	"opening/highest_guess",
})

var openingTrainingAndValidationTags = []string{
	"opening/loss",
	"opening/teacher_rank",
	"opening/current_guess_id",
}

var miniValidationHistogramTags = []string{
	"model/legal_logits",
	"model/beta",
	"model/parameters",
	"optimizer/per_layer_gradient_norms",
}

var gameScalarTags = []string{
	"games/solved_fraction",
	"games/mean_guesses",
	"games/failures",
	"games/guess_count_1",
	"games/guess_count_2",
	"games/guess_count_3",
	"games/guess_count_4",
	"games/guess_count_5",
	"games/guess_count_6",
}

// TensorBoardProof is the host-verifiable TensorBoard evidence for one
// complete fixed-stage run. The step lists come from the event files, rather
// than runner state or final-metrics.json.
type TensorBoardProof struct {
	Stage               Stage              `json:"stage"`
	TrainingSteps       []int64            `json:"training_steps"`
	TrainingLossTrend   *TrainingLossTrend `json:"training_loss_trend,omitempty"`
	ValidationSteps     []int64            `json:"validation_steps"`
	HistogramStepsByTag map[string][]int64 `json:"histogram_steps_by_tag"`
}

// TrainingLossTrend is host-calculated evidence that full-stage training loss
// fell overall without claiming every noisy per-batch point was monotonic.
type TrainingLossTrend struct {
	FirstFiveMean     float64 `json:"first_five_mean"`
	FinalFiveMean     float64 `json:"final_five_mean"`
	LeastSquaresSlope float64 `json:"least_squares_slope"`
}

// MiniTelemetryProof is the mini-stage subset retained for the explicit
// stop-and-resume proof artifact.
type MiniTelemetryProof struct {
	TrainingSteps       []int64            `json:"training_steps"`
	ValidationSteps     []int64            `json:"validation_steps"`
	HistogramStepsByTag map[string][]int64 `json:"histogram_steps_by_tag"`
}

// VerifyTensorBoardEvents proves that eventsDir contains all required training
// and validation telemetry for one complete fixed proof stage. It enforces the
// exact 10-update training cadence, the exact initial-plus-100-update
// validation and histogram cadence, and every fixed plan tag written by the
// proof telemetry writers. Mini additionally proves its 500 -> 510 boundary.
func VerifyTensorBoardEvents(eventsDir string, stage Stage) (TensorBoardProof, error) {
	inspection, err := tensorboard.InspectDir(eventsDir)
	if err != nil {
		return TensorBoardProof{}, fmt.Errorf("inspect TensorBoard events: %w", err)
	}
	return verifyTensorBoardInspection(inspection, stage)
}

// VerifyMiniTensorBoardEvents preserves the focused mini resume verifier for
// callers that only need the checkpoint/resume telemetry evidence.
func VerifyMiniTensorBoardEvents(eventsDir string) (MiniTelemetryProof, error) {
	inspection, err := tensorboard.InspectDir(eventsDir)
	if err != nil {
		return MiniTelemetryProof{}, fmt.Errorf("inspect TensorBoard events: %w", err)
	}
	return verifyMiniTelemetry(inspection)
}

// VerifyGameTensorBoardEvents optionally verifies one game-evaluation summary
// after that evaluation has been written at step. It is separate because game
// evaluation is not emitted by every fixed training stage.
func VerifyGameTensorBoardEvents(eventsDir string, step int64) error {
	inspection, err := tensorboard.InspectDir(eventsDir)
	if err != nil {
		return fmt.Errorf("inspect TensorBoard events: %w", err)
	}
	for _, tag := range gameScalarTags {
		occurrences := 0
		for _, recordedStep := range stepsForScalars(inspection.Scalars, tag) {
			if recordedStep == step {
				occurrences++
			}
		}
		if occurrences != 1 {
			return fmt.Errorf("%s has %d events at game-evaluation update %d, want one", tag, occurrences, step)
		}
	}
	return nil
}

func verifyTensorBoardInspection(inspection tensorboard.Inspection, stage Stage) (TensorBoardProof, error) {
	config, err := ConfigFor(stage)
	if err != nil {
		return TensorBoardProof{}, err
	}
	trainingSteps := expectedSteps(config.ScalarEvery, config.ScalarEvery, config.TargetUpdates)
	validationSteps := expectedSteps(0, config.ValidationEvery, config.TargetUpdates)
	recordedTrainingSteps, err := requireStageScalarCadence(stage, "train/loss", inspection.Scalars, trainingSteps)
	if err != nil {
		return TensorBoardProof{}, err
	}
	recordedValidationSteps, err := requireStageScalarCadence(stage, "validation/loss", inspection.Scalars, validationSteps)
	if err != nil {
		return TensorBoardProof{}, err
	}
	proof := TensorBoardProof{
		Stage:               stage,
		TrainingSteps:       recordedTrainingSteps,
		ValidationSteps:     recordedValidationSteps,
		HistogramStepsByTag: make(map[string][]int64, len(miniValidationHistogramTags)),
	}
	if stage == Mini {
		if err := requireNoTrainingReset(proof.TrainingSteps); err != nil {
			return TensorBoardProof{}, err
		}
	}
	for _, tag := range trainingScalarTags {
		if _, err := requireStageScalarCadence(stage, tag, inspection.Scalars, trainingSteps); err != nil {
			return TensorBoardProof{}, err
		}
	}
	for _, tag := range validationScalarTags {
		if _, err := requireStageScalarCadence(stage, tag, inspection.Scalars, validationSteps); err != nil {
			return TensorBoardProof{}, err
		}
	}
	openingSteps := combinedSteps(trainingSteps, validationSteps)
	if stage == Production {
		openingSteps = uniqueSortedSteps(openingSteps)
	}
	for _, tag := range openingTrainingAndValidationTags {
		if _, err := requireStageScalarCadence(stage, tag, inspection.Scalars, openingSteps); err != nil {
			return TensorBoardProof{}, err
		}
	}
	for _, tag := range miniValidationHistogramTags {
		steps, err := requireStageHistogramCadence(stage, tag, inspection.Histograms, validationSteps)
		if err != nil {
			return TensorBoardProof{}, err
		}
		proof.HistogramStepsByTag[tag] = steps
	}
	if stage == Mini {
		if err := requireResumeBoundary(proof.TrainingSteps); err != nil {
			return TensorBoardProof{}, err
		}
	}
	if stage == Full {
		trend, err := trainingLossTrend(inspection.Scalars, proof.TrainingSteps)
		if err != nil {
			return TensorBoardProof{}, err
		}
		if trend.FinalFiveMean >= trend.FirstFiveMean {
			return TensorBoardProof{}, fmt.Errorf("full train/loss final five-point mean %g is not below first five-point mean %g", trend.FinalFiveMean, trend.FirstFiveMean)
		}
		if trend.LeastSquaresSlope >= 0 {
			return TensorBoardProof{}, fmt.Errorf("full train/loss least-squares slope %g is not negative", trend.LeastSquaresSlope)
		}
		proof.TrainingLossTrend = &trend
	}
	return proof, nil
}

func verifyMiniTelemetry(inspection tensorboard.Inspection) (MiniTelemetryProof, error) {
	proof, err := verifyTensorBoardInspection(inspection, Mini)
	if err != nil {
		return MiniTelemetryProof{}, err
	}
	return MiniTelemetryProof{
		TrainingSteps:       proof.TrainingSteps,
		ValidationSteps:     proof.ValidationSteps,
		HistogramStepsByTag: proof.HistogramStepsByTag,
	}, nil
}

func appendValidationGroupTags(tags []string) []string {
	metricNames := []string{"loss", "top1_accuracy", "top5_accuracy", "top16_accuracy", "output_entropy"}
	// The fixed validation artifact contains post-opening states only. Its
	// turn-depths 1..5 are reported by proofmetrics as turn_2..turn_6; the
	// writer intentionally omits empty groups, including turn_1.
	for turn := 2; turn <= 6; turn++ {
		for _, metric := range metricNames {
			tags = append(tags, fmt.Sprintf("validation/turn_%d/%s", turn, metric))
		}
	}
	for _, bucket := range []string{"1", "2-5", "6-20", "21-100", ">100"} {
		for _, metric := range metricNames {
			tags = append(tags, "validation/shortlist_"+bucket+"/"+metric)
		}
	}
	return tags
}

func requireResumeBoundary(trainingSteps []int64) error {
	checkpointIndex := slices.Index(trainingSteps, miniResumeCheckpointUpdate)
	if checkpointIndex < 0 {
		return fmt.Errorf("train/loss never reaches required checkpoint update %d", miniResumeCheckpointUpdate)
	}
	if checkpointIndex+1 >= len(trainingSteps) || trainingSteps[checkpointIndex+1] != miniResumeCheckpointUpdate+scalarEvery {
		return fmt.Errorf("train/loss does not resume at update %d after checkpoint update %d: steps=%v", miniResumeCheckpointUpdate+scalarEvery, miniResumeCheckpointUpdate, trainingSteps)
	}
	return nil
}

func requireNoTrainingReset(trainingSteps []int64) error {
	for _, step := range trainingSteps {
		if step == 0 {
			return fmt.Errorf("train/loss contains reset-to-zero training event")
		}
	}
	return nil
}

func requireExactCadence(tag string, got, want []int64) error {
	if !slices.Equal(got, want) {
		return fmt.Errorf("%s steps = %v, want exact cadence %v", tag, got, want)
	}
	return nil
}

// requireStageScalarCadence retains the proof stages' strict one-event-per-
// step rule. A production process can be interrupted after telemetry has been
// flushed but before the next checkpoint, so its resumed process can append
// finite replacements for an already-recorded suffix. Production evidence is
// therefore checked by unique step coverage rather than event count.
func requireStageScalarCadence(stage Stage, tag string, records []tensorboard.ScalarRecord, want []int64) ([]int64, error) {
	steps := stepsForScalars(records, tag)
	if stage == Production {
		for _, record := range records {
			if record.Tag == tag && !finiteEventScalar(record.Value) {
				return nil, fmt.Errorf("%s at update %d is non-finite: %v", tag, record.Step, record.Value)
			}
		}
		steps = uniqueSortedSteps(steps)
	}
	if err := requireExactCadence(tag, steps, want); err != nil {
		return nil, err
	}
	return steps, nil
}

// requireStageHistogramCadence applies the same production resume rule to
// validation histograms. Histogram counts are checked as finite so a resumed
// event segment cannot hide malformed numerical telemetry behind de-duplication.
func requireStageHistogramCadence(stage Stage, tag string, records []tensorboard.HistogramRecord, want []int64) ([]int64, error) {
	steps := stepsForHistograms(records, tag)
	if stage == Production {
		for _, record := range records {
			if record.Tag == tag && (math.IsNaN(record.Count) || math.IsInf(record.Count, 0)) {
				return nil, fmt.Errorf("%s histogram at update %d has non-finite count: %v", tag, record.Step, record.Count)
			}
		}
		steps = uniqueSortedSteps(steps)
	}
	if err := requireExactCadence(tag, steps, want); err != nil {
		return nil, err
	}
	return steps, nil
}

func combinedSteps(first, second []int64) []int64 {
	steps := make([]int64, 0, len(first)+len(second))
	steps = append(steps, first...)
	steps = append(steps, second...)
	slices.Sort(steps)
	return steps
}

func uniqueSortedSteps(steps []int64) []int64 {
	if len(steps) == 0 {
		return nil
	}
	sorted := slices.Clone(steps)
	slices.Sort(sorted)
	unique := make([]int64, 0, len(sorted))
	for _, step := range sorted {
		if len(unique) == 0 || unique[len(unique)-1] != step {
			unique = append(unique, step)
		}
	}
	return unique
}

func trainingLossTrend(records []tensorboard.ScalarRecord, steps []int64) (TrainingLossTrend, error) {
	if len(steps) < 5 {
		return TrainingLossTrend{}, fmt.Errorf("train/loss has %d points, need at least five for trend evidence", len(steps))
	}
	valuesByStep := make(map[int64]float32, len(records))
	for _, record := range records {
		if record.Tag != "train/loss" {
			continue
		}
		if !finiteEventScalar(record.Value) {
			return TrainingLossTrend{}, fmt.Errorf("train/loss at update %d is non-finite: %v", record.Step, record.Value)
		}
		valuesByStep[record.Step] = record.Value
	}
	values := make([]float64, len(steps))
	for index, step := range steps {
		value, found := valuesByStep[step]
		if !found {
			return TrainingLossTrend{}, fmt.Errorf("train/loss is missing update %d", step)
		}
		values[index] = float64(value)
	}
	first, final := mean(values[:5]), mean(values[len(values)-5:])
	var xMean, yMean float64
	for index, value := range values {
		xMean += float64(index)
		yMean += value
	}
	xMean /= float64(len(values))
	yMean /= float64(len(values))
	var numerator, denominator float64
	for index, value := range values {
		dx := float64(index) - xMean
		numerator += dx * (value - yMean)
		denominator += dx * dx
	}
	return TrainingLossTrend{FirstFiveMean: first, FinalFiveMean: final, LeastSquaresSlope: numerator / denominator}, nil
}

func finiteEventScalar(value float32) bool {
	return !math.IsNaN(float64(value)) && !math.IsInf(float64(value), 0)
}

func mean(values []float64) float64 {
	var total float64
	for _, value := range values {
		total += value
	}
	return total / float64(len(values))
}

func expectedSteps(first, every, last int64) []int64 {
	steps := make([]int64, 0, (last-first)/every+1)
	for step := first; step <= last; step += every {
		steps = append(steps, step)
	}
	return steps
}

func stepsForScalars(records []tensorboard.ScalarRecord, tag string) []int64 {
	steps := make([]int64, 0)
	for _, record := range records {
		if record.Tag == tag {
			steps = append(steps, record.Step)
		}
	}
	slices.Sort(steps)
	return steps
}

func stepsForHistograms(records []tensorboard.HistogramRecord, tag string) []int64 {
	steps := make([]int64, 0)
	for _, record := range records {
		if record.Tag == tag {
			steps = append(steps, record.Step)
		}
	}
	slices.Sort(steps)
	return steps
}

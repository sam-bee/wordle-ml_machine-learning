package proofrun

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"math"
	"sort"
	"time"

	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/sam-bee/wordle-ml_machine-learning/imitationdata"
	"github.com/sam-bee/wordle-ml_machine-learning/proofmetrics"
	"github.com/sam-bee/wordle-ml_machine-learning/supervised"
	"github.com/sam-bee/wordle-ml_machine-learning/tensorboard"
)

// ValidationSnapshot is the complete ordered host-side validation result kept
// in final-metrics.json. Metrics remains duplicated at Result's top level for
// backwards-compatible proof gates and concise reporting.
type ValidationSnapshot struct {
	Update          int64               `json:"update"`
	Metrics         Metrics             `json:"metrics"`
	Details         proofmetrics.Result `json:"details"`
	DurationSeconds float64             `json:"duration_seconds"`
	Beta            Distribution        `json:"beta"`
	LegalLogits     Distribution        `json:"legal_logits"`
	Parameters      Distribution        `json:"parameters"`
	OpeningWord     string              `json:"opening_word"`
}

// Distribution is a compact, JSON-friendly summary of a sampled finite tensor.
type Distribution struct {
	Count int     `json:"count"`
	Mean  float64 `json:"mean"`
	Min   float64 `json:"min"`
	Max   float64 `json:"max"`
}

// histogramSamples keeps TensorBoard validation telemetry bounded regardless
// of vocabulary size or validation-set size. The first values in fixed dataset
// order are deliberately used: this is deterministic and adequate for a
// diagnostic sample, rather than pretending to be a random population sample.
type histogramSamples struct {
	legalLogits []float64
	beta        []float64
	parameters  []float64
	legalSeen   uint64
	betaSeen    uint64
}

const histogramSampleLimit = 4096

func (h *histogramSamples) addLegalLogits(values []float32, availability []float32) {
	for index, value := range values {
		if availability[index] != 0 {
			h.legalLogits = appendSampledFinite(h.legalLogits, &h.legalSeen, float64(value), histogramSampleLimit)
		}
	}
}

func (h *histogramSamples) addBeta(values []float32) {
	for _, value := range values {
		h.beta = appendSampledFinite(h.beta, &h.betaSeen, float64(value), histogramSampleLimit)
	}
}

// appendSampledFinite is a deterministic reservoir sampler. It bounds memory
// while representing every ordered validation batch and every parameter layer,
// instead of silently filling a histogram from the first row alone.
func appendSampledFinite(values []float64, seen *uint64, value float64, limit int) []float64 {
	if math.IsNaN(value) || math.IsInf(value, 0) {
		return values
	}
	*seen++
	if len(values) < limit {
		return append(values, value)
	}
	// A fixed mixing function gives reservoir replacement without depending on
	// wall-clock randomness, which keeps proof artifacts reproducible.
	index := mix64(*seen) % *seen
	if index < uint64(limit) {
		values[index] = value
	}
	return values
}

func mix64(value uint64) uint64 {
	value ^= value >> 30
	value *= 0xbf58476d1ce4e5b9
	value ^= value >> 27
	value *= 0x94d049bb133111eb
	return value ^ (value >> 31)
}

func distribution(values []float64) Distribution {
	if len(values) == 0 {
		return Distribution{}
	}
	result := Distribution{Count: len(values), Min: values[0], Max: values[0]}
	for _, value := range values {
		result.Mean += value
		result.Min = min(result.Min, value)
		result.Max = max(result.Max, value)
	}
	result.Mean /= float64(len(values))
	return result
}

// evaluateDetailed makes one complete, ordered validation pass through the
// no-update inference executor. It checks the hard mask on every output and
// computes cross-entropy over exactly the legal raw logits on the host. A Store
// fingerprint before and after proves that this validation implementation did
// not mutate any model, optimizer, or diagnostic value.
func evaluateDetailed(session *supervised.Session, data *imitationdata.Data, opening imitationdata.Example, openingWord func(int) string) (ValidationSnapshot, histogramSamples, error) {
	if err := warmValidationShapes(session, data, opening); err != nil {
		return ValidationSnapshot{}, histogramSamples{}, err
	}
	before, err := StoreFingerprint(session.Store)
	if err != nil {
		return ValidationSnapshot{}, histogramSamples{}, fmt.Errorf("fingerprint before validation: %w", err)
	}
	started := time.Now()
	var collector proofmetrics.Collector
	var samples histogramSamples
	for start := 0; start < data.Len(); start += validationBatchSize {
		end := min(start+validationBatchSize, data.Len())
		examples := make([]imitationdata.Example, 0, end-start)
		for index := start; index < end; index++ {
			example, err := data.Example(index)
			if err != nil {
				return ValidationSnapshot{}, histogramSamples{}, err
			}
			examples = append(examples, example)
		}
		if err := evaluateDiagnosticExamples(session, examples, &collector, &samples, false); err != nil {
			return ValidationSnapshot{}, histogramSamples{}, fmt.Errorf("validation examples %d..%d: %w", start, end, err)
		}
	}
	if err := evaluateDiagnosticExamples(session, []imitationdata.Example{opening}, &collector, nil, true); err != nil {
		return ValidationSnapshot{}, histogramSamples{}, fmt.Errorf("opening diagnostic: %w", err)
	}
	after, err := StoreFingerprint(session.Store)
	if err != nil {
		return ValidationSnapshot{}, histogramSamples{}, fmt.Errorf("fingerprint after validation: %w", err)
	}
	if before != after {
		return ValidationSnapshot{}, histogramSamples{}, fmt.Errorf("validation mutated Store: before %s, after %s", before, after)
	}
	parameters, err := parameterHistogram(session.Store)
	if err != nil {
		return ValidationSnapshot{}, histogramSamples{}, fmt.Errorf("sample parameter statistics: %w", err)
	}
	samples.parameters = parameters
	details := collector.Result()
	metrics := Metrics{Loss: details.Loss, Top1: details.Top1Accuracy, Top5: details.Top5Accuracy, Top16: details.Top16Accuracy}
	if !metrics.Finite() {
		return ValidationSnapshot{}, histogramSamples{}, fmt.Errorf("host validation has non-finite metrics: %+v", metrics)
	}
	openingWordValue := ""
	if details.Opening.HighestGuess >= 0 {
		openingWordValue = openingWord(details.Opening.HighestGuess)
	}
	return ValidationSnapshot{
		Metrics: metrics, Details: details, DurationSeconds: time.Since(started).Seconds(),
		Beta: distribution(samples.beta), LegalLogits: distribution(samples.legalLogits), Parameters: distribution(samples.parameters), OpeningWord: openingWordValue,
	}, samples, nil
}

// warmValidationShapes materializes the only inference shapes used by the
// fixed proof runner. This is required before reading a lazily-restored Store
// for immutable run metadata: otherwise its parameter count can incorrectly
// describe only the variables touched before graph materialization.
func warmValidationShapes(session *supervised.Session, data *imitationdata.Data, opening imitationdata.Example) error {
	if err := warmInference(session, []imitationdata.Example{opening}); err != nil {
		return fmt.Errorf("warm opening inference: %w", err)
	}
	warmEnd := min(validationBatchSize, data.Len())
	warmExamples := make([]imitationdata.Example, 0, warmEnd)
	for index := 0; index < warmEnd; index++ {
		example, err := data.Example(index)
		if err != nil {
			return err
		}
		warmExamples = append(warmExamples, example)
	}
	if err := warmInference(session, warmExamples); err != nil {
		return fmt.Errorf("warm validation inference: %w", err)
	}
	return nil
}

func warmInference(session *supervised.Session, examples []imitationdata.Example) error {
	batch, err := imitationdata.MaterializeBatch(examples)
	if err != nil {
		return err
	}
	defer finalize(batch.Inputs)
	defer finalize(batch.Labels)
	raw, masked, beta, err := session.PredictDiagnostics(batch.Inputs[0], batch.Inputs[1], batch.Inputs[2], batch.Inputs[3], batch.Inputs[4])
	if err != nil {
		return err
	}
	defer func() { _ = raw.FinalizeAll() }()
	defer func() { _ = masked.FinalizeAll() }()
	defer func() { _ = beta.FinalizeAll() }()
	return nil
}

func evaluateOpening(session *supervised.Session, opening imitationdata.Example, openingWord func(int) string) (proofmetrics.OpeningResult, string, error) {
	var collector proofmetrics.Collector
	if err := evaluateDiagnosticExamples(session, []imitationdata.Example{opening}, &collector, nil, true); err != nil {
		return proofmetrics.OpeningResult{}, "", err
	}
	result := collector.Result().Opening
	word := ""
	if result.HighestGuess >= 0 {
		word = openingWord(result.HighestGuess)
	}
	return result, word, nil
}

func evaluateDiagnosticExamples(session *supervised.Session, examples []imitationdata.Example, collector *proofmetrics.Collector, samples *histogramSamples, opening bool) error {
	batch, err := imitationdata.MaterializeBatch(examples)
	if err != nil {
		return err
	}
	defer finalize(batch.Inputs)
	defer finalize(batch.Labels)
	rawTensor, maskedTensor, betaTensor, err := session.PredictDiagnostics(batch.Inputs[0], batch.Inputs[1], batch.Inputs[2], batch.Inputs[3], batch.Inputs[4])
	if err != nil {
		return err
	}
	defer func() { _ = rawTensor.FinalizeAll() }()
	defer func() { _ = maskedTensor.FinalizeAll() }()
	defer func() { _ = betaTensor.FinalizeAll() }()
	raw, err := tensors.CopyFlatData[float32](rawTensor)
	if err != nil {
		return fmt.Errorf("copy raw logits: %w", err)
	}
	masked, err := tensors.CopyFlatData[float32](maskedTensor)
	if err != nil {
		return fmt.Errorf("copy masked logits: %w", err)
	}
	beta, err := tensors.CopyFlatData[float32](betaTensor)
	if err != nil {
		return fmt.Errorf("copy beta: %w", err)
	}
	actions := len(examples[0].AvailableActionMask)
	if len(raw) != len(examples)*actions || len(masked) != len(raw) || len(beta) != len(examples) {
		return fmt.Errorf("inference shapes raw=%d masked=%d beta=%d for batch=%d actions=%d", len(raw), len(masked), len(beta), len(examples), actions)
	}
	if samples != nil {
		samples.addBeta(beta)
	}
	for index, example := range examples {
		rawRow := raw[index*actions : (index+1)*actions]
		maskedRow := masked[index*actions : (index+1)*actions]
		prediction, err := verifyHardMask(rawRow, maskedRow, example.AvailableActionMask)
		if err != nil {
			return fmt.Errorf("record %d: %w", example.RecordIndex, err)
		}
		loss, err := legalCrossEntropy(rawRow, example.AvailableActionMask, int(example.TeacherTopAction))
		if err != nil {
			return fmt.Errorf("record %d: %w", example.RecordIndex, err)
		}
		candidateCount := 0
		for _, value := range example.CandidateMask {
			if value != 0 {
				candidateCount++
			}
		}
		if samples != nil {
			samples.addLegalLogits(maskedRow, example.AvailableActionMask)
		}
		if opening {
			if err := collector.SetOpening(proofmetrics.Sample{Loss: loss, Logits: rawRow, MaskedPrediction: &prediction, TeacherTopActions: example.TeacherTopActions, Turn: int(example.Turn), CandidateCount: candidateCount, Availability: example.AvailableActionMask, Opening: true}); err != nil {
				return err
			}
		} else if err := collector.Add(proofmetrics.Sample{Loss: loss, Logits: rawRow, MaskedPrediction: &prediction, TeacherTopActions: example.TeacherTopActions, Turn: int(example.Turn), CandidateCount: candidateCount, Availability: example.AvailableActionMask}); err != nil {
			return err
		}
	}
	return nil
}

func verifyHardMask(raw, masked, available []float32) (int, error) {
	if len(raw) == 0 || len(raw) != len(masked) || len(raw) != len(available) {
		return 0, fmt.Errorf("raw/masked/availability lengths are %d/%d/%d", len(raw), len(masked), len(available))
	}
	prediction := -1
	best := float32(math.Inf(-1))
	for action := range raw {
		if !finite32(raw[action]) {
			return 0, fmt.Errorf("raw logit %d is non-finite: %v", action, raw[action])
		}
		if available[action] == 0 {
			if !math.IsInf(float64(masked[action]), -1) {
				return 0, fmt.Errorf("unavailable action %d is not hard-masked to -Inf: %v", action, masked[action])
			}
			continue
		}
		if !finite32(masked[action]) {
			return 0, fmt.Errorf("legal action %d has non-finite masked logit: %v", action, masked[action])
		}
		if masked[action] != raw[action] {
			return 0, fmt.Errorf("legal action %d changed from raw logit %v to masked logit %v", action, raw[action], masked[action])
		}
		if prediction < 0 || masked[action] > best {
			prediction, best = action, masked[action]
		}
	}
	if prediction < 0 {
		return 0, fmt.Errorf("example has no legal actions")
	}
	return prediction, nil
}

func legalCrossEntropy(logits, available []float32, target int) (float64, error) {
	if target < 0 || target >= len(logits) || len(logits) != len(available) {
		return 0, fmt.Errorf("teacher target %d is outside logits/availability", target)
	}
	if available[target] == 0 {
		return 0, fmt.Errorf("teacher target %d is unavailable", target)
	}
	maximum := math.Inf(-1)
	for action, value := range logits {
		if available[action] != 0 {
			if !finite32(value) {
				return 0, fmt.Errorf("legal logit %d is non-finite: %v", action, value)
			}
			maximum = max(maximum, float64(value))
		}
	}
	sum := 0.0
	for action, value := range logits {
		if available[action] != 0 {
			sum += math.Exp(float64(value) - maximum)
		}
	}
	loss := maximum + math.Log(sum) - float64(logits[target])
	if math.IsNaN(loss) || math.IsInf(loss, 0) {
		return 0, fmt.Errorf("cross-entropy is non-finite: %v", loss)
	}
	return loss, nil
}

func finite32(value float32) bool {
	return !math.IsNaN(float64(value)) && !math.IsInf(float64(value), 0)
}

// StoreFingerprint hashes every materialized Store variable in path order. It
// is intentionally a value fingerprint, not a model-architecture hash: it is
// used to fail if a supposedly inference-only validation pass mutates state.
func StoreFingerprint(store *model.Store) (string, error) {
	if store == nil {
		return "", fmt.Errorf("Store is nil")
	}
	variables := make([]*model.Variable, 0)
	for variable := range store.IterVariables() {
		variables = append(variables, variable)
	}
	sort.Slice(variables, func(left, right int) bool { return variables[left].Path() < variables[right].Path() })
	hash := sha256.New()
	for _, variable := range variables {
		if !variable.HasValue() {
			return "", fmt.Errorf("Store variable %q has no value", variable.Path())
		}
		if _, err := io.WriteString(hash, variable.Path()+"\x00"+variable.Shape().String()+"\x00"); err != nil {
			return "", err
		}
		value, err := variable.Value()
		if err != nil {
			return "", fmt.Errorf("read Store variable %q: %w", variable.Path(), err)
		}
		if err := value.ConstBytes(func(data []byte) { _, _ = hash.Write(data) }); err != nil {
			return "", fmt.Errorf("read Store variable bytes %q: %w", variable.Path(), err)
		}
	}
	return hex.EncodeToString(hash.Sum(nil)), nil
}

func parameterHistogram(store *model.Store) ([]float64, error) {
	variables := make([]*model.Variable, 0)
	for variable := range store.IterVariables() {
		if variable.Trainable {
			variables = append(variables, variable)
		}
	}
	sort.Slice(variables, func(left, right int) bool { return variables[left].Path() < variables[right].Path() })
	values := make([]float64, 0, histogramSampleLimit)
	var seen uint64
	for _, variable := range variables {
		data, err := tensors.CopyFlatData[float32](variable.MustValue())
		if err != nil {
			return nil, fmt.Errorf("copy parameter %q: %w", variable.Path(), err)
		}
		for _, value := range data {
			values = appendSampledFinite(values, &seen, float64(value), histogramSampleLimit)
		}
	}
	return values, nil
}

func validationHistograms(samples histogramSamples, session *supervised.Session) ([]tensorboard.Histogram, error) {
	parameters := samples.parameters
	if len(parameters) == 0 {
		var err error
		parameters, err = parameterHistogram(session.Store)
		if err != nil {
			return nil, err
		}
	}
	histograms := []tensorboard.Histogram{
		{Tag: "model/legal_logits", Values: samples.legalLogits},
		{Tag: "model/beta", Values: samples.beta},
		{Tag: "model/parameters", Values: parameters},
	}
	norms, err := supervised.ReadGradientNorms(session.Store)
	if err != nil {
		return nil, fmt.Errorf("read per-layer gradient norms: %w", err)
	}
	gradientNorms := make([]float64, 0, len(norms))
	for _, norm := range norms {
		gradientNorms = append(gradientNorms, float64(norm.Norm))
	}
	histograms = append(histograms, tensorboard.Histogram{Tag: "optimizer/per_layer_gradient_norms", Values: gradientNorms})
	return histograms, nil
}

func writeValidationTelemetry(events *tensorboard.Writer, step int64, snapshot ValidationSnapshot, samples histogramSamples, session *supervised.Session) error {
	scalars := snapshot.Details.TensorBoardScalars("validation")
	scalars = append(scalars,
		tensorboard.Scalar{Tag: "model/beta_mean", Value: float32(snapshot.Beta.Mean)},
		tensorboard.Scalar{Tag: "model/beta_min", Value: float32(snapshot.Beta.Min)},
		tensorboard.Scalar{Tag: "model/beta_max", Value: float32(snapshot.Beta.Max)},
		tensorboard.Scalar{Tag: "performance/validation_duration", Value: float32(snapshot.DurationSeconds)},
		tensorboard.Scalar{Tag: "opening/current_guess_id", Value: float32(snapshot.Details.Opening.HighestGuess)},
	)
	if err := events.WriteScalars(step, scalars...); err != nil {
		return err
	}
	histograms, err := validationHistograms(samples, session)
	if err != nil {
		return err
	}
	if err := events.WriteHistograms(step, histograms...); err != nil {
		return err
	}
	return events.Flush()
}

func writeTrainingTelemetry(events *tensorboard.Writer, step int64, metrics Metrics, diagnostics supervised.TrainingDiagnostics, examples []imitationdata.Example, epoch int64, examplesConsumed int64, batchDuration, inputWait time.Duration, opening proofmetrics.OpeningResult) error {
	shortlistTotal, shortlistMin, shortlistMax := 0, math.MaxInt, 0
	for _, example := range examples {
		count := 0
		for _, value := range example.CandidateMask {
			if value != 0 {
				count++
			}
		}
		shortlistTotal += count
		shortlistMin = min(shortlistMin, count)
		shortlistMax = max(shortlistMax, count)
	}
	if shortlistMin == math.MaxInt {
		shortlistMin = 0
	}
	seconds := batchDuration.Seconds()
	examplesPerSecond := 0.0
	if seconds > 0 {
		examplesPerSecond = float64(len(examples)) / seconds
	}
	return events.WriteScalars(step,
		tensorboard.Scalar{Tag: "train/loss", Value: float32(metrics.Loss)},
		tensorboard.Scalar{Tag: "train/top1_accuracy", Value: float32(metrics.Top1)},
		tensorboard.Scalar{Tag: "train/top5_accuracy", Value: float32(metrics.Top5)},
		tensorboard.Scalar{Tag: "train/top16_accuracy", Value: float32(metrics.Top16)},
		tensorboard.Scalar{Tag: "optimizer/learning_rate", Value: diagnostics.LearningRate},
		tensorboard.Scalar{Tag: "optimizer/global_gradient_norm", Value: diagnostics.PreclipGlobalGradientNorm},
		tensorboard.Scalar{Tag: "optimizer/applied_global_gradient_norm", Value: diagnostics.AppliedGlobalGradientNorm},
		tensorboard.Scalar{Tag: "optimizer/parameter_norm", Value: diagnostics.ParameterNorm},
		tensorboard.Scalar{Tag: "optimizer/update_to_parameter_norm", Value: diagnostics.UpdateToParameterNorm},
		tensorboard.Scalar{Tag: "data/epoch", Value: float32(epoch)},
		tensorboard.Scalar{Tag: "data/examples_consumed", Value: float32(examplesConsumed)},
		tensorboard.Scalar{Tag: "data/shortlist_size_mean", Value: float32(float64(shortlistTotal) / float64(len(examples)))},
		tensorboard.Scalar{Tag: "data/shortlist_size_min", Value: float32(shortlistMin)},
		tensorboard.Scalar{Tag: "data/shortlist_size_max", Value: float32(shortlistMax)},
		tensorboard.Scalar{Tag: "performance/examples_per_second", Value: float32(examplesPerSecond)},
		tensorboard.Scalar{Tag: "performance/batch_duration", Value: float32(seconds)},
		tensorboard.Scalar{Tag: "performance/input_wait_duration", Value: float32(inputWait.Seconds())},
		tensorboard.Scalar{Tag: "opening/loss", Value: float32(opening.Loss)},
		tensorboard.Scalar{Tag: "opening/teacher_rank", Value: float32(opening.TeacherRank)},
		tensorboard.Scalar{Tag: "opening/current_guess_id", Value: float32(opening.HighestGuess)},
	)
}

func validateTrainingDiagnostics(diagnostics supervised.TrainingDiagnostics) error {
	if !diagnostics.GradientsFinite || !diagnostics.ParametersFinite {
		return fmt.Errorf("non-finite optimizer diagnostics: gradients_finite=%t parameters_finite=%t", diagnostics.GradientsFinite, diagnostics.ParametersFinite)
	}
	for name, value := range map[string]float32{
		"preclip global gradient norm": diagnostics.PreclipGlobalGradientNorm,
		"applied global gradient norm": diagnostics.AppliedGlobalGradientNorm,
		"parameter norm":               diagnostics.ParameterNorm,
		"update to parameter norm":     diagnostics.UpdateToParameterNorm,
		"learning rate":                diagnostics.LearningRate,
	} {
		if !finite32(value) {
			return fmt.Errorf("%s is non-finite: %v", name, value)
		}
	}
	if diagnostics.AppliedGlobalGradientNorm > supervised.GlobalGradientClipNorm+1e-4 {
		return fmt.Errorf("applied global gradient norm %g exceeds clip limit %g", diagnostics.AppliedGlobalGradientNorm, supervised.GlobalGradientClipNorm)
	}
	return nil
}

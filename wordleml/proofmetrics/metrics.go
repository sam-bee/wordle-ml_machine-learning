// Package proofmetrics aggregates streaming, host-side diagnostics for a
// validation pass. It intentionally retains only scalar aggregates: callers
// can release each batch's logits as soon as it has been added.
package proofmetrics

import (
	"fmt"
	"math"

	"github.com/sam-bee/wordle-ml_machine-learning/tensorboard"
)

// TopK is the number of ranked teacher actions retained in WDIT v3.
const TopK = 16

const (
	numTurns              = 6
	openingCandidateCount = 2309
)

// Sample is one validation example after inference. Turn is the zero-based
// Wordle turn stored by the dataset; Result reports it as one-based turn 1..6.
//
// Supply Logits to calculate a legal argmax, entropy and teacher rank.
// MaskedPrediction is optional: use it when the inference path can expose the
// action selected after applying its mask. It allows this package to prove that
// the selected action itself is legal rather than confusing an unmasked raw
// argmax with a masking failure. When omitted, Add selects the best legal
// action from Logits.
// Availability uses zero for unavailable actions and a non-zero value for
// legal actions.
type Sample struct {
	Loss              float64
	Logits            []float32
	MaskedPrediction  *int
	TeacherTopActions [TopK]int32
	Turn              int
	CandidateCount    int
	Availability      []float32
	// Opening is rejected by Add: train opening evaluation belongs in
	// SetOpening and must not affect held-out validation metrics.
	Opening bool
}

// Result is JSON-friendly and deliberately ordered: ByTurn is always turns
// 1 through 6 and ByShortlistBucket is 1, 2-5, 6-20, 21-100, then >100.
type Result struct {
	Examples               int           `json:"examples"`
	Loss                   float64       `json:"loss"`
	Top1Accuracy           float64       `json:"top1_accuracy"`
	Top5Accuracy           float64       `json:"top5_accuracy"`
	Top16Accuracy          float64       `json:"top16_accuracy"`
	OutputEntropy          float64       `json:"output_entropy"`
	EntropyExamples        int           `json:"entropy_examples"`
	RawArgmaxUnavailable   int           `json:"raw_argmax_unavailable"`
	MaskedArgmaxViolations int           `json:"masked_argmax_violations"`
	ByTurn                 []GroupResult `json:"by_turn"`
	ByShortlistBucket      []GroupResult `json:"by_shortlist_bucket"`
	Opening                OpeningResult `json:"opening"`
}

// GroupResult contains the same compact metrics for a fixed validation group.
// Name is either turn_1 through turn_6 or a fixed shortlist bucket label.
type GroupResult struct {
	Name                   string  `json:"name"`
	Examples               int     `json:"examples"`
	Loss                   float64 `json:"loss"`
	Top1Accuracy           float64 `json:"top1_accuracy"`
	Top5Accuracy           float64 `json:"top5_accuracy"`
	Top16Accuracy          float64 `json:"top16_accuracy"`
	OutputEntropy          float64 `json:"output_entropy"`
	EntropyExamples        int     `json:"entropy_examples"`
	RawArgmaxUnavailable   int     `json:"raw_argmax_unavailable"`
	MaskedArgmaxViolations int     `json:"masked_argmax_violations"`
}

// OpeningResult keeps the empty-board diagnostic easy to inspect. TeacherRank
// is one-based. HighestGuess is -1 when no opening sample has been added.
type OpeningResult struct {
	Present      bool    `json:"present"`
	Loss         float64 `json:"loss"`
	TeacherRank  int     `json:"teacher_rank"`
	HighestGuess int     `json:"highest_guess"`
}

var shortlistBuckets = [...]string{"1", "2-5", "6-20", "21-100", ">100"}

// Collector is a streaming validation accumulator. It is not safe for
// concurrent use; validation is intentionally serial at the aggregation edge.
type Collector struct {
	overall    aggregate
	turns      [numTurns]aggregate
	shortlists [5]aggregate
	opening    openingAggregate
	openingSet bool
}

// Add adds one example after validating its diagnostic inputs.
func (c *Collector) Add(sample Sample) error {
	if sample.Opening {
		return fmt.Errorf("opening samples must be added with SetOpening, not Add")
	}
	measurement, err := validateAndMeasure(sample)
	if err != nil {
		return err
	}
	c.overall.add(sample.Loss, measurement)
	c.turns[sample.Turn].add(sample.Loss, measurement)
	c.shortlists[shortlistBucket(sample.CandidateCount)].add(sample.Loss, measurement)
	return nil
}

// SetOpening records the separately evaluated empty-board training example.
// It never alters the held-out validation aggregate or its turn/bucket groups.
func (c *Collector) SetOpening(sample Sample) error {
	if c.openingSet {
		return fmt.Errorf("opening result was already set")
	}
	if !sample.Opening {
		return fmt.Errorf("SetOpening requires the opening marker")
	}
	if sample.Turn != 0 || sample.CandidateCount != openingCandidateCount {
		return fmt.Errorf("opening must be turn 0 with %d candidates, got turn %d with %d", openingCandidateCount, sample.Turn, sample.CandidateCount)
	}
	measurement, err := validateAndMeasure(sample)
	if err != nil {
		return err
	}
	c.opening.set(sample.Loss, measurement)
	c.openingSet = true
	return nil
}

func validateAndMeasure(sample Sample) (measurement, error) {
	if !isFinite(sample.Loss) {
		return measurement{}, fmt.Errorf("loss must be finite, got %v", sample.Loss)
	}
	if sample.Turn < 0 || sample.Turn >= numTurns {
		return measurement{}, fmt.Errorf("turn %d outside fixed range 0..5", sample.Turn)
	}
	if sample.CandidateCount <= 0 {
		return measurement{}, fmt.Errorf("candidate count must be positive, got %d", sample.CandidateCount)
	}
	seenTeacher := make(map[int32]struct{}, TopK)
	for rank, action := range sample.TeacherTopActions {
		if action < 0 {
			return measurement{}, fmt.Errorf("teacher action at rank %d is negative: %d", rank+1, action)
		}
		if _, duplicate := seenTeacher[action]; duplicate {
			return measurement{}, fmt.Errorf("teacher action at rank %d duplicates an earlier action: %d", rank+1, action)
		}
		seenTeacher[action] = struct{}{}
	}
	return measure(sample)
}

// Result returns an immutable snapshot without retaining logits or examples.
func (c *Collector) Result() Result {
	result := c.overall.result()
	result.ByTurn = make([]GroupResult, len(c.turns))
	for turn := range c.turns {
		group := c.turns[turn].groupResult(fmt.Sprintf("turn_%d", turn+1))
		result.ByTurn[turn] = group
	}
	result.ByShortlistBucket = make([]GroupResult, len(c.shortlists))
	for bucket := range c.shortlists {
		result.ByShortlistBucket[bucket] = c.shortlists[bucket].groupResult(shortlistBuckets[bucket])
	}
	result.Opening = c.opening.result()
	return result
}

// TensorBoardScalars translates a snapshot to the fixed proof-plan scalar
// names. prefix normally is "validation". The output is deterministic and may
// be passed directly to tensorboard.Writer.WriteScalars.
func (r Result) TensorBoardScalars(prefix string) []tensorboard.Scalar {
	if prefix == "" {
		prefix = "validation"
	}
	scalars := make([]tensorboard.Scalar, 0, 7+len(r.ByTurn)*5+len(r.ByShortlistBucket)*5+3)
	appendMetrics := func(tag string, examples int, loss, top1, top5, top16, entropy float64) {
		if examples == 0 {
			return
		}
		scalars = append(scalars,
			tensorboard.Scalar{Tag: tag + "/loss", Value: float32(loss)},
			tensorboard.Scalar{Tag: tag + "/top1_accuracy", Value: float32(top1)},
			tensorboard.Scalar{Tag: tag + "/top5_accuracy", Value: float32(top5)},
			tensorboard.Scalar{Tag: tag + "/top16_accuracy", Value: float32(top16)},
			tensorboard.Scalar{Tag: tag + "/output_entropy", Value: float32(entropy)},
		)
	}
	appendMetrics(prefix, r.Examples, r.Loss, r.Top1Accuracy, r.Top5Accuracy, r.Top16Accuracy, r.OutputEntropy)
	if r.Examples > 0 {
		scalars = append(scalars,
			tensorboard.Scalar{Tag: prefix + "/raw_argmax_unavailable", Value: float32(r.RawArgmaxUnavailable)},
			tensorboard.Scalar{Tag: prefix + "/masked_argmax_violations", Value: float32(r.MaskedArgmaxViolations)},
			// This exact plan tag makes the validation output distribution
			// visible alongside the model's training diagnostics.
			tensorboard.Scalar{Tag: "model/output_entropy", Value: float32(r.OutputEntropy)},
		)
	}
	for _, group := range r.ByTurn {
		appendMetrics(prefix+"/"+group.Name, group.Examples, group.Loss, group.Top1Accuracy, group.Top5Accuracy, group.Top16Accuracy, group.OutputEntropy)
	}
	for _, group := range r.ByShortlistBucket {
		appendMetrics(prefix+"/shortlist_"+group.Name, group.Examples, group.Loss, group.Top1Accuracy, group.Top5Accuracy, group.Top16Accuracy, group.OutputEntropy)
	}
	if r.Opening.Present {
		scalars = append(scalars,
			tensorboard.Scalar{Tag: "opening/loss", Value: float32(r.Opening.Loss)},
			tensorboard.Scalar{Tag: "opening/teacher_rank", Value: float32(r.Opening.TeacherRank)},
			tensorboard.Scalar{Tag: "opening/highest_guess", Value: float32(r.Opening.HighestGuess)},
		)
	}
	return scalars
}

type measurement struct {
	prediction            int
	top1, top5, top16     bool
	entropy               float64
	rawArgmaxUnavailable  bool
	maskedArgmaxViolation bool
	teacherRank           int
}

func measure(sample Sample) (measurement, error) {
	if len(sample.Logits) == 0 {
		return measurement{}, fmt.Errorf("logits must not be empty")
	}
	if len(sample.Availability) != len(sample.Logits) {
		return measurement{}, fmt.Errorf("availability length %d, want logits length %d", len(sample.Availability), len(sample.Logits))
	}
	for rank, action := range sample.TeacherTopActions {
		if int(action) >= len(sample.Logits) {
			return measurement{}, fmt.Errorf("teacher action at rank %d (%d) outside logits length %d", rank+1, action, len(sample.Logits))
		}
		if sample.Availability[action] == 0 {
			return measurement{}, fmt.Errorf("teacher action at rank %d (%d) is unavailable", rank+1, action)
		}
	}

	rawBest, legalBest := -1, -1
	rawValue := float32(math.Inf(-1))
	legalValue := float32(math.Inf(-1))
	legalLogits := make([]float32, 0, len(sample.Logits))
	for action, logit := range sample.Logits {
		if !isFinite(float64(sample.Availability[action])) {
			return measurement{}, fmt.Errorf("availability %d must be finite, got %v", action, sample.Availability[action])
		}
		if sample.Availability[action] == 0 {
			// A hard availability mask commonly contributes negative infinity
			// at this point. It is valid only for an unavailable action.
			if math.IsNaN(float64(logit)) || math.IsInf(float64(logit), 1) {
				return measurement{}, fmt.Errorf("unavailable logit %d must be finite or -Inf, got %v", action, logit)
			}
			if rawBest < 0 || logit > rawValue {
				rawBest, rawValue = action, logit
			}
			continue
		}
		if !isFinite(float64(logit)) {
			return measurement{}, fmt.Errorf("legal logit %d must be finite, got %v", action, logit)
		}
		if rawBest < 0 || logit > rawValue {
			rawBest, rawValue = action, logit
		}
		legalLogits = append(legalLogits, logit)
		if legalBest < 0 || logit > legalValue {
			legalBest, legalValue = action, logit
		}
	}
	if legalBest < 0 {
		return measurement{}, fmt.Errorf("example has no legal actions")
	}
	prediction := legalBest
	maskedArgmaxViolation := false
	if sample.MaskedPrediction != nil {
		prediction = *sample.MaskedPrediction
		if prediction < 0 || prediction >= len(sample.Logits) {
			return measurement{}, fmt.Errorf("masked prediction %d outside logits length %d", prediction, len(sample.Logits))
		}
		maskedArgmaxViolation = sample.Availability[prediction] == 0
	}
	top1, top5, top16 := teacherMembership(prediction, sample.TeacherTopActions)
	return measurement{
		prediction:            prediction,
		top1:                  top1,
		top5:                  top5,
		top16:                 top16,
		entropy:               entropy(legalLogits),
		rawArgmaxUnavailable:  sample.Availability[rawBest] == 0,
		maskedArgmaxViolation: maskedArgmaxViolation,
		teacherRank:           rankOf(sample.TeacherTopActions[0], sample.Logits, sample.Availability),
	}, nil
}

func teacherMembership(prediction int, teacher [TopK]int32) (bool, bool, bool) {
	for rank, action := range teacher {
		if prediction != int(action) {
			continue
		}
		return rank == 0, rank < 5, true
	}
	return false, false, false
}

func rankOf(action int32, logits, availability []float32) int {
	if action < 0 || int(action) >= len(logits) || availability[action] == 0 {
		return -1
	}
	target := logits[action]
	rank := 1
	for candidate, logit := range logits {
		if availability[candidate] != 0 && (logit > target || (logit == target && candidate < int(action))) {
			rank++
		}
	}
	return rank
}

func entropy(logits []float32) float64 {
	maximum := float64(logits[0])
	for _, logit := range logits[1:] {
		maximum = max(maximum, float64(logit))
	}
	sum := 0.0
	for _, logit := range logits {
		sum += math.Exp(float64(logit) - maximum)
	}
	logZ := maximum + math.Log(sum)
	result := 0.0
	for _, logit := range logits {
		probability := math.Exp(float64(logit) - logZ)
		result -= probability * (float64(logit) - logZ)
	}
	return result
}

func shortlistBucket(count int) int {
	switch {
	case count == 1:
		return 0
	case count <= 5:
		return 1
	case count <= 20:
		return 2
	case count <= 100:
		return 3
	default:
		return 4
	}
}

type aggregate struct {
	examples, entropyExamples, top1, top5, top16, rawArgmaxUnavailable, maskedArgmaxViolations int
	loss, entropy                                                                              float64
}

func (a *aggregate) add(loss float64, m measurement) {
	a.examples++
	a.loss += loss
	if m.teacherRank >= 0 {
		a.entropy += m.entropy
		a.entropyExamples++
	}
	if m.top1 {
		a.top1++
	}
	if m.top5 {
		a.top5++
	}
	if m.top16 {
		a.top16++
	}
	if m.maskedArgmaxViolation {
		a.maskedArgmaxViolations++
	}
	if m.rawArgmaxUnavailable {
		a.rawArgmaxUnavailable++
	}
}

func (a aggregate) result() Result {
	return Result{Examples: a.examples, Loss: divide(a.loss, a.examples), Top1Accuracy: divide(float64(a.top1), a.examples), Top5Accuracy: divide(float64(a.top5), a.examples), Top16Accuracy: divide(float64(a.top16), a.examples), OutputEntropy: divide(a.entropy, a.entropyExamples), EntropyExamples: a.entropyExamples, RawArgmaxUnavailable: a.rawArgmaxUnavailable, MaskedArgmaxViolations: a.maskedArgmaxViolations}
}

func (a aggregate) groupResult(name string) GroupResult {
	return GroupResult{Name: name, Examples: a.examples, Loss: divide(a.loss, a.examples), Top1Accuracy: divide(float64(a.top1), a.examples), Top5Accuracy: divide(float64(a.top5), a.examples), Top16Accuracy: divide(float64(a.top16), a.examples), OutputEntropy: divide(a.entropy, a.entropyExamples), EntropyExamples: a.entropyExamples, RawArgmaxUnavailable: a.rawArgmaxUnavailable, MaskedArgmaxViolations: a.maskedArgmaxViolations}
}

type openingAggregate struct {
	examples int
	loss     float64
	last     measurement
}

func (a *openingAggregate) set(loss float64, m measurement) {
	a.examples = 1
	a.loss = loss
	a.last = m
}

func (a openingAggregate) result() OpeningResult {
	if a.examples == 0 {
		return OpeningResult{TeacherRank: -1, HighestGuess: -1}
	}
	return OpeningResult{Present: true, Loss: a.loss / float64(a.examples), TeacherRank: a.last.teacherRank, HighestGuess: a.last.prediction}
}

func divide(value float64, count int) float64 {
	if count == 0 {
		return 0
	}
	return value / float64(count)
}

func isFinite(value float64) bool { return !math.IsNaN(value) && !math.IsInf(value, 0) }

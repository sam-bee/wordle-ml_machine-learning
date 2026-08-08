package proofmetrics

import (
	"math"
	"reflect"
	"testing"

	"github.com/sam-bee/wordle-ml_machine-learning/tensorboard"
)

func TestCollectorSeparatesValidationAndOpeningMetrics(t *testing.T) {
	var collector Collector
	logits := logits17(0, 3, 4, 5)
	logits[16] = 9
	availability := ones17()
	availability[16] = 0
	if err := collector.Add(Sample{
		Loss: 2, Logits: logits, Availability: availability,
		TeacherTopActions: teacher(3, 2, 0), Turn: 0, CandidateCount: 1,
	}); err != nil {
		t.Fatalf("add validation: %v", err)
	}
	if err := collector.SetOpening(Sample{
		Loss: 4, Logits: logits17(0, 1, 2, 3), Availability: ones17(),
		TeacherTopActions: teacher(3, 2, 1), Turn: 0, CandidateCount: 2309, Opening: true,
	}); err != nil {
		t.Fatalf("set opening: %v", err)
	}

	result := collector.Result()
	if result.Examples != 1 || result.Loss != 2 {
		t.Fatalf("opening contaminated validation aggregate: %+v", result)
	}
	if result.Top1Accuracy != 1 || result.Top5Accuracy != 1 || result.Top16Accuracy != 1 {
		t.Fatalf("teacher-ranked accuracies = %+v", result)
	}
	if result.RawArgmaxUnavailable != 1 || result.MaskedArgmaxViolations != 0 {
		t.Fatalf("raw/masked diagnostics = %+v", result)
	}
	if result.EntropyExamples != 1 || !(result.OutputEntropy > 0 && result.OutputEntropy < math.Log(15)) {
		t.Fatalf("legal entropy = %v from %d examples", result.OutputEntropy, result.EntropyExamples)
	}
	if len(result.ByTurn) != 6 || result.ByTurn[0].Name != "turn_1" || result.ByTurn[0].Examples != 1 || result.ByTurn[5].Examples != 0 {
		t.Fatalf("turn groups = %+v", result.ByTurn)
	}
	if len(result.ByShortlistBucket) != 5 || result.ByShortlistBucket[0].Examples != 1 || result.ByShortlistBucket[4].Examples != 0 {
		t.Fatalf("shortlist groups = %+v", result.ByShortlistBucket)
	}
	if !result.Opening.Present || result.Opening.Loss != 4 || result.Opening.TeacherRank != 1 || result.Opening.HighestGuess != 3 {
		t.Fatalf("opening diagnostics = %+v", result.Opening)
	}
}

func TestMaskedPredictionViolationIsNotRawArgmaxDiagnostic(t *testing.T) {
	var collector Collector
	masked := 16 // Unavailable after masking: this is the actual violation.
	if err := collector.Add(Sample{
		Loss: 1, Logits: logits17(0, -5, 10), Availability: availability17(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
		MaskedPrediction: &masked, TeacherTopActions: teacher(2, 3, 4), Turn: 3, CandidateCount: 21,
	}); err != nil {
		t.Fatal(err)
	}
	result := collector.Result()
	if result.RawArgmaxUnavailable != 0 || result.MaskedArgmaxViolations != 1 {
		t.Fatalf("got raw=%d masked=%d, want raw=0 masked=1", result.RawArgmaxUnavailable, result.MaskedArgmaxViolations)
	}
}

func TestHardMaskedNegativeInfinityIsValidOutsideTheLegalSet(t *testing.T) {
	var collector Collector
	logits := logits17(0, 1)
	logits[16] = float32(math.Inf(-1))
	if err := collector.Add(Sample{
		Loss: 1, Logits: logits, Availability: availability17(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
		TeacherTopActions: teacher(1, 0), Turn: 1, CandidateCount: 2,
	}); err != nil {
		t.Fatalf("hard-masked logits were rejected: %v", err)
	}
	result := collector.Result()
	if result.RawArgmaxUnavailable != 0 || result.MaskedArgmaxViolations != 0 {
		t.Fatalf("hard mask produced false violation: %+v", result)
	}
}

func TestCollectorUsesPlanShortlistBucketBoundaries(t *testing.T) {
	var collector Collector
	for _, candidateCount := range []int{1, 2, 5, 6, 20, 21, 100, 101} {
		if err := collector.Add(Sample{
			Loss:              float64(candidateCount),
			Logits:            logits17(1, 0),
			Availability:      ones17(),
			TeacherTopActions: teacher(0),
			Turn:              0,
			CandidateCount:    candidateCount,
		}); err != nil {
			t.Fatalf("add candidate count %d: %v", candidateCount, err)
		}
	}

	got := collector.Result().ByShortlistBucket
	want := []struct {
		name     string
		examples int
		loss     float64
	}{
		{"1", 1, 1},
		{"2-5", 2, 3.5},
		{"6-20", 2, 13},
		{"21-100", 2, 60.5},
		{">100", 1, 101},
	}
	if len(got) != len(want) {
		t.Fatalf("shortlist buckets=%d, want %d", len(got), len(want))
	}
	for index, expected := range want {
		bucket := got[index]
		if bucket.Name != expected.name || bucket.Examples != expected.examples || bucket.Loss != expected.loss ||
			bucket.Top1Accuracy != 1 || bucket.Top5Accuracy != 1 || bucket.Top16Accuracy != 1 {
			t.Errorf("bucket %d=%+v, want name=%q examples=%d loss=%g and perfect top-k", index, bucket, expected.name, expected.examples, expected.loss)
		}
	}
}

func TestCollectorRejectsInvalidInputs(t *testing.T) {
	cases := []struct {
		name   string
		sample Sample
	}{
		{"non-finite loss", Sample{Loss: math.Inf(1), MaskedPrediction: action(0), Turn: 0, CandidateCount: 1, TeacherTopActions: teacher()}},
		{"bad turn", Sample{Loss: 1, MaskedPrediction: action(0), Turn: 6, CandidateCount: 1, TeacherTopActions: teacher()}},
		{"no legal action", Sample{Loss: 1, Logits: logits17(), Availability: availability17(), Turn: 0, CandidateCount: 1, TeacherTopActions: teacher()}},
		{"teacher unavailable", Sample{Loss: 1, Logits: logits17(), Availability: availability17(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16), Turn: 0, CandidateCount: 1, TeacherTopActions: teacher(0)}},
		{"duplicate teacher ID", Sample{Loss: 1, Logits: logits17(), Availability: ones17(), Turn: 0, CandidateCount: 1, TeacherTopActions: duplicateTeacher()}},
		{"opening added as validation", Sample{Loss: 1, Logits: logits17(), Availability: ones17(), Turn: 0, CandidateCount: 1, TeacherTopActions: teacher(), Opening: true}},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			var collector Collector
			if err := collector.Add(test.sample); err == nil {
				t.Fatal("Add unexpectedly succeeded")
			}
		})
	}
}

func TestTensorBoardScalarsAreStableAndUsePlanTags(t *testing.T) {
	var collector Collector
	if err := collector.Add(Sample{
		Loss: 1, Logits: logits17(0, 1), Availability: ones17(),
		TeacherTopActions: teacher(1, 0), Turn: 2, CandidateCount: 6,
	}); err != nil {
		t.Fatal(err)
	}
	if err := collector.SetOpening(Sample{
		Loss: 1, Logits: logits17(0, 1), Availability: ones17(),
		TeacherTopActions: teacher(1, 0), Turn: 0, CandidateCount: 2309, Opening: true,
	}); err != nil {
		t.Fatal(err)
	}
	first := collector.Result().TensorBoardScalars("validation")
	second := collector.Result().TensorBoardScalars("validation")
	if !reflect.DeepEqual(first, second) {
		t.Fatalf("scalar lists are not deterministic\n%+v\n%+v", first, second)
	}
	for _, wanted := range []string{
		"validation/loss", "validation/top1_accuracy", "validation/top5_accuracy", "validation/top16_accuracy",
		"validation/turn_3/loss", "validation/shortlist_6-20/loss", "validation/raw_argmax_unavailable",
		"validation/masked_argmax_violations", "model/output_entropy", "opening/loss", "opening/teacher_rank", "opening/highest_guess",
	} {
		if !containsTag(first, wanted) {
			t.Errorf("missing TensorBoard scalar %q", wanted)
		}
	}
}

func teacher(prefix ...int32) [TopK]int32 {
	var result [TopK]int32
	used := make(map[int32]struct{}, TopK)
	next := int32(0)
	for index := range result {
		if index < len(prefix) {
			result[index] = prefix[index]
			used[prefix[index]] = struct{}{}
			continue
		}
		for {
			if _, exists := used[next]; !exists {
				break
			}
			next++
		}
		result[index] = next
		used[next] = struct{}{}
		next++
	}
	return result
}

func duplicateTeacher() [TopK]int32 {
	result := teacher()
	result[1] = result[0]
	return result
}

func logits17(prefix ...float32) []float32 {
	result := make([]float32, TopK+1)
	for index := range result {
		result[index] = -100
	}
	copy(result, prefix)
	return result
}

func ones17() []float32 {
	result := make([]float32, TopK+1)
	for index := range result {
		result[index] = 1
	}
	return result
}

func availability17(available ...int) []float32 {
	result := make([]float32, TopK+1)
	for _, action := range available {
		result[action] = 1
	}
	return result
}

func action(value int) *int { return &value }

func containsTag(scalars []tensorboard.Scalar, wanted string) bool {
	for _, scalar := range scalars {
		if scalar.Tag == wanted {
			return true
		}
	}
	return false
}

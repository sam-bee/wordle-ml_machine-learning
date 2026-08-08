package proofrun

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/gomlx/gomlx/ml/model"
	"github.com/sam-bee/wordle-ml_machine-learning/imitationdata"
	"github.com/sam-bee/wordle-ml_machine-learning/proofmetrics"
	"github.com/sam-bee/wordle-ml_machine-learning/supervised"
	"github.com/sam-bee/wordle-ml_machine-learning/tensorboard"
)

func TestLegalCrossEntropyAndHardMaskRejectInvalidSelections(t *testing.T) {
	raw := []float32{0, 100, 2}
	available := []float32{1, 0, 1}
	masked := []float32{0, float32(math.Inf(-1)), 2}
	prediction, err := verifyHardMask(raw, masked, available)
	if err != nil {
		t.Fatalf("verifyHardMask: %v", err)
	}
	if prediction != 2 {
		t.Fatalf("prediction = %d, want legal action 2", prediction)
	}
	loss, err := legalCrossEntropy(raw, available, 2)
	if err != nil {
		t.Fatalf("legalCrossEntropy: %v", err)
	}
	want := math.Log1p(math.Exp(-2))
	if math.Abs(loss-want) > 1e-6 {
		t.Fatalf("loss = %.9f, want %.9f", loss, want)
	}
	if _, err := verifyHardMask(raw, []float32{0, -1e9, 2}, available); err == nil {
		t.Fatal("finite unavailable mask was accepted")
	}
	if _, err := legalCrossEntropy(raw, available, 1); err == nil {
		t.Fatal("unavailable teacher target was accepted")
	}
}

func TestDeterministicReservoirIsBoundedAndUsesLaterValues(t *testing.T) {
	var seen uint64
	values := make([]float64, 0, 8)
	for value := 0; value < 200; value++ {
		values = appendSampledFinite(values, &seen, float64(value), 8)
	}
	if len(values) != 8 || seen != 200 {
		t.Fatalf("reservoir len/seen = %d/%d, want 8/200", len(values), seen)
	}
	containsLate := false
	for _, value := range values {
		if value >= 8 {
			containsLate = true
		}
	}
	if !containsLate {
		t.Fatalf("reservoir retained only first values: %v", values)
	}
}

func TestStoreFingerprintChangesOnlyWhenValuesChange(t *testing.T) {
	store := model.NewStore()
	variable := store.VariableWithValue("/proof/value", []float32{1, 2})
	first, err := StoreFingerprint(store)
	if err != nil {
		t.Fatalf("first fingerprint: %v", err)
	}
	second, err := StoreFingerprint(store)
	if err != nil || second != first {
		t.Fatalf("unchanged Store fingerprint = %q, %v; want %q", second, err, first)
	}
	variable.MustSetValue(model.NewStore().VariableWithValue("/replacement", []float32{2, 3}).MustValue())
	fourth, err := StoreFingerprint(store)
	if err != nil {
		t.Fatalf("changed fingerprint: %v", err)
	}
	if fourth == first {
		t.Fatal("changed Store value retained its fingerprint")
	}
}

func TestTrainingTelemetryContainsRequiredScalarTags(t *testing.T) {
	dir := t.TempDir()
	writer, err := tensorboard.New(dir)
	if err != nil {
		t.Fatal(err)
	}
	example := imitationdata.Example{CandidateMask: []float32{1, 1, 1}}
	err = writeTrainingTelemetry(writer, 10, Metrics{Loss: 1, Top1: .1, Top5: .2, Top16: .3}, supervised.TrainingDiagnostics{
		PreclipGlobalGradientNorm: 2, AppliedGlobalGradientNorm: 2, GradientsFinite: true, ParametersFinite: true, ParameterNorm: 3, UpdateToParameterNorm: .01, LearningRate: .0003,
	}, []imitationdata.Example{example}, 2, 1280, 20*time.Millisecond, 3*time.Millisecond, proofmetrics.OpeningResult{Present: true, Loss: 2, TeacherRank: 4, HighestGuess: 7})
	if err != nil {
		t.Fatal(err)
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	files, err := filepath.Glob(filepath.Join(dir, "events.out.tfevents.*"))
	if err != nil || len(files) != 1 {
		t.Fatalf("event files = %v, %v", files, err)
	}
	contents, err := os.ReadFile(files[0])
	if err != nil {
		t.Fatal(err)
	}
	for _, tag := range []string{
		"train/loss", "optimizer/learning_rate", "optimizer/global_gradient_norm", "optimizer/parameter_norm", "optimizer/update_to_parameter_norm",
		"data/epoch", "data/examples_consumed", "data/shortlist_size_mean", "performance/examples_per_second", "performance/batch_duration", "performance/input_wait_duration",
		"opening/loss", "opening/teacher_rank", "opening/current_guess_id",
	} {
		if !strings.Contains(string(contents), tag) {
			t.Errorf("missing TensorBoard scalar tag %q", tag)
		}
	}
}

func TestMajorGroupLearningReportsOnlyNonEmptyImprovedGroups(t *testing.T) {
	initial := ValidationSnapshot{Details: proofmetrics.Result{
		ByTurn:            []proofmetrics.GroupResult{{Name: "turn_1", Examples: 25, Loss: 2, Top1Accuracy: .1}, {Name: "turn_2", Examples: 24, Loss: 2}},
		ByShortlistBucket: []proofmetrics.GroupResult{{Name: "1", Examples: 25, Loss: 3, Top1Accuracy: .2}},
	}}
	best := ValidationSnapshot{Details: proofmetrics.Result{
		ByTurn:            []proofmetrics.GroupResult{{Name: "turn_1", Examples: 25, Loss: 1.9, Top1Accuracy: .1}, {Name: "turn_2", Examples: 24, Loss: 1}},
		ByShortlistBucket: []proofmetrics.GroupResult{{Name: "1", Examples: 25, Loss: 2.9, Top1Accuracy: .2}},
	}}
	got := majorGroupLearning(initial, best)
	if got.MinimumExamples != 25 || got.TurnCount != 1 || got.ShortlistCount != 1 || got.Count != 2 || strings.Join(got.Groups, ",") != "turn/turn_1,shortlist/1" {
		t.Fatalf("major group learning = %+v", got)
	}
}

func TestValidationSnapshotEvidenceRejectsMissingOrInconsistentGroups(t *testing.T) {
	turns := make([]proofmetrics.GroupResult, 6)
	for index := range turns {
		examples := 100
		if index == 0 {
			examples = 0
		}
		turns[index] = proofmetrics.GroupResult{Name: fmt.Sprintf("turn_%d", index+1), Examples: examples, EntropyExamples: examples, Loss: 2}
	}
	shortlists := []proofmetrics.GroupResult{
		{Name: "1", Examples: 100, EntropyExamples: 100, Loss: 1},
		{Name: "2-5", Examples: 100, EntropyExamples: 100, Loss: 2},
		{Name: "6-20", Examples: 100, EntropyExamples: 100, Loss: 3},
		{Name: "21-100", Examples: 100, EntropyExamples: 100, Loss: 4},
		{Name: ">100", Examples: 100, EntropyExamples: 100, Loss: 5},
	}
	snapshot := ValidationSnapshot{
		Metrics: Metrics{Loss: 3, Top1: .1, Top5: .2, Top16: .3},
		Details: proofmetrics.Result{
			Examples:          500,
			Loss:              3,
			Top1Accuracy:      .1,
			Top5Accuracy:      .2,
			Top16Accuracy:     .3,
			EntropyExamples:   500,
			ByTurn:            turns,
			ByShortlistBucket: shortlists,
			Opening:           proofmetrics.OpeningResult{Present: true, TeacherRank: 2311, HighestGuess: 1},
		},
	}
	if err := validateValidationSnapshotEvidence(snapshot, 500); err != nil {
		t.Fatalf("valid evidence: %v", err)
	}

	for name, mutate := range map[string]func(*ValidationSnapshot){
		"empty details":   func(candidate *ValidationSnapshot) { candidate.Details = proofmetrics.Result{} },
		"metric mismatch": func(candidate *ValidationSnapshot) { candidate.Metrics.Loss++ },
		"missing turn":    func(candidate *ValidationSnapshot) { candidate.Details.ByTurn = candidate.Details.ByTurn[:5] },
		"turn total":      func(candidate *ValidationSnapshot) { candidate.Details.ByTurn[1].Examples++ },
		"wrong shortlist": func(candidate *ValidationSnapshot) { candidate.Details.ByShortlistBucket[0].Name = "wrong" },
		"missing opening": func(candidate *ValidationSnapshot) { candidate.Details.Opening.Present = false },
	} {
		t.Run(name, func(t *testing.T) {
			candidate := snapshot
			candidate.Details.ByTurn = slices.Clone(snapshot.Details.ByTurn)
			candidate.Details.ByShortlistBucket = slices.Clone(snapshot.Details.ByShortlistBucket)
			mutate(&candidate)
			if err := validateValidationSnapshotEvidence(candidate, 500); err == nil {
				t.Fatal("invalid evidence was accepted")
			}
		})
	}
}

func TestPriorValidationEvidenceChecksAllSummariesAndHistory(t *testing.T) {
	snapshot := validValidationSnapshotForTest()
	initial := snapshot
	initial.Update = 0
	final := snapshot
	final.Update = 1000
	best := snapshot
	best.Update = 900
	result := Result{
		GlobalUpdate:             1000,
		InitialValidation:        initial.Metrics,
		FinalValidation:          final.Metrics,
		BestValidation:           best.Metrics,
		BestValidationStep:       900,
		InitialValidationDetails: initial,
		FinalValidationDetails:   final,
		BestValidationDetails:    best,
		ValidationSnapshots:      []ValidationSnapshot{initial, best, final},
	}
	if err := validatePriorValidationEvidence(result, 500); err != nil {
		t.Fatalf("valid prior evidence: %v", err)
	}
	mismatchedFinal := result
	mismatchedFinal.FinalValidationDetails = initial
	if err := validatePriorValidationEvidence(mismatchedFinal, 500); err == nil || !strings.Contains(err.Error(), "final snapshot") {
		t.Fatalf("mismatched final snapshot error = %v", err)
	}
	result.ValidationSnapshots[1].Details.ByShortlistBucket = nil
	if err := validatePriorValidationEvidence(result, 500); err == nil || !strings.Contains(err.Error(), "history snapshot 1") {
		t.Fatalf("malformed history error = %v", err)
	}
}

func validValidationSnapshotForTest() ValidationSnapshot {
	turns := make([]proofmetrics.GroupResult, 6)
	for index := range turns {
		examples := 100
		if index == 0 {
			examples = 0
		}
		turns[index] = proofmetrics.GroupResult{Name: fmt.Sprintf("turn_%d", index+1), Examples: examples, EntropyExamples: examples, Loss: 2}
	}
	shortlists := make([]proofmetrics.GroupResult, 5)
	for index, name := range []string{"1", "2-5", "6-20", "21-100", ">100"} {
		shortlists[index] = proofmetrics.GroupResult{Name: name, Examples: 100, EntropyExamples: 100, Loss: float64(index + 1)}
	}
	return ValidationSnapshot{
		Metrics: Metrics{Loss: 3, Top1: .1, Top5: .2, Top16: .3},
		Details: proofmetrics.Result{
			Examples: 500, Loss: 3, Top1Accuracy: .1, Top5Accuracy: .2, Top16Accuracy: .3, EntropyExamples: 500,
			ByTurn: turns, ByShortlistBucket: shortlists,
			Opening: proofmetrics.OpeningResult{Present: true, TeacherRank: 2311, HighestGuess: 1},
		},
	}
}

package proofrun

import (
	"math"
	"os"
	"path/filepath"
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

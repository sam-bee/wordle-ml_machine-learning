package proofeval

import (
	"encoding/json"
	"math"
	"os"
	"strings"
	"testing"

	"github.com/sam-bee/wordle-ml_machine-learning/gameeval"
	"github.com/sam-bee/wordle-ml_machine-learning/proofgames"
	"github.com/sam-bee/wordle-ml_machine-learning/proofmetrics"
	"github.com/sam-bee/wordle-ml_machine-learning/proofrun"
	"github.com/sam-bee/wordle-ml_machine-learning/runstate"
	"github.com/sam-bee/wordle-ml_machine-learning/tensorboard"
)

func TestCrossEntropyIgnoresHardMaskedActions(t *testing.T) {
	logits := []float32{0, float32(math.Inf(-1)), float32(math.Log(3))}
	loss, err := crossEntropy(logits, 2)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := loss, math.Log(4.0/3.0); math.Abs(got-want) > 1e-7 {
		t.Fatalf("loss = %.9f, want %.9f", got, want)
	}
	if _, err := crossEntropy(logits, 1); err == nil {
		t.Fatal("masked target was accepted")
	}
}

func TestBestReproductionChecksMetricsAndMajorGroups(t *testing.T) {
	layout, err := runstate.Create(t.TempDir(), "full-proof")
	if err != nil {
		t.Fatal(err)
	}
	metrics := proofmetrics.Result{
		Loss: 1.25, Top1Accuracy: .2, Top5Accuracy: .4, Top16Accuracy: .8,
		ByTurn:            []proofmetrics.GroupResult{{Name: "turn_1", Examples: 10, Loss: 1.1, Top1Accuracy: .3, Top5Accuracy: .5, Top16Accuracy: .9}},
		ByShortlistBucket: []proofmetrics.GroupResult{{Name: "1", Examples: 4, Loss: 1.2, Top1Accuracy: .4, Top5Accuracy: .6, Top16Accuracy: 1}},
	}
	stored := proofrun.Metrics{Loss: metrics.Loss, Top1: metrics.Top1Accuracy, Top5: metrics.Top5Accuracy, Top16: metrics.Top16Accuracy}
	if err := layout.WriteFinalMetricsJSON(proofrun.Result{BestValidation: stored, BestValidationStep: 100, BestValidationDetails: proofrun.ValidationSnapshot{Details: metrics}}); err != nil {
		t.Fatal(err)
	}
	state := runstate.State{GlobalUpdate: 100, ShuffleSeed: 1, BestValidation: &runstate.BestValidation{Value: 1.25, Update: 100}}
	got, err := checkBestReproduction(metrics, state, layout)
	if err != nil || !got.GroupsMatch {
		t.Fatalf("checkBestReproduction = %+v, %v", got, err)
	}
	metrics.ByTurn[0].Loss++
	if _, err := checkBestReproduction(metrics, state, layout); err == nil {
		t.Fatal("changed major group was accepted")
	}
}

func TestHostMaskMakesExtremeUnavailableActionImpossible(t *testing.T) {
	raw := make([]float32, 4739)
	available := make([]float32, len(raw))
	for i := range available {
		available[i] = 1
	}
	raw[7] = math.MaxFloat32
	raw[3] = 1
	available[7] = 0
	masked := applyHostMask(raw, available)
	if err := validatePrediction(raw, masked, available); err != nil {
		t.Fatal(err)
	}
	if got := argMax(masked); got != 3 {
		t.Fatalf("hard-masked argmax = %d, want 3", got)
	}
	if !math.IsInf(float64(masked[7]), -1) {
		t.Fatalf("unavailable action = %v, want -Inf", masked[7])
	}
}

func TestMeasurementIsAblatedMinusNormal(t *testing.T) {
	normal := proofmetrics.Result{Loss: 3, Top1Accuracy: .5, Top5Accuracy: .7, Top16Accuracy: .9}
	abl := proofmetrics.Result{Loss: 4, Top1Accuracy: .2, Top5Accuracy: .4, Top16Accuracy: .8}
	got := measurement(normal, abl).Effect
	if got.Loss != 1 || math.Abs(got.Top1+.3) > 1e-12 || math.Abs(got.Top5+.3) > 1e-12 || math.Abs(got.Top16+.1) > 1e-12 {
		t.Fatalf("unexpected effect: %+v", got)
	}
}

func TestOptionsRejectSealedTestLikeMode(t *testing.T) {
	err := validateOptions(Options{DataDir: "data", RunsDir: "runs", RunID: "proof", Checkpoint: Best, Mode: "test"})
	if err == nil {
		t.Fatal("test-like mode unexpectedly accepted")
	}
}

func TestAllowedStageCheckpointModeCombinations(t *testing.T) {
	allowed := []struct {
		stage      proofrun.Stage
		checkpoint Checkpoint
		mode       Mode
	}{
		{proofrun.Mini, Latest, Games10},
		{proofrun.Full, Initial, Games100},
		{proofrun.Full, Best, Games100},
		{proofrun.Full, Best, Ablations},
	}
	for _, test := range allowed {
		if err := validateCombination(test.stage, test.checkpoint, test.mode); err != nil {
			t.Errorf("%+v rejected: %v", test, err)
		}
	}
	for _, test := range []struct {
		stage      proofrun.Stage
		checkpoint Checkpoint
		mode       Mode
	}{
		{proofrun.Overfit, Initial, Games10}, {proofrun.Overfit, Latest, Games10}, {proofrun.Mini, Initial, Games10}, {proofrun.Full, Latest, Games100}, {proofrun.Full, Best, Games10}, {proofrun.Mini, Best, Ablations},
	} {
		if err := validateCombination(test.stage, test.checkpoint, test.mode); err == nil {
			t.Errorf("%+v unexpectedly allowed", test)
		}
	}
}

func TestPersistEvaluationAddsAndPreservesRawEntries(t *testing.T) {
	layout, err := runstate.Create(t.TempDir(), "mini-proof")
	if err != nil {
		t.Fatal(err)
	}
	if err := layout.WriteFinalMetricsJSON(proofrun.Result{Stage: proofrun.Mini, Passed: true}); err != nil {
		t.Fatal(err)
	}
	first := Result{Stage: proofrun.Mini, Mode: Games10, Checkpoint: Latest, CheckpointUpdate: 1000, ValidationSplitHash: "split"}
	if err := persistEvaluation(layout, first); err != nil {
		t.Fatal(err)
	}
	if err := persistEvaluation(layout, first); err != nil {
		t.Fatalf("identical evaluation retry: %v", err)
	}
	if err := persistEvaluation(layout, Result{Stage: proofrun.Mini, Mode: Games10, Checkpoint: Latest, CheckpointUpdate: 1001, ValidationSplitHash: "split"}); err == nil {
		t.Fatal("different stable evaluation key replaced")
	}
	second := Result{Stage: proofrun.Mini, Mode: Games10, Checkpoint: Initial, CheckpointUpdate: 0, ValidationSplitHash: "split"}
	if err := persistEvaluation(layout, second); err != nil {
		t.Fatal(err)
	}
	contents, err := os.ReadFile(layout.FinalMetricsPath)
	if err != nil {
		t.Fatal(err)
	}
	var document struct {
		Evaluations map[string]json.RawMessage `json:"evaluations"`
	}
	if err := json.Unmarshal(contents, &document); err != nil {
		t.Fatal(err)
	}
	if len(document.Evaluations) != 2 {
		t.Fatalf("evaluation count = %d, want 2", len(document.Evaluations))
	}
	var got Result
	if err := json.Unmarshal(document.Evaluations["latest-games10"], &got); err != nil || got.CheckpointUpdate != 1000 {
		t.Fatalf("stored latest evaluation = %+v, %v", got, err)
	}
}

func TestPersistEvaluationRequiresPassedMiniAndFull(t *testing.T) {
	for _, stage := range []proofrun.Stage{proofrun.Mini, proofrun.Full} {
		layout, err := runstate.Create(t.TempDir(), "proof-"+string(stage))
		if err != nil {
			t.Fatal(err)
		}
		if err := layout.WriteFinalMetricsJSON(proofrun.Result{Stage: stage, Passed: false}); err != nil {
			t.Fatal(err)
		}
		if err := persistEvaluation(layout, Result{Stage: stage, Mode: Games10, Checkpoint: Latest}); err == nil {
			t.Errorf("unpassed %s evaluation persisted", stage)
		}
	}
}

func TestEvaluationPreflightRejectsUnpassedTrainingWithoutArtifacts(t *testing.T) {
	layout, err := runstate.Create(t.TempDir(), "unpassed-mini")
	if err != nil {
		t.Fatal(err)
	}
	if err := layout.WriteFinalMetricsJSON(proofrun.Result{Stage: proofrun.Mini, Passed: false}); err != nil {
		t.Fatal(err)
	}
	if err := validateEvaluationTrainingComplete(layout, proofrun.Mini); err == nil {
		t.Fatal("unpassed mini evaluation preflight succeeded")
	}
	entries, err := os.ReadDir(layout.EvaluationsDir)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 0 {
		t.Fatalf("unpassed preflight wrote evaluation artifacts: %v", entries)
	}
}

func TestCanonicalReportTrajectorySelection(t *testing.T) {
	for _, test := range []struct {
		stage      proofrun.Stage
		checkpoint Checkpoint
		mode       Mode
		want       bool
	}{
		{proofrun.Mini, Latest, Games10, true},
		{proofrun.Full, Best, Games100, true},
		{proofrun.Overfit, Initial, Games10, false},
		{proofrun.Full, Initial, Games100, false},
		{proofrun.Full, Best, Ablations, false},
	} {
		if got := isCanonicalReportTrajectory(test.stage, test.checkpoint, test.mode); got != test.want {
			t.Errorf("canonical(%s,%s,%s) = %t, want %t", test.stage, test.checkpoint, test.mode, got, test.want)
		}
	}
}

func TestSameJSONAcceptsSemanticIdentityOnly(t *testing.T) {
	same, err := sameJSON([]byte("{\n  \"checkpoint\": \"best\", \"mode\": \"games100\"\n}"), []byte(`{"mode":"games100","checkpoint":"best"}`))
	if err != nil || !same {
		t.Fatalf("semantic JSON identity = %t, %v", same, err)
	}
	same, err = sameJSON([]byte(`{"checkpoint":"best"}`), []byte(`{"checkpoint":"latest"}`))
	if err != nil || same {
		t.Fatalf("different JSON identity = %t, %v", same, err)
	}
}

func TestGameTelemetryIsIdempotentAndRejectsPartialDuplicateOrMismatch(t *testing.T) {
	evaluation := proofgames.Evaluation{Summary: gameeval.Summary{SolvedFraction: .5, MeanGuesses: 3, Failures: 5, GuessCountDistribution: [6]int{1, 2, 3, 4, 5, 6}}}
	t.Run("idempotent", func(t *testing.T) {
		dir := t.TempDir()
		if err := writeGameScalars(dir, 100, evaluation); err != nil {
			t.Fatal(err)
		}
		if err := writeGameScalars(dir, 100, evaluation); err != nil {
			t.Fatalf("idempotent retry: %v", err)
		}
		inspection, err := tensorboard.InspectDir(dir)
		if err != nil {
			t.Fatal(err)
		}
		if got := countGameRecords(inspection.Scalars, 100); got != len(proofgames.TensorBoardScalars(evaluation)) {
			t.Fatalf("game scalar records = %d, want one set", got)
		}
	})
	for _, test := range []struct {
		name  string
		write func(*tensorboard.Writer, []tensorboard.Scalar) error
	}{
		{"partial", func(writer *tensorboard.Writer, expected []tensorboard.Scalar) error {
			return writer.WriteScalars(100, expected[0])
		}},
		{"duplicate", func(writer *tensorboard.Writer, expected []tensorboard.Scalar) error {
			return writer.WriteScalars(100, expected[0], expected[0])
		}},
		{"mismatch", func(writer *tensorboard.Writer, expected []tensorboard.Scalar) error {
			return writer.WriteScalars(100, tensorboard.Scalar{Tag: expected[0].Tag, Value: expected[0].Value + 1})
		}},
	} {
		t.Run(test.name, func(t *testing.T) {
			dir := t.TempDir()
			writer, err := tensorboard.New(dir)
			if err != nil {
				t.Fatal(err)
			}
			if err := test.write(writer, proofgames.TensorBoardScalars(evaluation)); err != nil {
				t.Fatal(err)
			}
			if err := writer.Close(); err != nil {
				t.Fatal(err)
			}
			if err := writeGameScalars(dir, 100, evaluation); err == nil {
				t.Fatal("inconsistent game telemetry was accepted")
			}
		})
	}
}

func countGameRecords(records []tensorboard.ScalarRecord, step int64) int {
	count := 0
	for _, record := range records {
		if record.Step == step && strings.HasPrefix(record.Tag, "games/") {
			count++
		}
	}
	return count
}

package supervised

import (
	"bytes"
	"math"
	"sort"
	"testing"

	"github.com/gomlx/compute"
	_ "github.com/gomlx/gomlx/backends/default"
	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/sam-bee/wordle-ml_machine-learning/policy"
)

var testBackend = compute.MustNew()

func TestApplyAvailabilityMaskUsesNegativeInfinityAndCannotSelectMaskedAction(t *testing.T) {
	exec := model.MustNewExec(testBackend, model.NewStore(), func(_ *model.Scope, logits, available *graph.Node) (*graph.Node, *graph.Node) {
		masked := ApplyAvailabilityMask(logits, available)
		return masked, graph.ArgMax(masked, -1)
	})
	defer exec.Finalize()

	outputs := exec.MustCall(
		[][]float32{{-1e30, math.MaxFloat32, -100}},
		[][]float32{{1, 0, 1}},
	)
	defer finalizeAll(t, outputs)
	got := outputs[0].Value().([][]float32)[0]
	if got[0] != -1e30 || got[2] != -100 {
		t.Fatalf("available actions changed: %v", got)
	}
	if !math.IsInf(float64(got[1]), -1) {
		t.Fatalf("masked action = %v, want -Inf", got[1])
	}
	if selected := outputs[1].Value().([]int32)[0]; selected != 2 {
		t.Fatalf("ArgMax selected action %d, want available action 2", selected)
	}
}

func TestRankedTeacherTopKAccuracyGraph(t *testing.T) {
	exec := model.MustNewExec(testBackend, model.NewStore(), func(scope *model.Scope, labels, teacherRanking, logits *graph.Node) (*graph.Node, *graph.Node, *graph.Node) {
		allLabels := []*graph.Node{labels, teacherRanking}
		return topKAccuracyGraph(1)(scope, allLabels, []*graph.Node{logits}),
			topKAccuracyGraph(5)(scope, allLabels, []*graph.Node{logits}),
			topKAccuracyGraph(16)(scope, allLabels, []*graph.Node{logits})
	})
	defer exec.Finalize()

	logits := make([][]float32, 4)
	for row := range logits {
		logits[row] = make([]float32, 17)
		for action := range logits[row] {
			logits[row][action] = float32(17 - action)
		}
	}
	teacherRanking := [][]int32{
		{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
		{4, 0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
		{1, 2, 3, 4, 5, 0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
		{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
	}
	got := exec.MustCall([][]int32{{0}, {0}, {0}, {0}}, teacherRanking, logits)
	defer finalizeAll(t, got)
	assertClose(t, scalar(got[0]), 0.25)
	assertClose(t, scalar(got[1]), 0.5)
	assertClose(t, scalar(got[2]), 0.75)
}

func TestTopKAccuracyGraphFallsBackForLegacyOneLabelBatch(t *testing.T) {
	exec := model.MustNewExec(testBackend, model.NewStore(), func(scope *model.Scope, labels, logits *graph.Node) (*graph.Node, *graph.Node, *graph.Node) {
		return topKAccuracyGraph(1)(scope, []*graph.Node{labels}, []*graph.Node{logits}),
			topKAccuracyGraph(5)(scope, []*graph.Node{labels}, []*graph.Node{logits}),
			topKAccuracyGraph(16)(scope, []*graph.Node{labels}, []*graph.Node{logits})
	})
	defer exec.Finalize()

	logits := make([][]float32, 4)
	for row := range logits {
		logits[row] = make([]float32, 16)
		for action := range logits[row] {
			logits[row][action] = float32(16 - action)
		}
	}
	got := exec.MustCall([][]int32{{0}, {4}, {10}, {15}}, logits)
	defer finalizeAll(t, got)
	assertClose(t, scalar(got[0]), 0.25)
	assertClose(t, scalar(got[1]), 0.5)
	assertClose(t, scalar(got[2]), 1)
}

func TestMaskedLossUsesOnlyTopOneLabel(t *testing.T) {
	exec := model.MustNewExec(testBackend, model.NewStore(), func(_ *model.Scope, topOne, teacherTop16, logits *graph.Node) *graph.Node {
		return maskedSparseCategoricalCrossEntropy(
			[]*graph.Node{topOne, teacherTop16},
			[]*graph.Node{logits},
		)
	})
	defer exec.Finalize()

	// The middle action models a hard-masked unavailable action. The custom loss
	// must keep the legal target finite without looking at labels[1].
	logits := [][]float32{{2, float32(math.Inf(-1)), -3}}
	first := exec.MustCall1(
		[][]int32{{0}},
		[][]int32{{2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
		logits,
	)
	second := exec.MustCall1(
		[][]int32{{0}},
		[][]int32{{1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
		logits,
	)
	defer func() {
		_ = first.FinalizeAll()
		_ = second.FinalizeAll()
	}()
	if !isFinite(scalar(first)) || !isFinite(scalar(second)) {
		t.Fatalf("losses must remain finite: %v, %v", scalar(first), scalar(second))
	}
	assertClose(t, scalar(first), scalar(second))
}

func TestMaskedLossPreservesUnexpectedNonFiniteLogits(t *testing.T) {
	exec := model.MustNewExec(testBackend, model.NewStore(), func(_ *model.Scope, topOne, logits *graph.Node) *graph.Node {
		return maskedSparseCategoricalCrossEntropy([]*graph.Node{topOne}, []*graph.Node{logits})
	})
	defer exec.Finalize()

	for name, nonFinite := range map[string]float32{
		"NaN":          float32(math.NaN()),
		"positive inf": float32(math.Inf(1)),
	} {
		t.Run(name, func(t *testing.T) {
			result := exec.MustCall1([][]int32{{0}}, [][]float32{{2, nonFinite, -3}})
			got := scalar(result)
			if err := result.FinalizeAll(); err != nil {
				t.Fatalf("finalize loss: %v", err)
			}
			if isFinite(got) {
				t.Fatalf("loss with %s logit = %v, want a non-finite value visible to safety gates", name, got)
			}
		})
	}
}

func TestGlobalGradientClippingPublishesFiniteDiagnostics(t *testing.T) {
	store := model.NewStore()
	optimizer := NewAdamWithGlobalNormClip(0.01, GlobalGradientClipNorm)
	exec := model.MustNewExec(testBackend, store, func(scope *model.Scope, x *graph.Node) *graph.Node {
		weight := scope.VariableWithValue("weight", float32(0)).SetTrainable(true)
		prediction := graph.Mul(weight.NodeValue(x.Graph()), x)
		target := graph.MulScalar(x, 1000)
		stepLoss := graph.ReduceAllMean(graph.Square(graph.Sub(prediction, target)))
		optimizer.UpdateGraph(scope, x.Graph(), stepLoss)
		return stepLoss
	})
	defer exec.Finalize()

	loss := exec.MustCall1([]float32{1})
	defer func() { _ = loss.FinalizeAll() }()
	if !isFinite(scalar(loss)) {
		t.Fatalf("loss = %v, want finite", scalar(loss))
	}
	diagnostics, err := ReadTrainingDiagnostics(store)
	if err != nil {
		t.Fatalf("ReadTrainingDiagnostics: %v", err)
	}
	if diagnostics.PreclipGlobalGradientNorm <= GlobalGradientClipNorm {
		t.Fatalf("preclip norm = %v, want > %v", diagnostics.PreclipGlobalGradientNorm, GlobalGradientClipNorm)
	}
	if diagnostics.AppliedGlobalGradientNorm > GlobalGradientClipNorm+1e-5 {
		t.Fatalf("applied norm = %v, want <= %v", diagnostics.AppliedGlobalGradientNorm, GlobalGradientClipNorm)
	}
	if !diagnostics.GradientsFinite || !diagnostics.ParametersFinite {
		t.Fatalf("finite diagnostics = %+v, want true", diagnostics)
	}
	if diagnostics.ParameterNorm <= 0 || diagnostics.UpdateToParameterNorm <= 0 {
		t.Fatalf("norm diagnostics = %+v, want positive", diagnostics)
	}
	assertClose(t, diagnostics.LearningRate, 0.01)
	gradientNorms, err := ReadGradientNorms(store)
	if err != nil {
		t.Fatalf("ReadGradientNorms: %v", err)
	}
	if len(gradientNorms) != 1 || gradientNorms[0].Path != "/weight" {
		t.Fatalf("gradient norms = %+v, want only /weight", gradientNorms)
	}
	if !isFinite(gradientNorms[0].Norm) || gradientNorms[0].Norm <= GlobalGradientClipNorm {
		t.Fatalf("gradient norm = %v, want finite pre-clipping norm > %v", gradientNorms[0].Norm, GlobalGradientClipNorm)
	}
}

func TestGlobalGradientClippingHandlesExtremeFiniteGradients(t *testing.T) {
	store := model.NewStore()
	optimizer := NewAdamWithGlobalNormClip(0.01, GlobalGradientClipNorm).(*clippedAdam)
	exec := model.MustNewExec(testBackend, store, func(scope *model.Scope, gradient *graph.Node) *graph.Node {
		weight := scope.VariableWithValue("weight", []float32{0, 0}).SetTrainable(true)
		// Materialize the weight before the direct gradient update, so it is one
		// of the trainable variables paired with gradient.
		_ = weight.NodeValue(gradient.Graph())
		optimizer.UpdateGraphWithGradients(scope, []*graph.Node{gradient}, gradient.DType())
		return weight.NodeValue(gradient.Graph())
	})
	defer exec.Finalize()

	updated := exec.MustCall1([]float32{math.MaxFloat32, math.MaxFloat32})
	updatedValues := tensors.MustCopyFlatData[float32](updated)
	defer func() { _ = updated.FinalizeAll() }()
	if len(updatedValues) != 2 || !isFinite(updatedValues[0]) || !isFinite(updatedValues[1]) ||
		(updatedValues[0] == 0 && updatedValues[1] == 0) {
		t.Fatalf("extreme finite gradient produced invalid or zero update: %v", updatedValues)
	}

	diagnostics, err := ReadTrainingDiagnostics(store)
	if err != nil {
		t.Fatalf("ReadTrainingDiagnostics: %v", err)
	}
	if !diagnostics.GradientsFinite || !diagnostics.ParametersFinite {
		t.Fatalf("finite extreme gradient was marked non-finite: %+v", diagnostics)
	}
	if !isFinite(diagnostics.PreclipGlobalGradientNorm) || diagnostics.PreclipGlobalGradientNorm < math.MaxFloat32/2 {
		t.Fatalf("preclip norm = %v, want a finite saturated extreme norm", diagnostics.PreclipGlobalGradientNorm)
	}
	if !isFinite(diagnostics.AppliedGlobalGradientNorm) || diagnostics.AppliedGlobalGradientNorm < GlobalGradientClipNorm*.99 || diagnostics.AppliedGlobalGradientNorm > GlobalGradientClipNorm+1e-5 {
		t.Fatalf("applied norm = %v, want approximately %v", diagnostics.AppliedGlobalGradientNorm, GlobalGradientClipNorm)
	}
	gradientNorms, err := ReadGradientNorms(store)
	if err != nil {
		t.Fatalf("ReadGradientNorms: %v", err)
	}
	if len(gradientNorms) != 1 || !isFinite(gradientNorms[0].Norm) || gradientNorms[0].Norm < math.MaxFloat32/2 {
		t.Fatalf("gradient norms = %+v, want one finite saturated extreme norm", gradientNorms)
	}
}

func TestStableGlobalL2DoesNotHideNonFiniteInputs(t *testing.T) {
	exec := model.MustNewExec(testBackend, model.NewStore(), func(_ *model.Scope, values *graph.Node) (*graph.Node, *graph.Node) {
		norm := stableGlobalL2Norm([]*graph.Node{values}, values.DType())
		return norm.value, norm.finite
	})
	defer exec.Finalize()

	for name, nonFinite := range map[string]float32{
		"NaN":          float32(math.NaN()),
		"positive inf": float32(math.Inf(1)),
	} {
		t.Run(name, func(t *testing.T) {
			outputs := exec.MustCall([]float32{1, nonFinite})
			defer finalizeAll(t, outputs)
			if !math.IsInf(float64(scalar(outputs[0])), 1) {
				t.Fatalf("norm for %s input = %v, want +Inf", name, scalar(outputs[0]))
			}
			if outputs[1].Value().(bool) {
				t.Fatalf("finite flag for %s input = true, want false", name)
			}
		})
	}
}

func TestReadGradientNormsReturnsErrorsForNilAndMalformedStore(t *testing.T) {
	if _, err := ReadGradientNorms(nil); err == nil {
		t.Fatal("ReadGradientNorms(nil) unexpectedly succeeded")
	}
	store := model.NewStore()
	store.VariableWithValue("/weight", float32(0)).SetTrainable(true)
	store.VariableWithValue(gradientNormDiagnosticPath("/weight"), []float32{1, 2}).SetTrainable(false)
	if _, err := ReadGradientNorms(store); err == nil {
		t.Fatal("ReadGradientNorms accepted a non-scalar diagnostic")
	}
}

func TestTrainStepUpdatesNonBiasWeightsAndCheckpointRestoresAdamAndPredictions(t *testing.T) {
	config := Config{
		Policy:       policy.Config{NumSolutions: 4, NumActions: 16},
		LearningRate: 0.01,
		Seed:         42,
	}
	session, err := New(config, testBackend)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer session.Finalize()

	// Eval builds the policy variables without taking an optimisation step.
	metrics, err := session.Trainer.EvalStep(testBatch())
	if err != nil {
		t.Fatalf("EvalStep: %v", err)
	}
	finalizeAll(t, metrics)
	bias := session.Store.GetVariable("/wordle_policy/base_logits/dense/biases")
	before := tensors.MustCopyFlatData[float32](bias.MustValue())
	weights := session.Store.GetVariable("/wordle_policy/base_logits/dense/weights")
	weightsBefore := tensors.MustCopyFlatData[float32](weights.MustValue())

	metrics, err = session.Trainer.TrainStep(testBatch())
	if err != nil {
		t.Fatalf("TrainStep: %v", err)
	}
	if !isFinite(scalar(metrics[0])) {
		t.Fatalf("loss is not finite: %v", scalar(metrics[0]))
	}
	finalizeAll(t, metrics)
	if got := session.Trainer.GlobalStep(); got != 1 {
		t.Fatalf("global step = %d, want 1", got)
	}
	after := tensors.MustCopyFlatData[float32](bias.MustValue())
	if equalFloat32Slices(before, after) {
		t.Fatal("base logits bias did not change after TrainStep")
	}
	weightsAfterFirst := tensors.MustCopyFlatData[float32](weights.MustValue())
	if equalFloat32Slices(weightsBefore, weightsAfterFirst) {
		t.Fatal("base logits weights did not change after TrainStep")
	}
	diagnostics, err := session.Diagnostics()
	if err != nil {
		t.Fatalf("Diagnostics: %v", err)
	}
	if !diagnostics.GradientsFinite || !diagnostics.ParametersFinite ||
		diagnostics.AppliedGlobalGradientNorm > GlobalGradientClipNorm+1e-5 {
		t.Fatalf("unexpected diagnostics after train step: %+v", diagnostics)
	}
	gradientNormsBeforeCheckpoint := assertGradientNorms(t, session.Store)

	beforeCheckpointLogits, beforeCheckpointBeta := mustPredict(t, session)
	dir := t.TempDir()
	handler, err := NewCheckpoint(session.Store, dir)
	if err != nil {
		t.Fatalf("NewCheckpoint: %v", err)
	}
	if err := handler.Save(); err != nil {
		t.Fatalf("checkpoint Save: %v", err)
	}

	// Continue uninterrupted once; a restored session must take exactly the same
	// next Adam update from the checkpointed moments.
	metrics, err = session.Trainer.TrainStep(testBatch())
	if err != nil {
		t.Fatalf("uninterrupted second TrainStep: %v", err)
	}
	finalizeAll(t, metrics)
	weightsAfterSecond := tensors.MustCopyFlatData[float32](weights.MustValue())
	gradientNormsAfterSecond := assertGradientNorms(t, session.Store)

	restored, err := New(config, testBackend)
	if err != nil {
		t.Fatalf("new restored session: %v", err)
	}
	defer restored.Finalize()
	if _, err := NewCheckpoint(restored.Store, dir); err != nil {
		t.Fatalf("resume checkpoint: %v", err)
	}
	restoredLogits, restoredBeta := mustPredict(t, restored)
	if !equalFloat32Slices(beforeCheckpointLogits, restoredLogits) {
		t.Fatal("checkpointed logits do not reproduce predictions")
	}
	if !equalFloat32Slices(beforeCheckpointBeta, restoredBeta) {
		t.Fatal("checkpointed beta does not reproduce predictions")
	}
	if got := restored.Trainer.GlobalStep(); got != 1 {
		t.Fatalf("restored global step = %d, want 1", got)
	}
	restoredDiagnostics, err := restored.Diagnostics()
	if err != nil {
		t.Fatalf("restored Diagnostics: %v", err)
	}
	if restoredDiagnostics != diagnostics {
		t.Fatalf("restored diagnostics = %+v, want %+v", restoredDiagnostics, diagnostics)
	}
	assertNamedNormsEqual(t, assertGradientNorms(t, restored.Store), gradientNormsBeforeCheckpoint)

	metrics, err = restored.Trainer.TrainStep(testBatch())
	if err != nil {
		t.Fatalf("train restored session: %v", err)
	}
	finalizeAll(t, metrics)
	if got := restored.Trainer.GlobalStep(); got != 2 {
		t.Fatalf("resumed global step = %d, want 2", got)
	}
	restoredWeights := restored.Store.GetVariable("/wordle_policy/base_logits/dense/weights")
	if restoredWeights == nil || !equalFloat32Slices(weightsAfterSecond, tensors.MustCopyFlatData[float32](restoredWeights.MustValue())) {
		t.Fatal("resumed Adam update differs from uninterrupted continuation")
	}
	assertAdamMomentEqual(t, session.Store, restored.Store, "/AdamOptimizer/wordle_policy/base_logits/dense/weights_1st_moment")
	assertNamedNormsEqual(t, assertGradientNorms(t, restored.Store), gradientNormsAfterSecond)
}

func TestEvalDoesNotUpdateParameters(t *testing.T) {
	session, err := New(Config{
		Policy:       policy.Config{NumSolutions: 4, NumActions: 16},
		LearningRate: 0.01,
		Seed:         99,
	}, testBackend)
	if err != nil {
		t.Fatal(err)
	}
	defer session.Finalize()

	metrics, err := session.Trainer.EvalStep(testBatch())
	if err != nil {
		t.Fatal(err)
	}
	finalizeAll(t, metrics)
	weights := session.Store.GetVariable("/wordle_policy/base_logits/dense/weights")
	before := tensors.MustCopyFlatData[float32](weights.MustValue())
	metrics, err = session.Trainer.EvalStep(testBatch())
	if err != nil {
		t.Fatal(err)
	}
	finalizeAll(t, metrics)
	after := tensors.MustCopyFlatData[float32](weights.MustValue())
	if !equalFloat32Slices(before, after) {
		t.Fatal("EvalStep updated model weights")
	}
}

func TestInferenceReturnsRawAndHardMaskedLogitsWithoutUpdatingStore(t *testing.T) {
	session, err := New(Config{
		Policy:       policy.Config{NumSolutions: 4, NumActions: 16},
		LearningRate: 0.01,
		Seed:         77,
	}, testBackend)
	if err != nil {
		t.Fatal(err)
	}
	defer session.Finalize()

	// Materialize the inference graph and initialize the Store before taking its
	// immutability snapshot. Lazy initialization is the only permitted first-call
	// Store change; every later inference call must be read-only.
	raw, masked, beta, err := session.PredictDiagnostics(
		[][]float32{{1, 0, 0, 0}},
		zeroMatrix(1, policy.CandidateStatsSize),
		[]int32{0},
		zeroMatrix(1, 16),
		[][]float32{ones(16)},
	)
	if err != nil {
		t.Fatalf("initialize inference: %v", err)
	}
	finalizeAll(t, []*tensors.Tensor{raw, masked, beta})

	// Make the raw preference deterministic: action 1 is an extreme but
	// unavailable logit, while action 0 is the highest legal logit. With every
	// other trainable value zero, the policy output is exactly this bias vector.
	zeroTrainableVariables(t, session.Store)
	bias := session.Store.GetVariable("/wordle_policy/base_logits/dense/biases")
	if bias == nil {
		t.Fatal("base logits bias is missing")
	}
	biasValues := make([]float32, 16)
	biasValues[0] = 7
	biasValues[1] = math.MaxFloat32
	bias.MustSetValue(tensors.FromFlatDataAndDimensions(biasValues, 16))

	availability := [][]float32{{1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}}
	before := snapshotStore(t, session.Store)
	raw, masked, beta, err = session.PredictDiagnostics(
		[][]float32{{1, 0, 0, 0}},
		zeroMatrix(1, policy.CandidateStatsSize),
		[]int32{0},
		zeroMatrix(1, 16),
		availability,
	)
	if err != nil {
		t.Fatalf("PredictDiagnostics: %v", err)
	}
	rawValues := tensors.MustCopyFlatData[float32](raw)
	maskedValues := tensors.MustCopyFlatData[float32](masked)
	if len(tensors.MustCopyFlatData[float32](beta)) != 1 {
		t.Fatal("beta output has the wrong shape")
	}
	finalizeAll(t, []*tensors.Tensor{raw, masked, beta})

	if rawValues[1] != math.MaxFloat32 || hostArgMax(rawValues) != 1 {
		t.Fatalf("raw logits did not retain extreme unavailable action: %v", rawValues)
	}
	if !math.IsInf(float64(maskedValues[1]), -1) {
		t.Fatalf("masked unavailable action = %v, want -Inf", maskedValues[1])
	}
	if selected := hostArgMax(maskedValues); selected != 0 {
		t.Fatalf("hard-masked argmax = %d, want legal action 0; logits=%v", selected, maskedValues)
	}
	assertStoreSnapshot(t, session.Store, before)
}

func TestSessionFinalizeReleasesOwnedResourcesIdempotently(t *testing.T) {
	store := model.NewStore()
	store.VariableWithValue("/owned", float32(1))
	var trainer train.Trainer
	session := &Session{Store: store, Trainer: &trainer}

	session.Finalize()
	if session.Policy != nil || session.Store != nil || session.Trainer != nil || session.Inference != nil {
		t.Fatalf("Finalize retained owned fields: %+v", session)
	}
	count := 0
	for range store.IterVariables() {
		count++
	}
	if count != 0 {
		t.Fatalf("finalized Store retains %d variables", count)
	}

	// A second call must be a no-op rather than touching finalized GoMLX state.
	session.Finalize()
}

func TestConfigValidation(t *testing.T) {
	base := Config{Policy: policy.Config{NumSolutions: 4, NumActions: 16}, LearningRate: 0.01, Seed: 1}
	for name, config := range map[string]Config{
		"zero learning rate": {Policy: base.Policy, Seed: base.Seed},
		"NaN learning rate":  {Policy: base.Policy, LearningRate: math.NaN(), Seed: base.Seed},
		"infinite learning rate": {
			Policy: base.Policy, LearningRate: math.Inf(1), Seed: base.Seed,
		},
		"zero seed":       {Policy: base.Policy, LearningRate: base.LearningRate},
		"too few actions": {Policy: policy.Config{NumSolutions: 4, NumActions: 15}, LearningRate: base.LearningRate, Seed: base.Seed},
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := New(config, testBackend); err == nil {
				t.Fatal("New unexpectedly succeeded")
			}
		})
	}
}

func testBatch() train.Batch {
	return train.Batch{
		Inputs: []*tensors.Tensor{
			tensors.FromFlatDataAndDimensions([]float32{1, 0, 0, 0, 0, 1, 0, 0}, 2, 4),
			tensors.FromFlatDataAndDimensions(make([]float32, 2*policy.CandidateStatsSize), 2, policy.CandidateStatsSize),
			tensors.FromFlatDataAndDimensions([]int32{0, 1}, 2),
			tensors.FromFlatDataAndDimensions(make([]float32, 2*16), 2, 16),
			tensors.FromFlatDataAndDimensions(ones(2*16), 2, 16),
		},
		Labels: []*tensors.Tensor{
			tensors.FromFlatDataAndDimensions([]int32{1, 2}, 2, 1),
			tensors.FromFlatDataAndDimensions([]int32{
				1, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
				2, 0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
			}, 2, 16),
		},
	}
}

func mustPredict(t *testing.T, session *Session) ([]float32, []float32) {
	t.Helper()
	logits, beta, err := session.Predict(
		[][]float32{{1, 0, 0, 0}, {0, 1, 0, 0}},
		zeroMatrix(2, policy.CandidateStatsSize),
		[]int32{0, 1},
		zeroMatrix(2, 16),
		[][]float32{ones(16), ones(16)},
	)
	if err != nil {
		t.Fatalf("Predict: %v", err)
	}
	defer func() {
		_ = logits.FinalizeAll()
		_ = beta.FinalizeAll()
	}()
	return tensors.MustCopyFlatData[float32](logits), tensors.MustCopyFlatData[float32](beta)
}

func assertAdamMomentEqual(t *testing.T, first, second *model.Store, path string) {
	t.Helper()
	firstVar := first.GetVariable(path)
	secondVar := second.GetVariable(path)
	if firstVar == nil || secondVar == nil {
		t.Fatalf("Adam moment %q is missing after continuation", path)
	}
	if !equalFloat32Slices(
		tensors.MustCopyFlatData[float32](firstVar.MustValue()),
		tensors.MustCopyFlatData[float32](secondVar.MustValue()),
	) {
		t.Fatalf("Adam moment %q differs after restored continuation", path)
	}
}

func assertGradientNorms(t *testing.T, store *model.Store) []NamedNorm {
	t.Helper()
	wantPaths := make([]string, 0)
	for variable := range store.IterVariables() {
		if variable.Trainable {
			wantPaths = append(wantPaths, variable.Path())
		}
	}
	sort.Strings(wantPaths)
	got, err := ReadGradientNorms(store)
	if err != nil {
		t.Fatalf("ReadGradientNorms: %v", err)
	}
	if len(got) != len(wantPaths) {
		t.Fatalf("gradient norm count = %d, want %d; got=%+v", len(got), len(wantPaths), got)
	}
	for index, norm := range got {
		if norm.Path != wantPaths[index] {
			t.Fatalf("gradient norm path %d = %q, want %q", index, norm.Path, wantPaths[index])
		}
		if !isFinite(norm.Norm) || norm.Norm < 0 {
			t.Fatalf("gradient norm %q = %v, want finite non-negative", norm.Path, norm.Norm)
		}
		diagnostic := store.GetVariable(gradientNormDiagnosticPath(norm.Path))
		if diagnostic == nil || diagnostic.Trainable {
			t.Fatalf("gradient norm diagnostic for %q is missing or trainable", norm.Path)
		}
	}
	return got
}

func assertNamedNormsEqual(t *testing.T, got, want []NamedNorm) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("gradient norm length = %d, want %d", len(got), len(want))
	}
	for index := range want {
		if got[index] != want[index] {
			t.Fatalf("gradient norm %d = %+v, want %+v", index, got[index], want[index])
		}
	}
}

func ones(size int) []float32 {
	values := make([]float32, size)
	for i := range values {
		values[i] = 1
	}
	return values
}

func zeroMatrix(rows, columns int) [][]float32 {
	values := make([][]float32, rows)
	for row := range values {
		values[row] = make([]float32, columns)
	}
	return values
}

func zeroTrainableVariables(t *testing.T, store *model.Store) {
	t.Helper()
	for variable := range store.IterVariables() {
		if !variable.Trainable {
			continue
		}
		shape := variable.Shape()
		variable.MustSetValue(tensors.FromFlatDataAndDimensions(make([]float32, shape.Size()), shape.Dimensions...))
	}
}

func snapshotStore(t *testing.T, store *model.Store) map[string][]byte {
	t.Helper()
	snapshot := make(map[string][]byte)
	for variable := range store.IterVariables() {
		value := variable.MustValue()
		var copied []byte
		if err := value.ConstBytes(func(data []byte) {
			copied = append([]byte(nil), data...)
		}); err != nil {
			t.Fatalf("copy %s: %v", variable.Path(), err)
		}
		snapshot[variable.Path()] = copied
	}
	return snapshot
}

func assertStoreSnapshot(t *testing.T, store *model.Store, want map[string][]byte) {
	t.Helper()
	seen := 0
	for variable := range store.IterVariables() {
		seen++
		before, ok := want[variable.Path()]
		if !ok {
			t.Fatalf("inference added Store variable %q", variable.Path())
		}
		var after []byte
		if err := variable.MustValue().ConstBytes(func(data []byte) {
			after = append([]byte(nil), data...)
		}); err != nil {
			t.Fatalf("copy %s after inference: %v", variable.Path(), err)
		}
		if !bytes.Equal(before, after) {
			t.Fatalf("inference changed Store variable %q", variable.Path())
		}
	}
	if seen != len(want) {
		t.Fatalf("Store variable count after inference = %d, want %d", seen, len(want))
	}
}

func hostArgMax(values []float32) int {
	if len(values) == 0 {
		return -1
	}
	best := 0
	for index := 1; index < len(values); index++ {
		if values[index] > values[best] {
			best = index
		}
	}
	return best
}

func scalar(tensor *tensors.Tensor) float32 {
	return tensor.Value().(float32)
}

func assertClose(t *testing.T, got, want float32) {
	t.Helper()
	if math.Abs(float64(got-want)) > 1e-6 {
		t.Fatalf("got %v, want %v", got, want)
	}
}

func isFinite(value float32) bool {
	return !math.IsNaN(float64(value)) && !math.IsInf(float64(value), 0)
}

func equalFloat32Slices(a, b []float32) bool {
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

func finalizeAll(t *testing.T, tensors []*tensors.Tensor) {
	t.Helper()
	for _, tensor := range tensors {
		if err := tensor.FinalizeAll(); err != nil {
			t.Fatalf("finalize metric: %v", err)
		}
	}
}

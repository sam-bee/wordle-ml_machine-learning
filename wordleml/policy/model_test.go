package policy

import (
	"fmt"
	"math"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	_ "github.com/gomlx/gomlx/backends/default"
	"github.com/gomlx/gomlx/core/tensors"
	gomlxmodel "github.com/gomlx/gomlx/ml/model"
)

var testBackend = compute.MustNew()

func TestForwardOutputShapeAndFiniteFP32(t *testing.T) {
	model, store, exec := newTestExec(t, Config{NumSolutions: 7, NumActions: 11})
	defer exec.Finalize()

	output := runModel(exec, model.config, 3)
	if err := output.Shape().Check(dtypes.Float32, 3, 11); err != nil {
		t.Fatalf("unexpected output shape: %v", err)
	}
	for rowIndex, row := range output.Value().([][]float32) {
		for actionIndex, value := range row {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				t.Fatalf("logit [%d,%d] is not finite: %v", rowIndex, actionIndex, value)
			}
		}
	}

	if got := TrainableParameterCount(store.RootScope()); got == 0 {
		t.Fatal("forward graph did not materialize trainable parameters")
	}
}

func TestConfigurableVocabularySizes(t *testing.T) {
	testCases := []Config{
		{NumSolutions: 3, NumActions: 4},
		{NumSolutions: 13, NumActions: 9},
	}

	for _, config := range testCases {
		t.Run(configName(config), func(t *testing.T) {
			model, _, exec := newTestExec(t, config)
			defer exec.Finalize()

			output := runModel(exec, config, 2)
			if err := output.Shape().Check(dtypes.Float32, 2, config.NumActions); err != nil {
				t.Fatalf("unexpected output shape: %v", err)
			}
			if model.Config() != config {
				t.Fatalf("Config() = %+v, want %+v", model.Config(), config)
			}
		})
	}
}

func TestCandidateMaskIsMeanPooled(t *testing.T) {
	config := Config{NumSolutions: 3, NumActions: 4}
	_, _, exec := newTestExec(t, config)
	defer exec.Finalize()

	stats := zeroMatrix(2, CandidateStatsSize)
	actionMask := zeroMatrix(2, config.NumActions)
	output := exec.MustCall1(
		[][]float32{{1, 0, 0}, {7, 0, 0}},
		stats,
		[]int32{2, 2},
		actionMask,
	).Value().([][]float32)

	for action := range config.NumActions {
		if delta := math.Abs(float64(output[0][action] - output[1][action])); delta > 1e-5 {
			t.Fatalf("scaled candidate masks produced different logit %d: %v versus %v", action, output[0][action], output[1][action])
		}
	}
}

func TestCandidateBonusOnlyChangesSelectedLogits(t *testing.T) {
	config := Config{NumSolutions: 3, NumActions: 4}
	_, store, exec := newTestExec(t, config)
	defer exec.Finalize()

	// The first call builds the graph and initializes every variable.
	runModel(exec, config, 1)
	zeroTrainableVariables(store)
	store.GetVariable("/wordle_policy/base_logits/dense/biases").MustSetValue(
		tensors.FromFlatDataAndDimensions([]float32{10, 20, 30, 40}, config.NumActions),
	)
	store.GetVariable("/wordle_policy/candidate_bonus/dense/biases").MustSetValue(
		tensors.FromFlatDataAndDimensions([]float32{3}, 1),
	)

	candidateMask := [][]float32{{1, 0, 0}}
	stats := zeroMatrix(1, CandidateStatsSize)
	turn := []int32{0}
	withoutBonus := exec.MustCall1(candidateMask, stats, turn, [][]float32{{0, 0, 0, 0}}).Value().([][]float32)[0]
	withBonus := exec.MustCall1(candidateMask, stats, turn, [][]float32{{0, 1, 0, 1}}).Value().([][]float32)[0]

	wantWithoutBonus := []float32{10, 20, 30, 40}
	wantWithBonus := []float32{10, 23, 30, 43}
	assertFloat32SliceEqual(t, withoutBonus, wantWithoutBonus)
	assertFloat32SliceEqual(t, withBonus, wantWithBonus)

	// Actions 0 and 2 are not candidate solutions. Their finite base logits remain
	// untouched, rather than being suppressed as they would be by a legality mask.
	if withBonus[0] != 10 || withBonus[2] != 30 {
		t.Fatalf("non-candidate logits were masked: got %v", withBonus)
	}
}

func TestKnownArchitectureParameterCount(t *testing.T) {
	testCases := []struct {
		name           string
		config         Config
		wantParameters int
	}{
		{
			name:           "architecture brief reference",
			config:         Config{NumSolutions: 2315, NumActions: 4800},
			wantParameters: 1_056_993,
		},
		{
			name:           "repository vocabularies",
			config:         Config{NumSolutions: 2309, NumActions: 4739},
			wantParameters: 1_046_596,
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			_, store, exec := newTestExec(t, testCase.config)
			defer exec.Finalize()

			runModel(exec, testCase.config, 1)
			root := store.RootScope()
			if got := TrainableParameterCount(root); got != testCase.wantParameters {
				t.Fatalf("trainable parameter count = %d, want %d", got, testCase.wantParameters)
			}
			if got, want := TrainableParameterBytes(root), int64(testCase.wantParameters*4); got != want {
				t.Fatalf("FP32 parameter bytes = %d, want %d", got, want)
			}
		})
	}
}

func newTestExec(t *testing.T, config Config) (*Model, *gomlxmodel.Store, *gomlxmodel.Exec) {
	t.Helper()
	policyModel, err := New(config)
	if err != nil {
		t.Fatalf("New(%+v): %v", config, err)
	}
	store := gomlxmodel.NewStore()
	exec := gomlxmodel.MustNewExec(testBackend, store, policyModel.Forward)
	return policyModel, store, exec
}

func runModel(exec *gomlxmodel.Exec, config Config, batchSize int) *tensors.Tensor {
	candidateMask := zeroMatrix(batchSize, config.NumSolutions)
	for row := range candidateMask {
		candidateMask[row][row%config.NumSolutions] = 1
	}
	return exec.MustCall1(
		candidateMask,
		zeroMatrix(batchSize, CandidateStatsSize),
		make([]int32, batchSize),
		zeroMatrix(batchSize, config.NumActions),
	)
}

func zeroMatrix(rows, columns int) [][]float32 {
	values := make([][]float32, rows)
	for row := range values {
		values[row] = make([]float32, columns)
	}
	return values
}

func zeroTrainableVariables(store *gomlxmodel.Store) {
	for variable := range store.IterVariables() {
		if variable.Trainable {
			variable.MustSetValue(tensors.FromShape(variable.Shape()))
		}
	}
}

func assertFloat32SliceEqual(t *testing.T, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("got %d values, want %d", len(got), len(want))
	}
	for index := range want {
		if got[index] != want[index] {
			t.Fatalf("value %d = %v, want %v (all got: %v)", index, got[index], want[index], got)
		}
	}
}

func configName(config Config) string {
	return fmt.Sprintf("S%d_A%d", config.NumSolutions, config.NumActions)
}

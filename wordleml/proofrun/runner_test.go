package proofrun

import (
	"encoding/json"
	"errors"
	"math"
	"path/filepath"
	"slices"
	"testing"
	"time"

	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/sam-bee/wordle-ml_machine-learning/imitationdata"
	"github.com/sam-bee/wordle-ml_machine-learning/runmetadata"
	"github.com/sam-bee/wordle-ml_machine-learning/runstate"
	"github.com/sam-bee/wordle-ml_machine-learning/vocabulary"
)

func TestValidateFiniteTrainableParameters(t *testing.T) {
	finiteStore := model.NewStore()
	finiteStore.VariableWithValue("/weights", []float32{1, -2}).SetTrainable(true)
	if err := validateFiniteTrainableParameters(finiteStore); err != nil {
		t.Fatalf("finite trainable parameters: %v", err)
	}

	for name, value := range map[string]float32{"nan": float32(math.NaN()), "infinity": float32(math.Inf(1))} {
		t.Run(name, func(t *testing.T) {
			store := model.NewStore()
			store.VariableWithValue("/weights", []float32{1, value}).SetTrainable(true)
			if err := validateFiniteTrainableParameters(store); err == nil {
				t.Fatal("non-finite trainable parameter was accepted")
			}
		})
	}
}

func TestProductionCompletionIsTenThousandUpdatesWithoutFullProofGate(t *testing.T) {
	config, err := ConfigFor(Production)
	if err != nil {
		t.Fatal(err)
	}
	passing := productionResultForTest(config)
	if !productionRunPassed(config, passing) {
		t.Fatal("complete finite production result did not pass")
	}
	for name, mutate := range map[string]func(*Result){
		"old full target": func(result *Result) { result.GlobalUpdate = 2000 },
		"missing snapshot": func(result *Result) {
			result.ValidationSnapshots = result.ValidationSnapshots[:len(result.ValidationSnapshots)-1]
		},
		"off cadence snapshot": func(result *Result) { result.ValidationSnapshots[1].Update++ },
		"non-finite train":     func(result *Result) { result.FinalTraining.Loss = math.Inf(1) },
		"best after target":    func(result *Result) { result.BestValidationStep = config.TargetUpdates + 1 },
		"missing safety":       func(result *Result) { result.ProductionSafety = nil },
		"partial safety":       func(result *Result) { result.ProductionSafety.UpdatesChecked-- },
	} {
		t.Run(name, func(t *testing.T) {
			candidate := passing
			candidate.ValidationSnapshots = append([]ValidationSnapshot(nil), passing.ValidationSnapshots...)
			candidate.ProductionSafety = cloneProductionSafety(passing.ProductionSafety)
			mutate(&candidate)
			if productionRunPassed(config, candidate) {
				t.Fatal("incomplete production result passed")
			}
		})
	}
}

func productionResultForTest(config Config) Result {
	snapshots := make([]ValidationSnapshot, 0, config.TargetUpdates/config.ValidationEvery+1)
	for update := int64(0); update <= config.TargetUpdates; update += config.ValidationEvery {
		snapshots = append(snapshots, ValidationSnapshot{Update: update, Metrics: Metrics{Loss: 1, Top1: .1, Top5: .2, Top16: .3}})
	}
	metrics := Metrics{Loss: 1, Top1: .1, Top5: .2, Top16: .3}
	return Result{
		Stage:                    Production,
		GlobalUpdate:             config.TargetUpdates,
		InitialTraining:          metrics,
		FinalTraining:            metrics,
		InitialValidation:        metrics,
		FinalValidation:          metrics,
		BestValidation:           metrics,
		BestValidationStep:       0,
		ProductionSafety:         &ProductionSafety{LossFinite: true, GradientsFinite: true, ParametersFinite: true, UpdatesChecked: config.TargetUpdates},
		ValidationSnapshots:      snapshots,
		InitialValidationDetails: ValidationSnapshot{Update: 0, Metrics: metrics},
		FinalValidationDetails:   ValidationSnapshot{Update: config.TargetUpdates, Metrics: metrics},
		BestValidationDetails:    ValidationSnapshot{Update: 0, Metrics: metrics},
	}
}

func TestProductionPriorResultRecoversOnlyMatchingCheckpointProgress(t *testing.T) {
	layout, err := runstate.Create(t.TempDir(), "production-resume")
	if err != nil {
		t.Fatal(err)
	}
	stale := productionProgressForTest(100, 100)
	matching := productionProgressForTest(200, 100)
	if err := layout.WriteFinalMetricsJSON(stale); err != nil {
		t.Fatal(err)
	}
	if err := layout.WriteCheckpointProgress(matching); err != nil {
		t.Fatal(err)
	}
	state := runstate.State{GlobalUpdate: 200, ShuffleSeed: Seed, BestValidation: &runstate.BestValidation{Update: 100, Value: 1}}
	got, recovered, err := loadPriorResult(layout, state, true)
	if err != nil {
		t.Fatalf("loadPriorResult: %v", err)
	}
	if !recovered || got.GlobalUpdate != state.GlobalUpdate {
		t.Fatalf("result=%+v recovered=%t", got, recovered)
	}

	if err := layout.WriteCheckpointProgress(productionProgressForTest(300, 100)); err != nil {
		t.Fatal(err)
	}
	if _, _, err := loadPriorResult(layout, state, true); err == nil {
		t.Fatal("ahead checkpoint progress was accepted")
	}
	wrongBest := productionProgressForTest(200, 100)
	wrongBest.BestValidation.Loss = 2
	if err := layout.WriteCheckpointProgress(wrongBest); err != nil {
		t.Fatal(err)
	}
	if _, _, err := loadPriorResult(layout, state, true); err == nil {
		t.Fatal("checkpoint progress with a mismatched best state was accepted")
	}
}

func TestProductionFinalMetricsMustMatchCheckpointBestAndSafety(t *testing.T) {
	layout, err := runstate.Create(t.TempDir(), "production-final-identity")
	if err != nil {
		t.Fatal(err)
	}
	state := runstate.State{GlobalUpdate: 200, ShuffleSeed: Seed, BestValidation: &runstate.BestValidation{Update: 100, Value: 1}}
	wrongBest := productionProgressForTest(200, 100)
	wrongBest.BestValidation.Loss = 2
	if err := layout.WriteFinalMetricsJSON(wrongBest); err != nil {
		t.Fatal(err)
	}
	if _, _, err := loadPriorResult(layout, state, true); err == nil {
		t.Fatal("matching-update final metrics with a mismatched best state were accepted")
	}

	wrongSafety := productionProgressForTest(200, 100)
	wrongSafety.ProductionSafety.UpdatesChecked = 199
	if err := layout.WriteFinalMetricsJSON(wrongSafety); err != nil {
		t.Fatal(err)
	}
	if _, _, err := loadPriorResult(layout, state, true); err == nil {
		t.Fatal("matching-update final metrics with incomplete safety evidence were accepted")
	}
}

func productionProgressForTest(update, bestUpdate int64) Result {
	return Result{
		Stage:              Production,
		GlobalUpdate:       update,
		InitialValidation:  Metrics{Loss: 1},
		BestValidation:     Metrics{Loss: 1},
		BestValidationStep: bestUpdate,
		ProductionSafety:   &ProductionSafety{LossFinite: true, GradientsFinite: true, ParametersFinite: true, UpdatesChecked: update},
	}
}

func TestProductionBootstrapAndCheckpointStateMatchers(t *testing.T) {
	production, err := ConfigFor(Production)
	if err != nil {
		t.Fatal(err)
	}
	if !isProductionBootstrap(production, runstate.ErrStateNotFound) {
		t.Fatal("production bootstrap was not accepted without initial state")
	}
	full, err := ConfigFor(Full)
	if err != nil {
		t.Fatal(err)
	}
	if isProductionBootstrap(full, runstate.ErrStateNotFound) || isProductionBootstrap(production, errors.New("other")) {
		t.Fatal("non-production or unrelated state error was accepted as bootstrap")
	}
	latest := runstate.State{GlobalUpdate: 200, ShuffleSeed: Seed, BestValidation: &runstate.BestValidation{Update: 100, Value: 1.25}}
	best := runstate.State{GlobalUpdate: 100, ShuffleSeed: Seed, BestValidation: &runstate.BestValidation{Update: 100, Value: 1.25}}
	if !bestCheckpointMatchesLatestState(best, latest) {
		t.Fatal("matching best checkpoint was rejected")
	}
	best.BestValidation.Value++
	if bestCheckpointMatchesLatestState(best, latest) {
		t.Fatal("mismatched best checkpoint was accepted")
	}
}

func TestManifestIdentityComparesEffectiveConfigSemantically(t *testing.T) {
	collected := time.Date(2026, 8, 8, 12, 0, 0, 0, time.UTC)
	stored := runmetadata.Manifest{
		SchemaVersion:   runmetadata.SchemaVersion,
		CollectedAt:     collected,
		EffectiveConfig: json.RawMessage("{\n  \"seed\": 20260808,\n  \"batch_size\": 128\n}"),
	}
	current := stored
	current.CollectedAt = collected.Add(time.Hour)
	current.EffectiveConfig = json.RawMessage(`{"batch_size":128,"seed":20260808}`)

	same, err := manifestsHaveSameIdentity(stored, current)
	if err != nil {
		t.Fatalf("manifestsHaveSameIdentity: %v", err)
	}
	if !same {
		t.Fatal("semantically identical effective configs differed by formatting or key order")
	}

	current.Seed = 1
	same, err = manifestsHaveSameIdentity(stored, current)
	if err != nil {
		t.Fatalf("manifestsHaveSameIdentity after mutation: %v", err)
	}
	if same {
		t.Fatal("manifest identity accepted a changed seed")
	}
}

func TestTrainingMetricsUseEMALossAndTopKOffsets(t *testing.T) {
	values := []*tensors.Tensor{
		tensors.FromScalar(float32(9)), // per-batch loss, intentionally ignored
		tensors.FromScalar(float32(8)), // EMA loss
		tensors.FromScalar(float32(1)),
		tensors.FromScalar(float32(2)),
		tensors.FromScalar(float32(3)),
	}
	metrics, err := trainingMetricsFromTensors(values)
	if err != nil {
		t.Fatalf("trainingMetricsFromTensors: %v", err)
	}
	if metrics != (Metrics{Loss: 8, Top1: 1, Top5: 2, Top16: 3}) {
		t.Fatalf("metrics = %#v", metrics)
	}
}

func TestResumedSamplerMustMatchCheckpointPeek(t *testing.T) {
	dataDir := filepath.Join("..", "..", "data")
	vocabulary, err := vocabulary.Load(dataDir)
	if err != nil {
		t.Fatal(err)
	}
	trainData, err := imitationdata.Load(vocabulary, filepath.Join(dataDir, "imitation"), imitationdata.Train)
	if err != nil {
		t.Fatal(err)
	}
	miniData, err := imitationdata.Load(vocabulary, filepath.Join(dataDir, "imitation"), imitationdata.Mini)
	if err != nil {
		t.Fatal(err)
	}
	opening, found, err := trainData.FindOpening()
	if err != nil || !found {
		t.Fatalf("FindOpening = found %t, err %v", found, err)
	}
	sampler, err := imitationdata.NewTrainingSampler(miniData, opening, 128, Seed, imitationdata.Cursor{})
	if err != nil {
		t.Fatal(err)
	}
	state := runstate.State{ShuffleSeed: Seed, NextRecordIDs: sampler.Peek()}
	if err := validateSamplerPeek(state, sampler); err != nil {
		t.Fatalf("matching sampler peek: %v", err)
	}
	state.NextRecordIDs[0]++
	if err := validateSamplerPeek(state, sampler); err == nil {
		t.Fatal("mismatched sampler peek was accepted")
	}
}

func TestProductionResumedSamplerMustMatchGlobalUpdate(t *testing.T) {
	dataDir := filepath.Join("..", "..", "data")
	vocabulary, err := vocabulary.Load(dataDir)
	if err != nil {
		t.Fatal(err)
	}
	trainData, err := imitationdata.Load(vocabulary, filepath.Join(dataDir, "imitation"), imitationdata.Train)
	if err != nil {
		t.Fatal(err)
	}
	opening, found, err := trainData.FindOpening()
	if err != nil || !found {
		t.Fatalf("FindOpening = found %t, err %v", found, err)
	}
	config, err := ConfigFor(Production)
	if err != nil {
		t.Fatal(err)
	}
	reference, err := imitationdata.NewTrainingSampler(trainData, opening, config.BatchSize, config.Seed, imitationdata.Cursor{})
	if err != nil {
		t.Fatal(err)
	}
	const update = int64(100)
	if err := reference.AdvanceBatches(int(update)); err != nil {
		t.Fatal(err)
	}
	cursor := reference.Cursor()
	restored, err := imitationdata.NewTrainingSampler(trainData, opening, config.BatchSize, config.Seed, cursor)
	if err != nil {
		t.Fatal(err)
	}
	state := runstate.State{GlobalUpdate: update, ShuffleSeed: Seed, DatasetEpoch: cursor.Epoch, ShuffledCursor: cursor.Offset, NextRecordIDs: restored.Peek()}
	if err := validateProductionSamplerResumeState(state, restored, trainData, opening, config); err != nil {
		t.Fatalf("matching production sampler: %v", err)
	}

	// This cursor and its peek agree with each other, but they describe update
	// zero rather than the global update claimed by the checkpoint.
	corrupt, err := imitationdata.NewTrainingSampler(trainData, opening, config.BatchSize, config.Seed, imitationdata.Cursor{})
	if err != nil {
		t.Fatal(err)
	}
	badCursor := corrupt.Cursor()
	state.DatasetEpoch, state.ShuffledCursor, state.NextRecordIDs = badCursor.Epoch, badCursor.Offset, corrupt.Peek()
	if err := validateSamplerPeek(state, corrupt); err != nil {
		t.Fatalf("corrupt sampler was not self-consistent: %v", err)
	}
	if err := validateProductionSamplerResumeState(state, corrupt, trainData, opening, config); err == nil {
		t.Fatal("self-consistent cursor unrelated to GlobalUpdate was accepted")
	}
}

func TestMiniResumeProofFollowsUninterruptedSampler(t *testing.T) {
	dataDir := filepath.Join("..", "..", "data")
	vocabulary, err := vocabulary.Load(dataDir)
	if err != nil {
		t.Fatal(err)
	}
	trainData, err := imitationdata.Load(vocabulary, filepath.Join(dataDir, "imitation"), imitationdata.Train)
	if err != nil {
		t.Fatal(err)
	}
	miniData, err := imitationdata.Load(vocabulary, filepath.Join(dataDir, "imitation"), imitationdata.Mini)
	if err != nil {
		t.Fatal(err)
	}
	opening, found, err := trainData.FindOpening()
	if err != nil || !found {
		t.Fatalf("FindOpening = found %t, err %v", found, err)
	}
	checkpointSampler, err := imitationdata.NewTrainingSampler(miniData, opening, 128, Seed, imitationdata.Cursor{})
	if err != nil {
		t.Fatal(err)
	}
	if err := checkpointSampler.AdvanceBatches(500); err != nil {
		t.Fatal(err)
	}
	state := runstate.State{GlobalUpdate: 500, ShuffleSeed: Seed, NextRecordIDs: checkpointSampler.Peek()}
	uninterrupted, err := imitationdata.NewTrainingSampler(miniData, opening, 128, Seed, imitationdata.Cursor{})
	if err != nil {
		t.Fatal(err)
	}
	proof, err := resumeProofForStop(state, uninterrupted, 500)
	if err != nil {
		t.Fatalf("resumeProofForStop: %v", err)
	}
	if !slices.Equal(proof.CheckpointNextRecordIDs, proof.UninterruptedReferenceNextRecordIDs) {
		t.Fatalf("checkpoint IDs %v differ from uninterrupted IDs %v", proof.CheckpointNextRecordIDs, proof.UninterruptedReferenceNextRecordIDs)
	}
	resumed, err := resumeProofForResume(state, proof)
	if err != nil {
		t.Fatalf("resumeProofForResume: %v", err)
	}
	if resumed.ResumeFromUpdate != 500 || resumed.FirstResumedScalarUpdate != 0 || resumed.Completed {
		t.Fatalf("initial resumed proof = %#v", resumed)
	}
	// This assignment mirrors the point immediately after successful scalar
	// telemetry publication in Run.
	resumed.FirstResumedScalarUpdate = 510
	resumed.Completed = true
	contents, err := json.Marshal(Result{Stage: Mini, GlobalUpdate: 1000, ResumeProof: resumed})
	if err != nil {
		t.Fatalf("marshal final mini result: %v", err)
	}
	var decoded Result
	if err := json.Unmarshal(contents, &decoded); err != nil {
		t.Fatalf("unmarshal final mini result: %v", err)
	}
	if decoded.ResumeProof == nil || decoded.ResumeProof.ResumeFromUpdate != 500 || decoded.ResumeProof.FirstResumedScalarUpdate != 510 || !decoded.ResumeProof.Completed {
		t.Fatalf("persisted completed resume proof = %#v", decoded.ResumeProof)
	}
}

func TestMiniGateRequiresCompletedResumeProof(t *testing.T) {
	passing := Result{
		Stage:           Mini,
		GlobalUpdate:    1000,
		InitialTraining: Metrics{Loss: 4, Top1: 0.1, Top5: 0.2, Top16: 0.3},
		FinalTraining:   Metrics{Loss: 1.9, Top1: 0.3, Top5: 0.4, Top16: 0.5},
		ResumeProof: &ResumeProof{
			CheckpointNextRecordIDs:             []int{1, 2, 3},
			UninterruptedReferenceNextRecordIDs: []int{1, 2, 3},
			ResumeFromUpdate:                    500,
			FirstResumedScalarUpdate:            510,
			Completed:                           true,
		},
		TelemetryProof: &MiniTelemetryProof{TrainingSteps: []int64{10, 500, 510, 1000}},
	}
	if !miniGatePassed(passing) {
		t.Fatal("complete learning and resume evidence did not pass the mini gate")
	}
	for name, mutate := range map[string]func(*Result){
		"nil proof":         func(result *Result) { result.ResumeProof = nil },
		"not completed":     func(result *Result) { result.ResumeProof.Completed = false },
		"wrong resume step": func(result *Result) { result.ResumeProof.ResumeFromUpdate = 400 },
		"wrong scalar step": func(result *Result) { result.ResumeProof.FirstResumedScalarUpdate = 500 },
		"wrong IDs":         func(result *Result) { result.ResumeProof.CheckpointNextRecordIDs[0]++ },
		"no event proof":    func(result *Result) { result.TelemetryProof = nil },
	} {
		t.Run(name, func(t *testing.T) {
			candidate := passing
			candidate.ResumeProof = cloneResumeProof(passing.ResumeProof)
			mutate(&candidate)
			if miniGatePassed(candidate) {
				t.Fatal("incomplete resume evidence passed the mini gate")
			}
		})
	}
}

func TestFullGateRequiresOptimizationHeldOutTopKAndBroadGroups(t *testing.T) {
	passing := Result{
		GlobalUpdate:           2000,
		InitialTraining:        Metrics{Loss: 8, Top1: .1, Top5: .2, Top16: .3},
		FinalTraining:          Metrics{Loss: 4, Top1: .4, Top5: .5, Top16: .6},
		InitialValidation:      Metrics{Loss: 8, Top1: .1, Top5: .2, Top16: .3},
		BestValidation:         Metrics{Loss: 7.5, Top1: .11, Top5: .21, Top16: .31},
		ValidationImprovements: 2,
		MajorGroupLearning:     MajorGroupLearning{TurnCount: 2, ShortlistCount: 2},
	}
	if !fullGatePassed(passing) {
		t.Fatal("complete full-stage learning evidence did not pass")
	}
	for name, mutate := range map[string]func(*Result){
		"training loss":   func(result *Result) { result.FinalTraining.Loss = result.InitialTraining.Loss },
		"validation loss": func(result *Result) { result.BestValidation.Loss = result.InitialValidation.Loss * .96 },
		"validation top1": func(result *Result) { result.BestValidation.Top1 = result.InitialValidation.Top1 },
		"checkpoints":     func(result *Result) { result.ValidationImprovements = 1 },
		"turn groups":     func(result *Result) { result.MajorGroupLearning.TurnCount = 1 },
		"shortlist groups": func(result *Result) {
			result.MajorGroupLearning.ShortlistCount = 1
		},
	} {
		t.Run(name, func(t *testing.T) {
			candidate := passing
			mutate(&candidate)
			if fullGatePassed(candidate) {
				t.Fatal("incomplete full-stage evidence passed")
			}
		})
	}
}

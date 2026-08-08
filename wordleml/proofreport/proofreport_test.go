package proofreport

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/sam-bee/wordle-ml_machine-learning/tensorboard"
)

func TestWriteValidatesFixtureAndPublishesMarkdown(t *testing.T) {
	runs := t.TempDir()
	makeFixtureRun(t, runs, "overfit", "overfit-1", time.Date(2026, 8, 8, 9, 0, 0, 0, time.UTC))
	makeFixtureRun(t, runs, "mini", "mini-1", time.Date(2026, 8, 8, 10, 0, 0, 0, time.UTC))
	makeFixtureRun(t, runs, "full", "full-1", time.Date(2026, 8, 8, 11, 0, 0, 0, time.UTC))
	output := filepath.Join(t.TempDir(), "proof.md")
	report, err := Write(Options{RunsDir: runs, OverfitRunID: "overfit-1", MiniRunID: "mini-1", FullRunID: "full-1", OutputPath: output})
	if err != nil {
		t.Fatal(err)
	}
	if len(report.Stages) != 4 || report.Stages[3].Checkpoint != "best" {
		t.Fatalf("proof rows/checkpoint = %#v", report.Stages)
	}
	contents, err := os.ReadFile(output)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(contents), "initial and best 100-game") || !strings.Contains(string(contents), "`full-1`") {
		t.Fatalf("unexpected report:\n%s", contents)
	}
}

func TestValidateRefusesCandidateStateAblationWithoutMaterialEffect(t *testing.T) {
	runs := t.TempDir()
	makeFixtureRun(t, runs, "overfit", "overfit-1", time.Date(2026, 8, 8, 9, 0, 0, 0, time.UTC))
	makeFixtureRun(t, runs, "mini", "mini-1", time.Date(2026, 8, 8, 10, 0, 0, 0, time.UTC))
	makeFixtureRun(t, runs, "full", "full-1", time.Date(2026, 8, 8, 11, 0, 0, 0, time.UTC))
	path := filepath.Join(runs, "full-1", "final-metrics.json")
	var result runResult
	decodeFixture(t, path, &result)
	var ablation evaluation
	if err := json.Unmarshal(result.Evaluations["best-ablations"], &ablation); err != nil {
		t.Fatal(err)
	}
	ablation.Ablations.OpeningCandidateState.Validation = ablation.Ablations.Normal
	ablation.Ablations.OpeningCandidateState.Effect = metricEffect{}
	result.Evaluations["best-ablations"], _ = json.Marshal(ablation)
	writeFixture(t, path, result)
	writeFixture(t, filepath.Join(runs, "full-1", "evaluations", "best-ablations.json"), ablation)
	_, err := Validate(Options{RunsDir: runs, OverfitRunID: "overfit-1", MiniRunID: "mini-1", FullRunID: "full-1"})
	if err == nil || !strings.Contains(err.Error(), "materially worsen") {
		t.Fatalf("Validate error = %v, want material-effect rejection", err)
	}
}

func TestValidateRejectsMissingEvaluationBadResumeTensorBoardAndDifferentPopulation(t *testing.T) {
	for _, scenario := range []string{"missing evaluation", "bad resume", "bad telemetry proof", "empty tensorboard", "missing TensorBoard tag", "duplicate game telemetry", "different game population", "initial game validation mismatch"} {
		t.Run(scenario, func(t *testing.T) {
			runs := t.TempDir()
			makeFixtureRun(t, runs, "overfit", "overfit-1", time.Date(2026, 8, 8, 9, 0, 0, 0, time.UTC))
			makeFixtureRun(t, runs, "mini", "mini-1", time.Date(2026, 8, 8, 10, 0, 0, 0, time.UTC))
			makeFixtureRun(t, runs, "full", "full-1", time.Date(2026, 8, 8, 11, 0, 0, 0, time.UTC))
			switch scenario {
			case "missing evaluation":
				path := filepath.Join(runs, "full-1", "final-metrics.json")
				var result runResult
				decodeFixture(t, path, &result)
				delete(result.Evaluations, "initial-games100")
				writeFixture(t, path, result)
			case "bad resume":
				path := filepath.Join(runs, "mini-1", "final-metrics.json")
				var result runResult
				decodeFixture(t, path, &result)
				result.ResumeProof.Completed = false
				writeFixture(t, path, result)
			case "empty tensorboard":
				if err := os.RemoveAll(filepath.Join(runs, "mini-1", "events")); err != nil {
					t.Fatal(err)
				}
				if err := os.Mkdir(filepath.Join(runs, "mini-1", "events"), 0o755); err != nil {
					t.Fatal(err)
				}
			case "bad telemetry proof":
				path := filepath.Join(runs, "mini-1", "final-metrics.json")
				var result runResult
				decodeFixture(t, path, &result)
				result.TelemetryProof.TrainingSteps[0] = 0
				writeFixture(t, path, result)
			case "missing TensorBoard tag":
				events := filepath.Join(runs, "overfit-1", "events")
				if err := os.RemoveAll(events); err != nil {
					t.Fatal(err)
				}
				if err := os.Mkdir(events, 0o755); err != nil {
					t.Fatal(err)
				}
				writer, err := tensorboard.New(events)
				if err != nil {
					t.Fatal(err)
				}
				if err := writer.WriteScalars(10, tensorboard.Scalar{Tag: "train/loss", Value: 1}); err != nil {
					t.Fatal(err)
				}
				if err := writer.Close(); err != nil {
					t.Fatal(err)
				}
			case "duplicate game telemetry":
				writeGameEvents(t, filepath.Join(runs, "overfit-1", "events"), 0)
			case "different game population":
				path := filepath.Join(runs, "full-1", "final-metrics.json")
				var result runResult
				decodeFixture(t, path, &result)
				var initial evaluation
				if err := json.Unmarshal(result.Evaluations["initial-games100"], &initial); err != nil {
					t.Fatal(err)
				}
				initial.Games.Games[0].Solution = "other"
				rewriteEvaluationFixture(t, filepath.Join(runs, "full-1"), &result, "initial-games100", initial)
				writeFixture(t, path, result)
			case "initial game validation mismatch":
				path := filepath.Join(runs, "full-1", "final-metrics.json")
				var result runResult
				decodeFixture(t, path, &result)
				var initial evaluation
				if err := json.Unmarshal(result.Evaluations["initial-games100"], &initial); err != nil {
					t.Fatal(err)
				}
				initial.Validation.Top1 += .01
				rewriteEvaluationFixture(t, filepath.Join(runs, "full-1"), &result, "initial-games100", initial)
				writeFixture(t, path, result)
			}
			if _, err := Validate(Options{RunsDir: runs, OverfitRunID: "overfit-1", MiniRunID: "mini-1", FullRunID: "full-1"}); err == nil {
				t.Fatal("Validate unexpectedly succeeded")
			}
		})
	}
}

func TestValidateReplaysEveryStageGate(t *testing.T) {
	for _, test := range []struct {
		name   string
		run    string
		mutate func(*runResult)
	}{
		{
			name: "overfit durable evidence", run: "overfit-1",
			mutate: func(result *runResult) { result.OverfitProof.CheckpointPredictionsReproduced = false },
		},
		{
			name: "mini learning threshold", run: "mini-1",
			mutate: func(result *runResult) { result.FinalTraining.Top16 = result.InitialTraining.Top16 + .099 },
		},
		{
			name: "full broad groups", run: "full-1",
			mutate: func(result *runResult) { result.MajorGroupLearning.ShortlistCount = 1 },
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			runs := t.TempDir()
			makeFixtureRun(t, runs, "overfit", "overfit-1", time.Date(2026, 8, 8, 9, 0, 0, 0, time.UTC))
			makeFixtureRun(t, runs, "mini", "mini-1", time.Date(2026, 8, 8, 10, 0, 0, 0, time.UTC))
			makeFixtureRun(t, runs, "full", "full-1", time.Date(2026, 8, 8, 11, 0, 0, 0, time.UTC))
			path := filepath.Join(runs, test.run, "final-metrics.json")
			var result runResult
			decodeFixture(t, path, &result)
			test.mutate(&result)
			writeFixture(t, path, result)
			if _, err := Validate(Options{RunsDir: runs, OverfitRunID: "overfit-1", MiniRunID: "mini-1", FullRunID: "full-1"}); err == nil {
				t.Fatal("Validate unexpectedly accepted a failed persisted stage gate")
			}
		})
	}
}

func TestValidateRejectsForgedMajorGroupCountsAndDetails(t *testing.T) {
	for _, test := range []struct {
		name   string
		mutate func(*runResult)
	}{
		{
			name:   "forged group names",
			mutate: func(result *runResult) { result.MajorGroupLearning.TurnGroups = []string{"turn_6", "turn_5"} },
		},
		{
			name: "forged group details",
			mutate: func(result *runResult) {
				result.BestValidationDetails.Details.ByShortlistBucket[1].Loss = result.InitialValidationDetails.Details.ByShortlistBucket[1].Loss
			},
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			runs := t.TempDir()
			makeFixtureRun(t, runs, "overfit", "overfit-1", time.Date(2026, 8, 8, 9, 0, 0, 0, time.UTC))
			makeFixtureRun(t, runs, "mini", "mini-1", time.Date(2026, 8, 8, 10, 0, 0, 0, time.UTC))
			makeFixtureRun(t, runs, "full", "full-1", time.Date(2026, 8, 8, 11, 0, 0, 0, time.UTC))
			path := filepath.Join(runs, "full-1", "final-metrics.json")
			var result runResult
			decodeFixture(t, path, &result)
			test.mutate(&result)
			writeFixture(t, path, result)
			if _, err := Validate(Options{RunsDir: runs, OverfitRunID: "overfit-1", MiniRunID: "mini-1", FullRunID: "full-1"}); err == nil || !strings.Contains(err.Error(), "broad-group") {
				t.Fatalf("Validate error = %v, want broad-group rejection", err)
			}
		})
	}
}

func TestWritePublishesIncompleteReportWithoutReplacingSuccessfulReport(t *testing.T) {
	runs := t.TempDir()
	makeFixtureRun(t, runs, "overfit", "overfit-1", time.Date(2026, 8, 8, 9, 0, 0, 0, time.UTC))
	makeFixtureRun(t, runs, "mini", "mini-1", time.Date(2026, 8, 8, 10, 0, 0, 0, time.UTC))
	makeFixtureRun(t, runs, "full", "full-1", time.Date(2026, 8, 8, 11, 0, 0, 0, time.UTC))
	path := filepath.Join(runs, "full-1", "final-metrics.json")
	var result runResult
	decodeFixture(t, path, &result)
	result.MajorGroupLearning.Count++
	writeFixture(t, path, result)

	output := filepath.Join(t.TempDir(), "proof.md")
	_, err := Write(Options{RunsDir: runs, OverfitRunID: "overfit-1", MiniRunID: "mini-1", FullRunID: "full-1", OutputPath: output})
	if err == nil {
		t.Fatal("Write unexpectedly succeeded")
	}
	contents, readErr := os.ReadFile(output)
	if readErr != nil {
		t.Fatal(readErr)
	}
	text := string(contents)
	if !strings.Contains(text, "proofreport: incomplete") || !strings.Contains(text, "`full-1`") || !strings.Contains(text, "| full | `full-1` | full | 2000 |") || !strings.Contains(text, "broad-group") {
		t.Fatalf("incomplete report lacks failure evidence:\n%s", text)
	}

	success := []byte("# Prior successful report\n\n<!-- proofreport: complete -->\n")
	if err := os.WriteFile(output, success, 0o644); err != nil {
		t.Fatal(err)
	}
	_, err = Write(Options{RunsDir: runs, OverfitRunID: "overfit-1", MiniRunID: "mini-1", FullRunID: "full-1", OutputPath: output})
	if err == nil {
		t.Fatal("Write unexpectedly succeeded with failed evidence")
	}
	contents, readErr = os.ReadFile(output)
	if readErr != nil {
		t.Fatal(readErr)
	}
	if string(contents) != string(success) {
		t.Fatalf("failure replaced a successful report:\n%s", contents)
	}
}

func makeFixtureRun(t *testing.T, root, stage, id string, collected time.Time) {
	t.Helper()
	dir := filepath.Join(root, id)
	for _, path := range []string{"events", "checkpoints/initial", "checkpoints/latest", "checkpoints/best", "evaluations"} {
		if err := os.MkdirAll(filepath.Join(dir, path), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	for _, path := range []string{"events/event", "checkpoints/initial/state", "checkpoints/latest/state", "checkpoints/best/state", "run-state.json", "validation-games.jsonl", "training.log"} {
		if err := os.WriteFile(filepath.Join(dir, path), []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	updates := expectedUpdates(stage)
	writeStageEvents(t, filepath.Join(dir, "events"), updates)
	writeFixture(t, filepath.Join(dir, "config.json"), config{Stage: stage, TargetUpdates: updates})
	meta := map[string]any{"collected_at": collected.Format(time.RFC3339Nano), "repositories": map[string]any{"machine_learning": map[string]string{"commit": "ml-commit"}, "synthetic_data": map[string]string{"commit": "data-commit"}, "game_engine": map[string]string{"commit": "engine-commit"}}}
	writeFixture(t, filepath.Join(dir, "metadata.json"), meta)
	initialGroups := validationGroups{
		ByTurn:            []validationGroup{{Name: "turn_1", Examples: 25, Loss: 8}, {Name: "turn_2", Examples: 25, Loss: 8}, {Name: "turn_3", Examples: 24, Loss: 8}},
		ByShortlistBucket: []validationGroup{{Name: "1", Examples: 25, Loss: 8}, {Name: "2-5", Examples: 25, Loss: 8}, {Name: "6-20", Examples: 24, Loss: 8}},
	}
	bestGroups := validationGroups{
		ByTurn:            []validationGroup{{Name: "turn_1", Examples: 25, Loss: 4}, {Name: "turn_2", Examples: 25, Loss: 4}, {Name: "turn_3", Examples: 24, Loss: 4}},
		ByShortlistBucket: []validationGroup{{Name: "1", Examples: 25, Loss: 4}, {Name: "2-5", Examples: 25, Loss: 4}, {Name: "6-20", Examples: 24, Loss: 4}},
	}
	result := runResult{Stage: stage, GlobalUpdate: updates, Passed: true, InitialValidation: metrics{Loss: 8, Top1: .1, Top5: .2, Top16: .3}, FinalValidation: metrics{Loss: 4, Top1: .4, Top5: .5, Top16: .6}, BestValidation: metrics{Loss: 4, Top1: .4, Top5: .5, Top16: .6}, InitialTraining: metrics{Loss: 4, Top1: .1, Top5: .2, Top16: .3}, FinalTraining: metrics{Loss: 1, Top1: .9, Top5: .95, Top16: .99}, ValidationImprovements: 2, MajorGroupLearning: majorGroupLearning{MinimumExamples: 25, TurnCount: 2, TurnGroups: []string{"turn_1", "turn_2"}, ShortlistCount: 2, ShortlistGroups: []string{"1", "2-5"}, Count: 4, Groups: []string{"turn/turn_1", "turn/turn_2", "shortlist/1", "shortlist/2-5"}}, InitialValidationDetails: validationSnapshot{Details: initialGroups}, BestValidationDetails: validationSnapshot{Details: bestGroups}, BestValidationStep: updates, DataOverlapAudit: dataOverlapAudit{TrainingRecords: 100, TrainingUniqueStates: 90, ValidationRecords: 50, ValidationUniqueStates: 45, OverlappingUniqueStates: 5}, Warnings: []string{"5 of 45 unique validation model states also occur in training; their teacher top-1 labels agree. This is state-distribution overlap, not solution-ID split overlap."}, Evaluations: map[string]json.RawMessage{}}
	if stage == "overfit" {
		result.OverfitProof = &overfitProof{
			InitialFixedBatch:               metrics{Loss: 10, Top1: .1, Top5: .2, Top16: .3},
			FinalFixedBatch:                 metrics{Loss: 1, Top1: .95, Top5: 1, Top16: 1},
			LossReducedAtLeastNinetyPercent: true, Top1AtLeastNinetyFivePercent: true,
			DiagnosticsFinite: true, ParametersFinite: true, NonBiasWeightChanged: true, CheckpointPredictionsReproduced: true,
		}
	}
	if stage == "mini" {
		result.ResumeProof = &resumeProof{CheckpointNextRecordIDs: []int{1, 2}, UninterruptedReferenceNextRecordIDs: []int{1, 2}, ResumeFromUpdate: 500, FirstResumedScalarUpdate: 510, Completed: true}
		result.TelemetryProof = &telemetryProof{TrainingSteps: expectedSteps(10, 10, 1000), ValidationSteps: expectedSteps(0, 100, 1000), HistogramStepsByTag: fixtureHistogramSteps()}
	}
	if stage == "overfit" {
		addFixtureEvaluation(t, dir, &result, "initial-games10", "initial", "games10", 0, gameSet(10, 5))
	}
	if stage == "mini" {
		addFixtureEvaluation(t, dir, &result, "latest-games10", "latest", "games10", updates, gameSet(10, 5))
	}
	if stage == "full" {
		addFixtureEvaluation(t, dir, &result, "initial-games100", "initial", "games100", 0, gameSet(100, 50))
		best := gameSet(100, 51)
		addFixtureEvaluation(t, dir, &result, "best-games100", "best", "games100", updates, best)
		normal := validationDetails{Loss: 4, Top1: .4, Top5: .5, Top16: .6}
		ablated := validationDetails{Loss: 4.2, Top1: .39, Top5: .5, Top16: .6}
		turn := validationDetails{Loss: 4.05, Top1: .395, Top5: .5, Top16: .6}
		bonus := validationDetails{Loss: 4.01, Top1: .4, Top5: .499, Top16: .6}
		eval := evaluation{RunID: id, Stage: stage, Mode: "ablations", Checkpoint: "best", CheckpointUpdate: updates, ValidationSplitHash: "validation-hash", Validation: normal, BestLoss: &lossReproduction{Stored: result.BestValidation, Measured: result.BestValidation, Tolerance: metricTolerance, GroupsMatch: true}, Ablations: &ablations{Normal: normal, OpeningCandidateState: ablationMeasurement{Validation: ablated, Effect: metricEffect{Loss: .2, Top1: -.01}}, TurnZero: ablationMeasurement{Validation: turn, Effect: metricEffect{Loss: .05, Top1: -.005}}, NoCandidateBonus: ablationMeasurement{Validation: bonus, Effect: metricEffect{Loss: .01, Top5: -.001}}}}
		addRawFixtureEvaluation(t, dir, &result, "best-ablations", eval)
	}
	switch stage {
	case "overfit":
		writeGameJSONL(t, filepath.Join(dir, "validation-games.jsonl"), gameSet(10, 5).Games)
		writeGameEvents(t, filepath.Join(dir, "events"), 0)
	case "mini":
		writeGameJSONL(t, filepath.Join(dir, "validation-games.jsonl"), gameSet(10, 5).Games)
		writeGameEvents(t, filepath.Join(dir, "events"), updates)
	case "full":
		writeGameJSONL(t, filepath.Join(dir, "validation-games.jsonl"), gameSet(100, 51).Games)
		writeGameEvents(t, filepath.Join(dir, "events"), 0)
		writeGameEvents(t, filepath.Join(dir, "events"), updates)
	}
	writeFixture(t, filepath.Join(dir, "final-metrics.json"), result)
}

func addFixtureEvaluation(t *testing.T, dir string, result *runResult, key, checkpoint, mode string, update int64, value games) {
	t.Helper()
	validation := validationDetails{Loss: 4, Top1: .4, Top5: .5, Top16: .6}
	if (result.Stage == "overfit" && key == "initial-games10") || (result.Stage == "full" && key == "initial-games100") {
		validation = validationDetails{Loss: result.InitialValidation.Loss, Top1: result.InitialValidation.Top1, Top5: result.InitialValidation.Top5, Top16: result.InitialValidation.Top16}
	}
	eval := evaluation{RunID: filepath.Base(dir), Stage: result.Stage, Mode: mode, Checkpoint: checkpoint, CheckpointUpdate: update, ValidationSplitHash: "validation-hash", Validation: validation, Games: &value}
	if checkpoint == "best" {
		eval.BestLoss = &lossReproduction{Stored: result.BestValidation, Measured: result.BestValidation, Tolerance: metricTolerance, GroupsMatch: true}
	}
	if result.Stage == "overfit" && key == "initial-games10" {
		encoded, err := json.Marshal(eval)
		if err != nil {
			t.Fatal(err)
		}
		var document map[string]any
		if err := json.Unmarshal(encoded, &document); err != nil {
			t.Fatal(err)
		}
		document["validation"] = map[string]float64{"loss": validation.Loss, "top1": validation.Top1, "top5": validation.Top5, "top16": validation.Top16}
		encoded, err = json.Marshal(document)
		if err != nil {
			t.Fatal(err)
		}
		result.Evaluations[key] = encoded
		if err := os.WriteFile(filepath.Join(dir, "evaluations", checkpoint+"-"+mode+".json"), encoded, 0o644); err != nil {
			t.Fatal(err)
		}
	} else {
		addRawFixtureEvaluation(t, dir, result, key, eval)
	}
	lines := make([]string, 0, len(value.Games))
	for _, game := range value.Games {
		data, _ := json.Marshal(game)
		lines = append(lines, string(data))
	}
	if err := os.WriteFile(filepath.Join(dir, "evaluations", checkpoint+"-"+mode+".jsonl"), []byte(strings.Join(lines, "\n")+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
}

func addRawFixtureEvaluation(t *testing.T, dir string, result *runResult, key string, value evaluation) {
	t.Helper()
	encoded, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	result.Evaluations[key] = encoded
	writeFixture(t, filepath.Join(dir, "evaluations", value.Checkpoint+"-"+value.Mode+".json"), value)
}

func rewriteEvaluationFixture(t *testing.T, dir string, result *runResult, key string, value evaluation) {
	t.Helper()
	encoded, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	result.Evaluations[key] = encoded
	if err := os.WriteFile(filepath.Join(dir, "evaluations", value.Checkpoint+"-"+value.Mode+".json"), encoded, 0o644); err != nil {
		t.Fatal(err)
	}
	if value.Games != nil {
		lines := make([]string, 0, len(value.Games.Games))
		for _, game := range value.Games.Games {
			data, _ := json.Marshal(game)
			lines = append(lines, string(data))
		}
		if err := os.WriteFile(filepath.Join(dir, "evaluations", value.Checkpoint+"-"+value.Mode+".jsonl"), []byte(strings.Join(lines, "\n")+"\n"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
}

func writeGameJSONL(t *testing.T, path string, games []game) {
	t.Helper()
	lines := make([]string, 0, len(games))
	for _, game := range games {
		data, _ := json.Marshal(game)
		lines = append(lines, string(data))
	}
	if err := os.WriteFile(path, []byte(strings.Join(lines, "\n")+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
}

func fixtureHistogramSteps() map[string][]int64 {
	steps := expectedSteps(0, 100, 1000)
	return map[string][]int64{
		"model/legal_logits":                 slices.Clone(steps),
		"model/beta":                         slices.Clone(steps),
		"model/parameters":                   slices.Clone(steps),
		"optimizer/per_layer_gradient_norms": slices.Clone(steps),
	}
}

func writeStageEvents(t *testing.T, eventsDir string, target int64) {
	t.Helper()
	writer, err := tensorboard.New(eventsDir)
	if err != nil {
		t.Fatal(err)
	}
	training := scalarTags([]string{
		"train/loss", "train/top1_accuracy", "train/top5_accuracy", "train/top16_accuracy",
		"optimizer/learning_rate", "optimizer/global_gradient_norm", "optimizer/applied_global_gradient_norm", "optimizer/parameter_norm", "optimizer/update_to_parameter_norm",
		"data/epoch", "data/examples_consumed", "data/shortlist_size_mean", "data/shortlist_size_min", "data/shortlist_size_max",
		"performance/examples_per_second", "performance/batch_duration", "performance/input_wait_duration",
	})
	opening := scalarTags([]string{"opening/loss", "opening/teacher_rank", "opening/current_guess_id"})
	for _, step := range expectedSteps(10, 10, target) {
		trainValues := slices.Clone(training)
		// Full's event verifier requires an actual downward loss trend.
		trainValues[0].Value = float32(target - step + 1)
		if err := writer.WriteScalars(step, trainValues...); err != nil {
			t.Fatal(err)
		}
		if err := writer.WriteScalars(step, opening...); err != nil {
			t.Fatal(err)
		}
	}
	validationNames := []string{"validation/loss", "validation/top1_accuracy", "validation/top5_accuracy", "validation/top16_accuracy", "validation/output_entropy", "validation/raw_argmax_unavailable", "validation/masked_argmax_violations", "model/output_entropy", "model/beta_mean", "model/beta_min", "model/beta_max", "performance/validation_duration", "opening/highest_guess"}
	for turn := 1; turn <= 6; turn++ {
		for _, metric := range []string{"loss", "top1_accuracy", "top5_accuracy", "top16_accuracy", "output_entropy"} {
			validationNames = append(validationNames, fmt.Sprintf("validation/turn_%d/%s", turn, metric))
		}
	}
	for _, bucket := range []string{"1", "2-5", "6-20", "21-100", ">100"} {
		for _, metric := range []string{"loss", "top1_accuracy", "top5_accuracy", "top16_accuracy", "output_entropy"} {
			validationNames = append(validationNames, "validation/shortlist_"+bucket+"/"+metric)
		}
	}
	validation := scalarTags(validationNames)
	for _, step := range expectedSteps(0, 100, target) {
		if err := writer.WriteScalars(step, validation...); err != nil {
			t.Fatal(err)
		}
		if err := writer.WriteScalars(step, opening...); err != nil {
			t.Fatal(err)
		}
		if err := writer.WriteHistograms(step,
			tensorboard.Histogram{Tag: "model/legal_logits", Values: []float64{1}},
			tensorboard.Histogram{Tag: "model/beta", Values: []float64{1}},
			tensorboard.Histogram{Tag: "model/parameters", Values: []float64{1}},
			tensorboard.Histogram{Tag: "optimizer/per_layer_gradient_norms", Values: []float64{1}},
		); err != nil {
			t.Fatal(err)
		}
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
}

func writeGameEvents(t *testing.T, eventsDir string, step int64) {
	t.Helper()
	writer, err := tensorboard.New(eventsDir)
	if err != nil {
		t.Fatal(err)
	}
	if err := writer.WriteScalars(step, scalarTags([]string{
		"games/solved_fraction", "games/mean_guesses", "games/failures",
		"games/guess_count_1", "games/guess_count_2", "games/guess_count_3",
		"games/guess_count_4", "games/guess_count_5", "games/guess_count_6",
	})...); err != nil {
		t.Fatal(err)
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
}

func scalarTags(tags []string) []tensorboard.Scalar {
	values := make([]tensorboard.Scalar, len(tags))
	for index, tag := range tags {
		values[index] = tensorboard.Scalar{Tag: tag, Value: 1}
	}
	return values
}

func gameSet(count, solved int) games {
	values := make([]game, count)
	for i := range values {
		guess := fmtWord(i)
		values[i] = game{Solution: fmtWord(i + 1000), Solved: i < solved, Guesses: 4, Turns: []turn{{Guess: guess}, {Guess: fmtWord(i + 2000)}, {Guess: fmtWord(i + 3000)}, {Guess: fmtWord(i + 4000)}}}
	}
	return games{Summary: summarize(values), Games: values}
}
func fmtWord(value int) string { return fmt.Sprintf("w%04d", value) }
func writeFixture(t *testing.T, path string, value any) {
	t.Helper()
	contents, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(path, contents, 0o644); err != nil {
		t.Fatal(err)
	}
}
func decodeFixture(t *testing.T, path string, value any) {
	t.Helper()
	contents, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(contents, value); err != nil {
		t.Fatal(err)
	}
}

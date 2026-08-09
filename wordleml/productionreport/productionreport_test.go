package productionreport

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/sam-bee/wordle-ml_machine-learning/runmetadata"
	"github.com/sam-bee/wordle-ml_machine-learning/runstate"
	"github.com/sam-bee/wordle-ml_machine-learning/tensorboard"
)

func TestValidateAndWriteComparesCompleteProductionWithProof(t *testing.T) {
	runsDir := filepath.Join(t.TempDir(), "runs")
	production := createFixtureRun(t, runsDir, "production-001", "production", productionUpdates, 100)
	proof := createFixtureRun(t, runsDir, "proof-full-20260808", "full", proofUpdates, 97)

	report, err := Validate(Options{RunsDir: runsDir, ProductionRunID: production.layout.ID, ProofRunID: proof.layout.ID})
	if err != nil {
		t.Fatalf("Validate: %v", err)
	}
	if report.Production.Updates != productionUpdates || report.Proof.Updates != proofUpdates {
		t.Fatalf("report updates = production %d proof %d", report.Production.Updates, report.Proof.Updates)
	}
	if report.Production.Solved != 100 || report.Proof.Solved != 97 || report.Delta.Solved != 3 {
		t.Fatalf("unexpected game comparison: %+v", report)
	}
	output := filepath.Join(t.TempDir(), "nested", "production-training-report.md")
	written, err := Write(Options{RunsDir: runsDir, ProductionRunID: production.layout.ID, ProofRunID: proof.layout.ID, OutputPath: output})
	if err != nil {
		t.Fatalf("Write: %v", err)
	}
	if written != report {
		t.Fatalf("written report differs from validated report\n got: %+v\nwant: %+v", written, report)
	}
	contents, err := os.ReadFile(output)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"<!-- productionreport: complete -->", "production best", "initial proof best", "production − proof", "10000 / 10000", "2000 / 2000", "validation-split-hash", "machine-learning `ml`", "final-test split is not opened"} {
		if !strings.Contains(string(contents), want) {
			t.Errorf("report lacks %q:\n%s", want, contents)
		}
	}
}

func TestValidateRejectsBrokenProductionArtifactsAndWritePreservesOutput(t *testing.T) {
	t.Run("missing production safety evidence", func(t *testing.T) {
		runsDir := filepath.Join(t.TempDir(), "runs")
		production := createFixtureRun(t, runsDir, "production-001", "production", productionUpdates, 100)
		proof := createFixtureRun(t, runsDir, "proof-full-20260808", "full", proofUpdates, 97)
		production.final.ProductionSafety = nil
		writeJSON(t, production.layout.FinalMetricsPath, production.final)
		if _, err := Validate(Options{RunsDir: runsDir, ProductionRunID: production.layout.ID, ProofRunID: proof.layout.ID}); err == nil || !strings.Contains(err.Error(), "finite loss, gradient, and parameter evidence") {
			t.Fatalf("Validate error = %v, want production safety evidence", err)
		}
	})

	t.Run("unmatched reproduction groups", func(t *testing.T) {
		runsDir := filepath.Join(t.TempDir(), "runs")
		production := createFixtureRun(t, runsDir, "production-001", "production", productionUpdates, 100)
		proof := createFixtureRun(t, runsDir, "proof-full-20260808", "full", proofUpdates, 97)
		production.evaluation.BestLoss.GroupsMatch = false
		replaceEvaluation(t, production)
		if _, err := Validate(Options{RunsDir: runsDir, ProductionRunID: production.layout.ID, ProofRunID: proof.layout.ID}); err == nil || !strings.Contains(err.Error(), "best-loss reproduction") {
			t.Fatalf("Validate error = %v, want best-loss reproduction", err)
		}
	})

	t.Run("trajectory does not match evaluation", func(t *testing.T) {
		runsDir := filepath.Join(t.TempDir(), "runs")
		production := createFixtureRun(t, runsDir, "production-001", "production", productionUpdates, 100)
		proof := createFixtureRun(t, runsDir, "proof-full-20260808", "full", proofUpdates, 97)
		writeFile(t, filepath.Join(production.layout.EvaluationsDir, "best-games100.jsonl"), []byte("{}\n"))
		if _, err := Validate(Options{RunsDir: runsDir, ProductionRunID: production.layout.ID, ProofRunID: proof.layout.ID}); err == nil || !strings.Contains(err.Error(), "trajectories") {
			t.Fatalf("Validate error = %v, want trajectories", err)
		}
	})

	t.Run("invalid config leaves existing output untouched", func(t *testing.T) {
		runsDir := filepath.Join(t.TempDir(), "runs")
		production := createFixtureRun(t, runsDir, "production-001", "production", productionUpdates, 100)
		proof := createFixtureRun(t, runsDir, "proof-full-20260808", "full", proofUpdates, 97)
		production.config.TargetUpdates = 9_999
		writeJSON(t, production.layout.ConfigPath, production.config)
		output := filepath.Join(t.TempDir(), "production-report.md")
		writeFile(t, output, []byte("existing complete report\n"))
		if _, err := Write(Options{RunsDir: runsDir, ProductionRunID: production.layout.ID, ProofRunID: proof.layout.ID, OutputPath: output}); err == nil {
			t.Fatal("Write unexpectedly succeeded with invalid config")
		}
		contents, err := os.ReadFile(output)
		if err != nil {
			t.Fatal(err)
		}
		if got, want := string(contents), "existing complete report\n"; got != want {
			t.Fatalf("output = %q, want %q", got, want)
		}
	})
}

type fixtureRun struct {
	layout     runstate.Layout
	config     fixedConfig
	final      finalMetrics
	evaluation evaluation
}

func createFixtureRun(t *testing.T, runsDir, runID, stage string, updates int64, solved int) fixtureRun {
	t.Helper()
	layout, err := runstate.Create(runsDir, runID)
	if err != nil {
		t.Fatal(err)
	}
	config := fixedConfig{
		Stage: stage, BatchSize: 256, LearningRate: 3e-4, TargetUpdates: updates,
		ValidationEvery: validationEvery, CheckpointEvery: validationEvery, ScalarEvery: scalarEvery,
		Seed: 20260808, Precision: "float32", Objective: "masked_sparse_cross_entropy_teacher_top1",
		Optimizer: "Adam", LearningRateMode: "constant", WeightDecay: 0, GradientClipNorm: 5,
	}
	configBytes, err := json.Marshal(config)
	if err != nil {
		t.Fatal(err)
	}
	writeJSON(t, layout.ConfigPath, config)
	metadata := fixtureMetadata(t, configBytes)
	writeJSON(t, layout.MetadataPath, metadata)
	for _, dir := range []string{layout.InitialCheckpointDir, layout.LatestCheckpointDir, layout.BestCheckpointDir} {
		writeFile(t, filepath.Join(dir, "checkpoint.bin"), []byte("checkpoint"))
	}
	writeFile(t, layout.TrainingLogPath, []byte("training complete\n"))

	initial := metrics{Loss: 8, Top1: .01, Top5: .02, Top16: .03}
	best := metrics{Loss: 2, Top1: .60, Top5: .70, Top16: .80}
	snapshots := make([]snapshot, 0, updates/validationEvery+1)
	for update := int64(0); update <= updates; update += validationEvery {
		value := best
		if update == 0 {
			value = initial
		}
		snapshots = append(snapshots, fixtureSnapshot(update, value))
	}
	final := finalMetrics{
		Stage: stage, GlobalUpdate: updates, Passed: true,
		InitialTraining:   metrics{Loss: 7, Top1: .1, Top5: .2, Top16: .3},
		FinalTraining:     metrics{Loss: 1, Top1: .7, Top5: .8, Top16: .9},
		InitialValidation: initial, FinalValidation: best, BestValidation: best, BestValidationStep: updates,
		InitialValidationDetails: snapshots[0], FinalValidationDetails: snapshots[len(snapshots)-1], BestValidationDetails: snapshots[len(snapshots)-1],
		ValidationSnapshots: snapshots, Evaluations: make(map[string]json.RawMessage),
	}
	if stage == "production" {
		final.ProductionSafety = &productionSafety{LossFinite: true, GradientsFinite: true, ParametersFinite: true, UpdatesChecked: updates}
	}
	games := fixtureGames(solved)
	evaluation := evaluation{
		RunID: runID, Stage: stage, Mode: "games100", Checkpoint: "best", CheckpointUpdate: updates,
		ValidationSplitHash: "validation-split-hash", Validation: snapshots[len(snapshots)-1].Details,
		BestLoss: &reproduction{Stored: best, Measured: best, Tolerance: metricTolerance, GroupsMatch: true}, Games: &games,
	}
	encodedEvaluation, err := json.Marshal(evaluation)
	if err != nil {
		t.Fatal(err)
	}
	final.Evaluations["best-games100"] = encodedEvaluation
	writeJSON(t, layout.FinalMetricsPath, final)
	state := runstate.State{GlobalUpdate: updates, ShuffleSeed: 20260808, BestValidation: &runstate.BestValidation{Value: best.Loss, Update: updates}}
	if err := layout.WriteStateMirror(state); err != nil {
		t.Fatal(err)
	}
	writeJSON(t, filepath.Join(layout.EvaluationsDir, "best-games100.json"), evaluation)
	jsonl := marshalJSONL(t, games.Games)
	writeFile(t, filepath.Join(layout.EvaluationsDir, "best-games100.jsonl"), jsonl)
	writeFile(t, layout.ValidationGamesPath, jsonl)
	writeEvents(t, layout.EventsDir, stage, updates, games.Summary)
	return fixtureRun{layout: layout, config: config, final: final, evaluation: evaluation}
}

func replaceEvaluation(t *testing.T, run fixtureRun) {
	t.Helper()
	encoded, err := json.Marshal(run.evaluation)
	if err != nil {
		t.Fatal(err)
	}
	run.final.Evaluations["best-games100"] = encoded
	writeJSON(t, run.layout.FinalMetricsPath, run.final)
	writeJSON(t, filepath.Join(run.layout.EvaluationsDir, "best-games100.json"), run.evaluation)
}

func fixtureMetadata(t *testing.T, config []byte) runmetadata.Manifest {
	t.Helper()
	hash := strings.Repeat("a", 64)
	artifact := func(path string) runmetadata.Artifact { return runmetadata.Artifact{Path: path, SHA256: hash} }
	manifest := runmetadata.Manifest{
		SchemaVersion: 1,
		CollectedAt:   time.Date(2026, 8, 9, 0, 0, 0, 0, time.UTC),
		Repositories: runmetadata.Repositories{
			MachineLearning: runmetadata.Repository{Path: ".", Commit: "ml"},
			SyntheticData:   runmetadata.Repository{Path: "synthetic", Commit: "synthetic"},
			GameEngine:      runmetadata.Repository{Path: "engine", Commit: "engine"},
		},
		Dataset:    runmetadata.Dataset{Format: "WDIT", Version: "3", Files: []runmetadata.Artifact{artifact("data/imitation/wordle-validation.bin")}},
		Vocabulary: []runmetadata.Artifact{artifact("data/wordlist-action-space-4739.csv")},
		Splits: runmetadata.Splits{
			Training:   []runmetadata.Artifact{artifact("data/wordlist-valid-solutions-train-2109.csv")},
			Validation: []runmetadata.Artifact{artifact("data/wordlist-valid-solutions-validation-100.csv")},
			Test:       []runmetadata.Artifact{artifact("data/wordlist-valid-solutions-test-100.csv")},
		},
		ModelParameterCount: 1_046_596,
		Runtime: runmetadata.RuntimeMetadata{
			GoVersion: "go1.26", GOOS: "linux", GOARCH: "amd64", GoMLXVersion: "v0.28.0", Backend: "xla:cuda",
			GoMLXDetails: map[string]string{"backend_name": "xla"}, GPUDetails: map[string]string{"name": "approved GPU"},
			CUDADetails: map[string]string{"version": "13"}, PJRTDetails: map[string]string{"backend_description": "xla cuda"},
		},
		Seed: 20260808, EffectiveConfig: config,
	}
	if err := manifest.Validate(); err != nil {
		t.Fatalf("fixture manifest: %v", err)
	}
	return manifest
}

func fixtureSnapshot(update int64, value metrics) snapshot {
	return snapshot{
		Update: update, Metrics: value,
		Details: validation{
			Examples: 2500, Loss: value.Loss, Top1: value.Top1, Top5: value.Top5, Top16: value.Top16,
			ByTurn:            []group{{Name: "turn_2", Examples: 2500, Loss: value.Loss, Top1: value.Top1, Top5: value.Top5, Top16: value.Top16}},
			ByShortlistBucket: []group{{Name: "1", Examples: 2500, Loss: value.Loss, Top1: value.Top1, Top5: value.Top5, Top16: value.Top16}},
		},
	}
}

func fixtureGames(solved int) gameEvaluation {
	games := make([]game, 0, 100)
	var summary gameSummary
	for index := 0; index < 100; index++ {
		guesses := index%6 + 1
		result := game{Solution: fmtSolution(index), Guesses: guesses, Solved: index < solved, Turns: make([]turn, 0, guesses)}
		if !result.Solved {
			result.Failure = "unsolved_after_six_guesses"
		}
		for turnIndex := 0; turnIndex < guesses; turnIndex++ {
			result.Turns = append(result.Turns, turn{Turn: turnIndex + 1, Guess: fmtGuess(index, turnIndex)})
		}
		games = append(games, result)
		summary.GuessCountDistribution[guesses-1]++
		summary.MeanGuesses += float64(guesses)
		if result.Solved {
			summary.Solved++
		} else {
			summary.Failures++
			summary.FailedSolutions = append(summary.FailedSolutions, result.Solution)
		}
	}
	summary.Games = len(games)
	summary.SolvedFraction = float64(summary.Solved) / float64(summary.Games)
	summary.MeanGuesses /= float64(summary.Games)
	return gameEvaluation{Summary: summary, Games: games}
}

func fmtSolution(index int) string             { return fmt.Sprintf("S%04d", index) }
func fmtGuess(gameIndex, turnIndex int) string { return fmt.Sprintf("G%03d%d", gameIndex, turnIndex) }

func marshalJSONL(t *testing.T, games []game) []byte {
	t.Helper()
	var builder strings.Builder
	for _, game := range games {
		encoded, err := json.Marshal(game)
		if err != nil {
			t.Fatal(err)
		}
		builder.Write(encoded)
		builder.WriteByte('\n')
	}
	return []byte(builder.String())
}

func writeEvents(t *testing.T, dir, stage string, updates int64, summary gameSummary) {
	t.Helper()
	writer, err := tensorboard.New(dir)
	if err != nil {
		t.Fatal(err)
	}
	trainingTags := []string{
		"train/loss", "train/top1_accuracy", "train/top5_accuracy", "train/top16_accuracy",
		"optimizer/learning_rate", "optimizer/global_gradient_norm", "optimizer/applied_global_gradient_norm", "optimizer/parameter_norm", "optimizer/update_to_parameter_norm",
		"data/epoch", "data/examples_consumed", "data/shortlist_size_mean", "data/shortlist_size_min", "data/shortlist_size_max",
		"performance/examples_per_second", "performance/batch_duration", "performance/input_wait_duration",
	}
	validationTags := []string{
		"validation/loss", "validation/top1_accuracy", "validation/top5_accuracy", "validation/top16_accuracy", "validation/output_entropy", "validation/raw_argmax_unavailable", "validation/masked_argmax_violations",
		"model/output_entropy", "model/beta_mean", "model/beta_min", "model/beta_max", "performance/validation_duration", "opening/highest_guess",
	}
	for turn := 2; turn <= 6; turn++ {
		for _, metric := range []string{"loss", "top1_accuracy", "top5_accuracy", "top16_accuracy", "output_entropy"} {
			validationTags = append(validationTags, fmt.Sprintf("validation/turn_%d/%s", turn, metric))
		}
	}
	for _, bucket := range []string{"1", "2-5", "6-20", "21-100", ">100"} {
		for _, metric := range []string{"loss", "top1_accuracy", "top5_accuracy", "top16_accuracy", "output_entropy"} {
			validationTags = append(validationTags, "validation/shortlist_"+bucket+"/"+metric)
		}
	}
	openingTags := []string{"opening/loss", "opening/teacher_rank", "opening/current_guess_id"}
	histogramTags := []string{"model/legal_logits", "model/beta", "model/parameters", "optimizer/per_layer_gradient_norms"}
	for step := scalarEvery; step <= updates; step += scalarEvery {
		values := make([]tensorboard.Scalar, 0, len(trainingTags)+len(openingTags))
		for _, tag := range trainingTags {
			value := float32(1)
			if tag == "train/loss" && stage == "full" {
				value = float32(updates-step+1) / float32(updates)
			}
			values = append(values, tensorboard.Scalar{Tag: tag, Value: value})
		}
		for _, tag := range openingTags {
			values = append(values, tensorboard.Scalar{Tag: tag, Value: 1})
		}
		if err := writer.WriteScalars(step, values...); err != nil {
			t.Fatal(err)
		}
	}
	for step := int64(0); step <= updates; step += validationEvery {
		values := make([]tensorboard.Scalar, 0, len(validationTags)+len(openingTags))
		for _, tag := range validationTags {
			values = append(values, tensorboard.Scalar{Tag: tag, Value: 1})
		}
		for _, tag := range openingTags {
			values = append(values, tensorboard.Scalar{Tag: tag, Value: 1})
		}
		if err := writer.WriteScalars(step, values...); err != nil {
			t.Fatal(err)
		}
		histograms := make([]tensorboard.Histogram, 0, len(histogramTags))
		for _, tag := range histogramTags {
			histograms = append(histograms, tensorboard.Histogram{Tag: tag, Values: []float64{1}})
		}
		if err := writer.WriteHistograms(step, histograms...); err != nil {
			t.Fatal(err)
		}
	}
	games := []tensorboard.Scalar{
		{Tag: "games/solved_fraction", Value: float32(summary.SolvedFraction)},
		{Tag: "games/mean_guesses", Value: float32(summary.MeanGuesses)},
		{Tag: "games/failures", Value: float32(summary.Failures)},
	}
	for index, count := range summary.GuessCountDistribution {
		games = append(games, tensorboard.Scalar{Tag: fmt.Sprintf("games/guess_count_%d", index+1), Value: float32(count)})
	}
	if err := writer.WriteScalars(updates, games...); err != nil {
		t.Fatal(err)
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
}

func writeJSON(t *testing.T, path string, value any) {
	t.Helper()
	contents, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	writeFile(t, path, append(contents, '\n'))
}

func writeFile(t *testing.T, path string, contents []byte) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, contents, 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestDifferenceIncludesGuessDistribution(t *testing.T) {
	production := Checkpoint{GuessCounts: [6]int{1, 2, 3, 4, 5, 6}, Loss: 1, Top1: .4, Solved: 8, SolvedRate: .8, MeanGuesses: 3, Failures: 2}
	proof := Checkpoint{GuessCounts: [6]int{6, 5, 4, 3, 2, 1}, Loss: 2, Top1: .2, Solved: 3, SolvedRate: .3, MeanGuesses: 4, Failures: 7}
	got := difference(production, proof)
	if got.GuessCounts != [6]int{-5, -3, -1, 1, 3, 5} || math.Abs(got.Loss+1) > 1e-12 || got.Solved != 5 || got.Failures != -5 {
		t.Fatalf("difference = %+v", got)
	}
}

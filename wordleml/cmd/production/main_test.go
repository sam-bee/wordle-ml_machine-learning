package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/sam-bee/wordle-ml_machine-learning/productionreport"
	"github.com/sam-bee/wordle-ml_machine-learning/proofeval"
	"github.com/sam-bee/wordle-ml_machine-learning/proofrun"
)

func TestParseConfigAllowsOnlyRunAndPathControls(t *testing.T) {
	for _, args := range [][]string{
		nil,
		{"-run-id="},
		{"-run-id=../escape"},
		{"-run-id=production", "-target-updates=100"},
		{"-run-id=production", "-stage=full"},
		{"-run-id=production", "-report="},
	} {
		if _, err := parseConfig(args, &bytes.Buffer{}); err == nil {
			t.Fatalf("parseConfig(%q) unexpectedly succeeded", args)
		}
	}

	got, err := parseConfig([]string{"-data-dir=/frozen", "-runs-dir=/runs", "-run-id=production-20260809-010203", "-report=/reports/production.md"}, &bytes.Buffer{})
	if err != nil {
		t.Fatal(err)
	}
	if got.dataDir != "/frozen" || got.runsDir != "/runs" || got.runID != "production-20260809-010203" || got.reportPath != "/reports/production.md" {
		t.Fatalf("config = %#v", got)
	}
}

func TestParseConfigUsesDataAndRunsEnvironmentDefaults(t *testing.T) {
	t.Setenv("WORDLEML_DATA_DIR", "/workspace/data")
	t.Setenv("WORDLEML_RUNS_DIR", "")
	got, err := parseConfig([]string{"-run-id=production-1"}, &bytes.Buffer{})
	if err != nil {
		t.Fatal(err)
	}
	if got.runsDir != "/workspace/runs" || got.dataDir != "/workspace/data" || got.reportPath != defaultReportPath {
		t.Fatalf("config defaults = %#v", got)
	}

	t.Setenv("WORDLEML_RUNS_DIR", "/mounted/runs")
	got, err = parseConfig([]string{"-run-id=production-1"}, &bytes.Buffer{})
	if err != nil {
		t.Fatal(err)
	}
	if got.runsDir != "/mounted/runs" {
		t.Fatalf("runs directory = %q", got.runsDir)
	}
}

func TestExecuteChainsFixedProductionWorkAndPublishesCompletedStatus(t *testing.T) {
	runs := t.TempDir()
	configuration := config{dataDir: "/frozen", runsDir: runs, runID: "production-20260809-010203", reportPath: "/reports/production.md"}
	var gotTrain proofrun.Options
	var gotEvaluation proofeval.Options
	var gotReport productionreport.Options
	var output bytes.Buffer
	statusPath := filepath.Join(runs, configuration.runID+".status.json")
	err := execute(configuration, dependencies{
		train: func(options proofrun.Options, writer io.Writer) (proofrun.Result, error) {
			assertRunningStatus(t, statusPath, "training")
			gotTrain = options
			return proofrun.Result{}, nil
		},
		evaluate: func(_ context.Context, options proofeval.Options) (proofeval.Result, error) {
			assertRunningStatus(t, statusPath, "evaluation")
			gotEvaluation = options
			return proofeval.Result{}, nil
		},
		report: func(options productionreport.Options) error {
			assertRunningStatus(t, statusPath, "report")
			gotReport = options
			return nil
		},
	}, &output)
	if err != nil {
		t.Fatal(err)
	}
	if gotTrain.Stage != proofrun.Production || gotTrain.StopAt != 0 || gotTrain.DataDir != configuration.dataDir || gotTrain.RunsDir != runs || gotTrain.RunID != configuration.runID {
		t.Fatalf("train options = %#v", gotTrain)
	}
	if gotEvaluation.DataDir != configuration.dataDir || gotEvaluation.RunsDir != runs || gotEvaluation.RunID != configuration.runID || gotEvaluation.Checkpoint != proofeval.Best || gotEvaluation.Mode != proofeval.Games100 {
		t.Fatalf("evaluation options = %#v", gotEvaluation)
	}
	if gotReport.RunsDir != runs || gotReport.ProductionRunID != configuration.runID || gotReport.ProofRunID != defaultProofRunID || gotReport.OutputPath != configuration.reportPath {
		t.Fatalf("report options = %#v", gotReport)
	}
	gotStatus := readStatus(t, statusPath)
	if gotStatus.RunID != configuration.runID || gotStatus.Phase != "completed" || gotStatus.Outcome != "completed" || gotStatus.Error != "" || gotStatus.UpdatedAt.IsZero() {
		t.Fatalf("status = %#v", gotStatus)
	}
	for _, phase := range []string{"training", "evaluating best checkpoint", "writing comparison report", "completed"} {
		if !strings.Contains(output.String(), phase) {
			t.Errorf("progress output %q does not contain %q", output.String(), phase)
		}
	}
}

func TestExecuteStopsAtTrainingFailure(t *testing.T) {
	testStopsAfter(t, "training", dependencies{
		train: func(proofrun.Options, io.Writer) (proofrun.Result, error) {
			return proofrun.Result{}, errors.New("training failed")
		},
		evaluate: func(context.Context, proofeval.Options) (proofeval.Result, error) {
			t.Fatal("evaluation ran after training failure")
			return proofeval.Result{}, nil
		},
		report: func(productionreport.Options) error {
			t.Fatal("report ran after training failure")
			return nil
		},
	})
}

func TestExecuteStopsAtEvaluationFailure(t *testing.T) {
	testStopsAfter(t, "evaluation", dependencies{
		train: func(proofrun.Options, io.Writer) (proofrun.Result, error) { return proofrun.Result{}, nil },
		evaluate: func(context.Context, proofeval.Options) (proofeval.Result, error) {
			return proofeval.Result{}, errors.New("evaluation failed")
		},
		report: func(productionreport.Options) error {
			t.Fatal("report ran after evaluation failure")
			return nil
		},
	})
}

func TestExecuteStopsAtReportFailure(t *testing.T) {
	testStopsAfter(t, "report", dependencies{
		train:    func(proofrun.Options, io.Writer) (proofrun.Result, error) { return proofrun.Result{}, nil },
		evaluate: func(context.Context, proofeval.Options) (proofeval.Result, error) { return proofeval.Result{}, nil },
		report:   func(productionreport.Options) error { return errors.New("report failed") },
	})
}

func testStopsAfter(t *testing.T, phase string, deps dependencies) {
	t.Helper()
	runs := t.TempDir()
	configuration := config{dataDir: "/frozen", runsDir: runs, runID: "production-1", reportPath: "/reports/production.md"}
	err := execute(configuration, deps, &bytes.Buffer{})
	if err == nil || !strings.Contains(err.Error(), phase+" failed") {
		t.Fatalf("execute error = %v, want %s failure", err, phase)
	}
	got := readStatus(t, filepath.Join(runs, configuration.runID+".status.json"))
	if got.Phase != phase || got.Outcome != "failed" || !strings.Contains(got.Error, phase+" failed") {
		t.Fatalf("failure status = %#v", got)
	}
}

func TestExecuteRejectsMissingDependencyBeforeWritingStatus(t *testing.T) {
	runs := t.TempDir()
	configuration := config{dataDir: "/frozen", runsDir: runs, runID: "production-1", reportPath: "/reports/production.md"}
	if err := execute(configuration, dependencies{}, &bytes.Buffer{}); err == nil {
		t.Fatal("execute unexpectedly accepted missing dependencies")
	}
	if _, err := os.Stat(filepath.Join(runs, configuration.runID+".status.json")); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("status exists or stat failed: %v", err)
	}
}

func readStatus(t *testing.T, path string) status {
	t.Helper()
	contents, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var value status
	if err := json.Unmarshal(contents, &value); err != nil {
		t.Fatal(err)
	}
	return value
}

func assertRunningStatus(t *testing.T, path, phase string) {
	t.Helper()
	got := readStatus(t, path)
	if got.Phase != phase || got.Outcome != "running" || got.Error != "" || got.UpdatedAt.IsZero() {
		t.Fatalf("running status = %#v, want %s/running", got, phase)
	}
}

// Command production runs the one fixed long supervised-training production
// sequence. It deliberately has no model or optimisation tuning surface.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/sam-bee/wordle-ml_machine-learning/productionreport"
	"github.com/sam-bee/wordle-ml_machine-learning/proofeval"
	"github.com/sam-bee/wordle-ml_machine-learning/proofrun"
	"github.com/sam-bee/wordle-ml_machine-learning/runstate"
)

const (
	defaultProofRunID                = "proof-full-20260808"
	defaultOriginalProductionRunID   = "production-20260809-005026Z"
	defaultReportPath                = "../docs/ml/production-training-report.md"
	defaultSeedReplicationReportPath = "../docs/ml/seed-replication-report.md"
)

type config struct {
	dataDir         string
	runsDir         string
	runID           string
	reportPath      string
	seedReplication bool
}

// status is deliberately separate from the self-contained run artifacts so
// an operator can identify a stopped production chain before entering a run
// directory. Every update is atomically published beside the runs directory.
type status struct {
	RunID     string    `json:"run_id"`
	Phase     string    `json:"phase"`
	Outcome   string    `json:"outcome"`
	Error     string    `json:"error,omitempty"`
	UpdatedAt time.Time `json:"updated_at"`
}

type dependencies struct {
	train             func(proofrun.Options, io.Writer) (proofrun.Result, error)
	evaluate          func(context.Context, proofeval.Options) (proofeval.Result, error)
	compare           func(context.Context, proofeval.PairedTop1Options) (proofeval.PairedTop1Comparison, error)
	report            func(productionreport.Options) error
	replicationReport func(productionreport.SeedReplicationOptions) error
}

func productionDependencies() dependencies {
	return dependencies{
		train:    proofrun.Run,
		evaluate: proofeval.Run,
		compare:  proofeval.CompareBestTeacherTop1,
		report: func(options productionreport.Options) error {
			_, err := productionreport.Write(options)
			return err
		},
		replicationReport: func(options productionreport.SeedReplicationOptions) error {
			_, err := productionreport.WriteSeedReplication(options)
			return err
		},
	}
}

func main() {
	if err := run(os.Args[1:], os.Stdout, os.Stderr); err != nil {
		fmt.Fprintf(os.Stderr, "production: %v\n", err)
		os.Exit(1)
	}
}

func run(args []string, stdout, stderr io.Writer) error {
	configuration, err := parseConfig(args, stderr)
	if err != nil {
		return err
	}
	return execute(configuration, productionDependencies(), stdout)
}

func parseConfig(args []string, stderr io.Writer) (config, error) {
	dataDir := os.Getenv("WORDLEML_DATA_DIR")
	if dataDir == "" {
		dataDir = "../data"
	}
	runsDir := os.Getenv("WORDLEML_RUNS_DIR")
	if runsDir == "" {
		runsDir = filepath.Join(filepath.Dir(dataDir), "runs")
	}

	flags := flag.NewFlagSet("production", flag.ContinueOnError)
	flags.SetOutput(stderr)
	flags.Usage = func() {
		fmt.Fprintln(stderr, "usage: production -run-id=<timestamped-id> [flags]")
		flags.PrintDefaults()
	}
	var value config
	flags.StringVar(&value.dataDir, "data-dir", dataDir, "directory containing frozen vocabularies and imitation data")
	flags.StringVar(&value.runsDir, "runs-dir", runsDir, "directory containing self-contained training runs")
	flags.StringVar(&value.runID, "run-id", "", "required fresh timestamped production-run identifier")
	flags.StringVar(&value.reportPath, "report", defaultReportPath, "output path for the production training report")
	flags.BoolVar(&value.seedReplication, "seed-replication-20260809", false, "run the one fixed seed-20260809 production replication")
	if err := flags.Parse(args); err != nil {
		return value, err
	}
	if flags.NArg() != 0 {
		return value, fmt.Errorf("unexpected argument %q", flags.Arg(0))
	}
	if strings.TrimSpace(value.runID) == "" {
		return value, errors.New("-run-id is required")
	}
	if err := runstate.ValidateRunID(value.runID); err != nil {
		return value, err
	}
	if strings.TrimSpace(value.dataDir) == "" || strings.TrimSpace(value.runsDir) == "" {
		return value, errors.New("-data-dir and -runs-dir must not be empty")
	}
	if strings.TrimSpace(value.reportPath) == "" {
		return value, errors.New("-report is required")
	}
	if value.seedReplication {
		reportExplicit := false
		flags.Visit(func(current *flag.Flag) {
			if current.Name == "report" {
				reportExplicit = true
			}
		})
		if !strings.HasPrefix(value.runID, "seed-replication-") {
			return value, errors.New("seed-replication run ID must begin with seed-replication-")
		}
		if !reportExplicit {
			value.reportPath = defaultSeedReplicationReportPath
		}
		if filepath.Base(value.reportPath) == filepath.Base(defaultReportPath) {
			return value, errors.New("seed replication must not overwrite the production training report")
		}
	}
	return value, nil
}

func execute(configuration config, deps dependencies, stdout io.Writer) error {
	if err := validateDependencies(configuration, deps); err != nil {
		return err
	}
	statusPath := filepath.Join(configuration.runsDir, configuration.runID+".status.json")
	if err := writeStatus(statusPath, status{RunID: configuration.runID, Phase: "training", Outcome: "running"}); err != nil {
		return fmt.Errorf("publish production training status: %w", err)
	}
	if _, err := fmt.Fprintf(stdout, "production run %s: training\n", configuration.runID); err != nil {
		return fail(statusPath, configuration.runID, "training", fmt.Errorf("write progress: %w", err))
	}
	stage := proofrun.Production
	if configuration.seedReplication {
		stage = proofrun.SeedReplication
	}
	if _, err := deps.train(proofrun.Options{
		DataDir: configuration.dataDir,
		RunsDir: configuration.runsDir,
		RunID:   configuration.runID,
		Stage:   stage,
	}, stdout); err != nil {
		return fail(statusPath, configuration.runID, "training", fmt.Errorf("run fixed production training: %w", err))
	}

	if err := writeStatus(statusPath, status{RunID: configuration.runID, Phase: "evaluation", Outcome: "running"}); err != nil {
		return fail(statusPath, configuration.runID, "evaluation", fmt.Errorf("publish production evaluation status: %w", err))
	}
	if _, err := fmt.Fprintf(stdout, "production run %s: evaluating best checkpoint on 100 validation games\n", configuration.runID); err != nil {
		return fail(statusPath, configuration.runID, "evaluation", fmt.Errorf("write progress: %w", err))
	}
	if _, err := deps.evaluate(context.Background(), proofeval.Options{
		DataDir:    configuration.dataDir,
		RunsDir:    configuration.runsDir,
		RunID:      configuration.runID,
		Checkpoint: proofeval.Best,
		Mode:       proofeval.Games100,
	}); err != nil {
		return fail(statusPath, configuration.runID, "evaluation", fmt.Errorf("evaluate best production checkpoint: %w", err))
	}
	if configuration.seedReplication {
		if err := writeStatus(statusPath, status{RunID: configuration.runID, Phase: "paired-validation", Outcome: "running"}); err != nil {
			return fail(statusPath, configuration.runID, "paired-validation", fmt.Errorf("publish paired validation status: %w", err))
		}
		if _, err := fmt.Fprintf(stdout, "production run %s: comparing paired validation states with %s\n", configuration.runID, defaultOriginalProductionRunID); err != nil {
			return fail(statusPath, configuration.runID, "paired-validation", fmt.Errorf("write progress: %w", err))
		}
		if _, err := deps.compare(context.Background(), proofeval.PairedTop1Options{
			DataDir: configuration.dataDir, RunsDir: configuration.runsDir,
			OriginalRunID: defaultOriginalProductionRunID, ReplicationRunID: configuration.runID,
		}); err != nil {
			return fail(statusPath, configuration.runID, "paired-validation", fmt.Errorf("compare paired validation states: %w", err))
		}
	}

	if err := writeStatus(statusPath, status{RunID: configuration.runID, Phase: "report", Outcome: "running"}); err != nil {
		return fail(statusPath, configuration.runID, "report", fmt.Errorf("publish production report status: %w", err))
	}
	if _, err := fmt.Fprintf(stdout, "production run %s: writing comparison report\n", configuration.runID); err != nil {
		return fail(statusPath, configuration.runID, "report", fmt.Errorf("write progress: %w", err))
	}
	if configuration.seedReplication {
		if err := deps.replicationReport(productionreport.SeedReplicationOptions{
			RunsDir: configuration.runsDir, OriginalRunID: defaultOriginalProductionRunID,
			ReplicationRunID: configuration.runID, OutputPath: configuration.reportPath,
		}); err != nil {
			return fail(statusPath, configuration.runID, "report", fmt.Errorf("write seed-replication comparison report: %w", err))
		}
	} else if err := deps.report(productionreport.Options{
		RunsDir: configuration.runsDir, ProductionRunID: configuration.runID,
		ProofRunID: defaultProofRunID, OutputPath: configuration.reportPath,
	}); err != nil {
		return fail(statusPath, configuration.runID, "report", fmt.Errorf("write production comparison report: %w", err))
	}
	if _, err := fmt.Fprintf(stdout, "production run %s: completed\n", configuration.runID); err != nil {
		return fail(statusPath, configuration.runID, "report", fmt.Errorf("write progress: %w", err))
	}
	if err := writeStatus(statusPath, status{RunID: configuration.runID, Phase: "completed", Outcome: "completed"}); err != nil {
		return fail(statusPath, configuration.runID, "completed", fmt.Errorf("publish completed production status: %w", err))
	}
	return nil
}

func validateDependencies(configuration config, deps dependencies) error {
	if deps.train == nil || deps.evaluate == nil {
		return errors.New("production orchestration dependencies must not be nil")
	}
	if configuration.seedReplication && (deps.compare == nil || deps.replicationReport == nil) {
		return errors.New("seed-replication orchestration dependencies must not be nil")
	}
	if !configuration.seedReplication && deps.report == nil {
		return errors.New("production orchestration dependencies must not be nil")
	}
	return nil
}

func fail(statusPath, runID, phase string, cause error) error {
	statusErr := writeStatus(statusPath, status{RunID: runID, Phase: phase, Outcome: "failed", Error: cause.Error()})
	if statusErr != nil {
		return fmt.Errorf("%w (also publish failed production status: %v)", cause, statusErr)
	}
	return cause
}

func writeStatus(path string, value status) error {
	if strings.TrimSpace(value.RunID) == "" {
		return errors.New("production status run ID must not be empty")
	}
	if value.Outcome != "running" && value.Outcome != "failed" && value.Outcome != "completed" {
		return fmt.Errorf("unsupported production status outcome %q", value.Outcome)
	}
	if strings.TrimSpace(value.Phase) == "" {
		return errors.New("production status phase must not be empty")
	}
	value.UpdatedAt = time.Now().UTC()
	contents, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return fmt.Errorf("encode production status: %w", err)
	}
	contents = append(contents, '\n')
	directory := filepath.Dir(path)
	if err := os.MkdirAll(directory, 0o755); err != nil {
		return fmt.Errorf("create production status directory: %w", err)
	}
	temporary, err := os.CreateTemp(directory, "."+filepath.Base(path)+".tmp-*")
	if err != nil {
		return fmt.Errorf("create production status temporary file: %w", err)
	}
	temporaryPath := temporary.Name()
	defer func() { _ = os.Remove(temporaryPath) }()
	if err := temporary.Chmod(0o644); err != nil {
		_ = temporary.Close()
		return fmt.Errorf("set production status permissions: %w", err)
	}
	if _, err := temporary.Write(contents); err != nil {
		_ = temporary.Close()
		return fmt.Errorf("write production status: %w", err)
	}
	if err := temporary.Sync(); err != nil {
		_ = temporary.Close()
		return fmt.Errorf("sync production status: %w", err)
	}
	if err := temporary.Close(); err != nil {
		return fmt.Errorf("close production status: %w", err)
	}
	if err := os.Rename(temporaryPath, path); err != nil {
		return fmt.Errorf("publish production status: %w", err)
	}
	directoryHandle, err := os.Open(directory)
	if err != nil {
		return fmt.Errorf("open production status directory: %w", err)
	}
	defer func() { _ = directoryHandle.Close() }()
	if err := directoryHandle.Sync(); err != nil {
		return fmt.Errorf("sync production status directory: %w", err)
	}
	return nil
}

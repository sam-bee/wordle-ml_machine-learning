// Command train runs one fixed supervised-training proof stage.
package main

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/sam-bee/wordle-ml_machine-learning/proofrun"
)

type config struct {
	dataDir string
	runsDir string
	runID   string
	stage   string
	stopAt  int64
}

func main() {
	if err := run(os.Args[1:], os.Stdout, os.Stderr); err != nil {
		fmt.Fprintf(os.Stderr, "train: %v\n", err)
		os.Exit(1)
	}
}

func run(args []string, stdout, stderr io.Writer) error {
	config, err := parseConfig(args, stderr)
	if err != nil {
		return err
	}
	_, err = proofrun.Run(proofrun.Options{
		DataDir: config.dataDir,
		RunsDir: config.runsDir,
		RunID:   config.runID,
		Stage:   proofrun.Stage(config.stage),
		StopAt:  config.stopAt,
	}, stdout)
	return err
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
	flags := flag.NewFlagSet("train", flag.ContinueOnError)
	flags.SetOutput(stderr)
	flags.Usage = func() {
		fmt.Fprintln(stderr, "usage: train -run-id=<id> -stage=overfit|mini|full [flags]")
		flags.PrintDefaults()
	}
	var config config
	flags.StringVar(&config.dataDir, "data-dir", dataDir, "directory containing frozen vocabularies and imitation data")
	flags.StringVar(&config.runsDir, "runs-dir", runsDir, "directory containing self-contained proof runs")
	flags.StringVar(&config.runID, "run-id", "", "required stable proof-run identifier")
	flags.StringVar(&config.stage, "stage", "", "required fixed stage: overfit, mini, or full")
	flags.Int64Var(&config.stopAt, "stop-at", 0, "normal mini resume-test stop update (only 500)")
	if err := flags.Parse(args); err != nil {
		return config, err
	}
	if flags.NArg() != 0 {
		return config, fmt.Errorf("unexpected argument %q", flags.Arg(0))
	}
	if config.runID == "" {
		return config, errors.New("-run-id is required")
	}
	if config.stage == "" {
		return config, errors.New("-stage is required")
	}
	if _, err := proofrun.ConfigFor(proofrun.Stage(config.stage)); err != nil {
		return config, err
	}
	return config, nil
}

// Command evaluate runs fixed, post-checkpoint proof evaluations. It has no
// training controls and cannot select the sealed test split.
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

	"github.com/sam-bee/wordle-ml_machine-learning/proofeval"
)

type config struct {
	dataDir    string
	runsDir    string
	runID      string
	checkpoint string
	mode       string
}

func main() {
	if err := run(os.Args[1:], os.Stdout, os.Stderr); err != nil {
		fmt.Fprintf(os.Stderr, "evaluate: %v\n", err)
		os.Exit(1)
	}
}

func run(args []string, stdout, stderr io.Writer) error {
	config, err := parseConfig(args, stderr)
	if err != nil {
		return err
	}
	result, err := proofeval.Run(context.Background(), proofeval.Options{
		DataDir: config.dataDir, RunsDir: config.runsDir, RunID: config.runID,
		Checkpoint: proofeval.Checkpoint(config.checkpoint), Mode: proofeval.Mode(config.mode),
	})
	if err != nil {
		return err
	}
	encoded, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		return err
	}
	_, err = fmt.Fprintln(stdout, string(encoded))
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
	flags := flag.NewFlagSet("evaluate", flag.ContinueOnError)
	flags.SetOutput(stderr)
	flags.Usage = func() {
		fmt.Fprintln(stderr, "usage: evaluate -run-id=<id> -checkpoint=initial|latest|best -mode=games10|games100|ablations [flags]")
		flags.PrintDefaults()
	}
	var config config
	flags.StringVar(&config.dataDir, "data-dir", dataDir, "directory containing frozen vocabularies and imitation data")
	flags.StringVar(&config.runsDir, "runs-dir", runsDir, "directory containing self-contained proof runs")
	flags.StringVar(&config.runID, "run-id", "", "required stable proof-run identifier")
	flags.StringVar(&config.checkpoint, "checkpoint", "best", "checkpoint to reload: initial, latest, or best")
	flags.StringVar(&config.mode, "mode", "", "required fixed mode: games10, games100, or ablations")
	if err := flags.Parse(args); err != nil {
		return config, err
	}
	if flags.NArg() != 0 {
		return config, fmt.Errorf("unexpected argument %q", flags.Arg(0))
	}
	if config.runID == "" {
		return config, errors.New("-run-id is required")
	}
	if config.mode == "" {
		return config, errors.New("-mode is required")
	}
	if config.checkpoint != string(proofeval.Initial) && config.checkpoint != string(proofeval.Latest) && config.checkpoint != string(proofeval.Best) {
		return config, errors.New("-checkpoint must be initial, latest, or best")
	}
	if config.mode != string(proofeval.Games10) && config.mode != string(proofeval.Games100) && config.mode != string(proofeval.Ablations) {
		return config, errors.New("-mode must be games10, games100, or ablations")
	}
	return config, nil
}

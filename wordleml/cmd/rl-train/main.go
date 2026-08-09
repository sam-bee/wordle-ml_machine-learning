// Command rl-train runs the optional, bounded PPO experiment. It has no path
// to the sealed final-test population and never mutates its supervised input
// checkpoint.
package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/sam-bee/wordle-ml_machine-learning/rl/experiment"
)

type commandConfig struct {
	algorithm            string
	configPath           string
	dataDir              string
	supervisedCheckpoint string
	runDir               string
}

func main() {
	if err := run(os.Args[1:], os.Stdout, os.Stderr); err != nil {
		fmt.Fprintf(os.Stderr, "rl-train: %v\n", err)
		os.Exit(1)
	}
}

func run(args []string, stdout, stderr io.Writer) error {
	config, err := parseConfig(args, stderr)
	if err != nil {
		return err
	}
	result, err := experiment.Run(experiment.Options{
		ConfigPath: config.configPath, DataDir: config.dataDir,
		SupervisedCheckpoint: config.supervisedCheckpoint, RunDir: config.runDir,
	}, stdout)
	encoded, encodeErr := json.MarshalIndent(result, "", "  ")
	if encodeErr == nil {
		fmt.Fprintln(stdout, string(encoded))
	}
	if err != nil {
		return err
	}
	return encodeErr
}

func parseConfig(args []string, stderr io.Writer) (commandConfig, error) {
	dataDir := os.Getenv("WORDLEML_DATA_DIR")
	if dataDir == "" {
		dataDir = "../data"
	}
	flags := flag.NewFlagSet("rl-train", flag.ContinueOnError)
	flags.SetOutput(stderr)
	flags.Usage = func() {
		fmt.Fprintln(stderr, "usage: rl-train --algorithm=ppo --config=<path> --supervised-checkpoint=<path> --run-dir=<fresh-path> [flags]")
		flags.PrintDefaults()
	}
	var config commandConfig
	flags.StringVar(&config.algorithm, "algorithm", "ppo", "only supported algorithm: ppo")
	flags.StringVar(&config.configPath, "config", "", "required bounded PPO JSON configuration")
	flags.StringVar(&config.dataDir, "data-dir", dataDir, "directory containing frozen vocabularies and versioned RL manifest")
	flags.StringVar(&config.supervisedCheckpoint, "supervised-checkpoint", "", "required read-only supervised checkpoint directory")
	flags.StringVar(&config.runDir, "run-dir", "", "required fresh generated PPO run directory")
	if err := flags.Parse(args); err != nil {
		return config, err
	}
	if flags.NArg() != 0 {
		return config, fmt.Errorf("unexpected argument %q", flags.Arg(0))
	}
	if config.algorithm != "ppo" {
		return config, fmt.Errorf("unsupported algorithm %q; only ppo is implemented", config.algorithm)
	}
	if config.configPath == "" || config.supervisedCheckpoint == "" || config.runDir == "" {
		return config, errors.New("--config, --supervised-checkpoint, and --run-dir are required")
	}
	for name, path := range map[string]string{"config": config.configPath, "data directory": config.dataDir, "supervised checkpoint": config.supervisedCheckpoint, "run directory": config.runDir} {
		absolute, err := filepath.Abs(path)
		if err != nil {
			return config, fmt.Errorf("resolve %s: %w", name, err)
		}
		switch name {
		case "config":
			config.configPath = absolute
		case "data directory":
			config.dataDir = absolute
		case "supervised checkpoint":
			config.supervisedCheckpoint = absolute
		case "run directory":
			config.runDir = absolute
		}
	}
	return config, nil
}

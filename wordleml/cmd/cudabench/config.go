package main

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"

	"github.com/sam-bee/wordle-ml_machine-learning/cudacheck"
)

type config struct {
	dataDir    string
	modelDir   string
	report     string
	tolerance  float64
	warmup     int
	iterations int
	profile    bool
}

func parseConfig(args []string, stderr io.Writer) (config, error) {
	dataDir := os.Getenv("DATA_DIR")
	if dataDir == "" {
		dataDir = os.Getenv("WORDLEML_DATA_DIR")
	}
	if dataDir == "" {
		dataDir = "../data"
	}
	modelDir := os.Getenv("MODEL_DIR")
	if modelDir == "" {
		modelDir = os.Getenv("WORDLEML_CUDA_MODEL_DIR")
	}
	flags := flag.NewFlagSet("cudabench", flag.ContinueOnError)
	flags.SetOutput(stderr)
	flags.Usage = func() {
		fmt.Fprintln(stderr, "usage: cudabench -model-dir=<cuda-export> [flags]")
		flags.PrintDefaults()
	}
	var configuration config
	flags.StringVar(&configuration.dataDir, "data-dir", dataDir, "directory containing the frozen vocabularies")
	flags.StringVar(&configuration.modelDir, "model-dir", modelDir, "directory containing the CUDA artifact and golden vectors")
	flags.StringVar(&configuration.report, "report", os.Getenv("CUDA_REPORT"), "path for the JSON benchmark report (default MODEL_DIR/benchmark-report.json)")
	flags.Float64Var(&configuration.tolerance, "abs-tolerance", cudacheck.DefaultTolerance, "strict absolute logit tolerance")
	flags.IntVar(&configuration.warmup, "warmup", -1, "warm-up calls before measuring (default 20, or 2 in profile mode)")
	flags.IntVar(&configuration.iterations, "iterations", -1, "measured calls (default 200, or 10 in profile mode)")
	flags.BoolVar(&configuration.profile, "profile", false, "use short deterministic warm-up and measurement counts unless explicitly set")
	if err := flags.Parse(args); err != nil {
		return configuration, err
	}
	if flags.NArg() != 0 {
		return configuration, fmt.Errorf("unexpected argument %q", flags.Arg(0))
	}
	if configuration.warmup == -1 {
		configuration.warmup = 20
		if configuration.profile {
			configuration.warmup = 2
		}
	}
	if configuration.iterations == -1 {
		configuration.iterations = 200
		if configuration.profile {
			configuration.iterations = 10
		}
	}
	if strings.TrimSpace(configuration.dataDir) == "" {
		return configuration, errors.New("-data-dir must not be empty")
	}
	if strings.TrimSpace(configuration.modelDir) == "" {
		return configuration, errors.New("-model-dir is required (or set MODEL_DIR)")
	}
	if configuration.report == "" {
		configuration.report = filepath.Join(configuration.modelDir, "benchmark-report.json")
	}
	if strings.TrimSpace(configuration.report) == "" {
		return configuration, errors.New("-report must not be empty")
	}
	if configuration.tolerance <= 0 || math.IsNaN(configuration.tolerance) || math.IsInf(configuration.tolerance, 0) {
		return configuration, fmt.Errorf("-abs-tolerance must be finite and positive, got %g", configuration.tolerance)
	}
	if configuration.warmup < 0 || configuration.iterations <= 0 {
		return configuration, errors.New("-warmup must be non-negative and -iterations must be positive")
	}
	return configuration, nil
}

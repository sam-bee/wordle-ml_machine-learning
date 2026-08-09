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
	dataDir   string
	modelDir  string
	report    string
	tolerance float64
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
	reportPath := os.Getenv("CUDA_REPORT")
	flags := flag.NewFlagSet("cudaverify", flag.ContinueOnError)
	flags.SetOutput(stderr)
	flags.Usage = func() {
		fmt.Fprintln(stderr, "usage: cudaverify -model-dir=<cuda-export> [flags]")
		flags.PrintDefaults()
	}
	var configuration config
	flags.StringVar(&configuration.dataDir, "data-dir", dataDir, "directory containing the frozen vocabularies")
	flags.StringVar(&configuration.modelDir, "model-dir", modelDir, "directory containing the CUDA artifact and golden references")
	flags.StringVar(&configuration.report, "report", reportPath, "path for the JSON verification report (default MODEL_DIR/verification-report.json)")
	flags.Float64Var(&configuration.tolerance, "abs-tolerance", cudacheck.DefaultTolerance, "strict absolute logit tolerance")
	if err := flags.Parse(args); err != nil {
		return configuration, err
	}
	if flags.NArg() != 0 {
		return configuration, fmt.Errorf("unexpected argument %q", flags.Arg(0))
	}
	if strings.TrimSpace(configuration.dataDir) == "" {
		return configuration, errors.New("-data-dir must not be empty")
	}
	if strings.TrimSpace(configuration.modelDir) == "" {
		return configuration, errors.New("-model-dir is required (or set MODEL_DIR)")
	}
	if configuration.report == "" {
		configuration.report = filepath.Join(configuration.modelDir, "verification-report.json")
	}
	if strings.TrimSpace(configuration.report) == "" {
		return configuration, errors.New("-report must not be empty")
	}
	if configuration.tolerance <= 0 || math.IsNaN(configuration.tolerance) || math.IsInf(configuration.tolerance, 0) {
		return configuration, fmt.Errorf("-abs-tolerance must be finite and positive, got %g", configuration.tolerance)
	}
	return configuration, nil
}

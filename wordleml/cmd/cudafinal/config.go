//go:build cuda_cgo

package main

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"
)

type config struct {
	dataDir          string
	modelDir         string
	confirmFinalTest bool
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
	flags := flag.NewFlagSet("cudafinal", flag.ContinueOnError)
	flags.SetOutput(stderr)
	flags.Usage = func() {
		fmt.Fprintln(stderr, "usage: cudafinal -confirm-final-test -model-dir=<cuda-export> [flags]")
		flags.PrintDefaults()
	}
	var configuration config
	flags.StringVar(&configuration.dataDir, "data-dir", dataDir, "directory containing frozen vocabularies, including the intentionally opened final-test list")
	flags.StringVar(&configuration.modelDir, "model-dir", modelDir, "directory containing the validated CUDA model export")
	flags.BoolVar(&configuration.confirmFinalTest, "confirm-final-test", false, "required explicit acknowledgement that this reads the final-test word list once")
	if err := flags.Parse(args); err != nil {
		return config{}, err
	}
	if flags.NArg() != 0 {
		return config{}, fmt.Errorf("unexpected argument %q", flags.Arg(0))
	}
	if !configuration.confirmFinalTest {
		return config{}, errors.New("refusing to read final-test vocabulary without -confirm-final-test")
	}
	if strings.TrimSpace(configuration.dataDir) == "" || strings.TrimSpace(configuration.modelDir) == "" {
		return config{}, errors.New("-data-dir and -model-dir must not be empty")
	}
	return configuration, nil
}

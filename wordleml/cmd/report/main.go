// Command report validates three completed proof runs and writes their Markdown report.
package main

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/sam-bee/wordle-ml_machine-learning/proofreport"
)

type config struct{ runsDir, overfitRunID, miniRunID, fullRunID, outputPath string }

func main() {
	if err := run(os.Args[1:], os.Stdout, os.Stderr); err != nil {
		fmt.Fprintf(os.Stderr, "report: %v\n", err)
		os.Exit(1)
	}
}

func run(args []string, stdout, stderr io.Writer) error {
	configuration, err := parseConfig(args, stderr)
	if err != nil {
		return err
	}
	report, err := proofreport.Write(proofreport.Options{RunsDir: configuration.runsDir, OverfitRunID: configuration.overfitRunID, MiniRunID: configuration.miniRunID, FullRunID: configuration.fullRunID, OutputPath: configuration.outputPath})
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(stdout, "wrote validated proof report (%d rows) to %s\n", len(report.Stages), configuration.outputPath)
	return err
}

func parseConfig(args []string, stderr io.Writer) (config, error) {
	runsDir := os.Getenv("WORDLEML_RUNS_DIR")
	if runsDir == "" {
		runsDir = "../runs"
	}
	flags := flag.NewFlagSet("report", flag.ContinueOnError)
	flags.SetOutput(stderr)
	flags.Usage = func() {
		fmt.Fprintln(stderr, "usage: report -overfit-run-id=<id> -mini-run-id=<id> -full-run-id=<id> -output=<path> [flags]")
		flags.PrintDefaults()
	}
	var value config
	flags.StringVar(&value.runsDir, "runs-dir", runsDir, "directory containing completed proof runs")
	flags.StringVar(&value.overfitRunID, "overfit-run-id", "", "required overfit proof run ID")
	flags.StringVar(&value.miniRunID, "mini-run-id", "", "required mini proof run ID")
	flags.StringVar(&value.fullRunID, "full-run-id", "", "required full proof run ID")
	flags.StringVar(&value.outputPath, "output", filepath.Join("..", proofreport.DefaultOutput), "Markdown report path")
	if err := flags.Parse(args); err != nil {
		return value, err
	}
	if flags.NArg() != 0 {
		return value, fmt.Errorf("unexpected argument %q", flags.Arg(0))
	}
	if value.overfitRunID == "" || value.miniRunID == "" || value.fullRunID == "" {
		return value, errors.New("-overfit-run-id, -mini-run-id, and -full-run-id are required")
	}
	if value.outputPath == "" {
		return value, errors.New("-output is required")
	}
	return value, nil
}

// Command serve loads one completed run's best checkpoint, permits selecting
// other completed runs, and serves validation games from warm CUDA inference.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"github.com/sam-bee/wordle-ml_machine-learning/serving"
)

type config struct {
	address string
	dataDir string
	runsDir string
	runID   string
}

func main() {
	if err := run(os.Args[1:], os.Stdout, os.Stderr); err != nil {
		fmt.Fprintf(os.Stderr, "serve: %v\n", err)
		os.Exit(1)
	}
}

func run(args []string, stdout, stderr io.Writer) error {
	config, err := parseConfig(args, stderr)
	if err != nil {
		return err
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	manager, err := serving.LoadManager(ctx, serving.Options{DataDir: config.dataDir, RunsDir: config.runsDir, RunID: config.runID})
	if err != nil {
		return err
	}
	defer manager.Close()
	handler, err := serving.NewHandler(manager)
	if err != nil {
		return err
	}
	server := &http.Server{
		Addr:              config.address,
		Handler:           handler,
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       10 * time.Second,
		WriteTimeout:      2 * time.Minute,
		IdleTimeout:       60 * time.Second,
	}
	go func() {
		<-ctx.Done()
		shutdownContext, cancel := context.WithTimeout(context.Background(), 45*time.Second)
		defer cancel()
		_ = server.Shutdown(shutdownContext)
	}()
	identity := manager.ModelIdentity()
	fmt.Fprintf(stdout, "serving run=%s checkpoint=%s update=%d address=%s\n", identity.RunID, identity.Checkpoint, identity.Update, config.address)
	if err := server.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		return err
	}
	return nil
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
	address := os.Getenv("WORDLEML_INFERENCE_ADDR")
	if address == "" {
		address = ":8090"
	}
	runID := os.Getenv("WORDLEML_INFERENCE_RUN_ID")
	flags := flag.NewFlagSet("serve", flag.ContinueOnError)
	flags.SetOutput(stderr)
	flags.Usage = func() {
		fmt.Fprintln(stderr, "usage: serve -run-id=<completed-run-id> [flags]")
		flags.PrintDefaults()
	}
	var config config
	flags.StringVar(&config.address, "addr", address, "HTTP listen address")
	flags.StringVar(&config.dataDir, "data-dir", dataDir, "directory containing frozen vocabularies")
	flags.StringVar(&config.runsDir, "runs-dir", runsDir, "directory containing self-contained training runs")
	flags.StringVar(&config.runID, "run-id", runID, "required initially selected completed run identifier")
	if err := flags.Parse(args); err != nil {
		return config, err
	}
	if flags.NArg() != 0 {
		return config, fmt.Errorf("unexpected argument %q", flags.Arg(0))
	}
	if strings.TrimSpace(config.runID) == "" {
		return config, errors.New("-run-id is required")
	}
	if strings.TrimSpace(config.address) == "" {
		return config, errors.New("-addr must not be empty")
	}
	return config, nil
}

// Command train runs a deliberately bounded supervised Wordle policy update.
package main

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"

	"github.com/gomlx/compute"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/sam-bee/wordle-ml_machine-learning/imitationdata"
	"github.com/sam-bee/wordle-ml_machine-learning/policy"
	"github.com/sam-bee/wordle-ml_machine-learning/supervised"
	"github.com/sam-bee/wordle-ml_machine-learning/tensorboard"
	"github.com/sam-bee/wordle-ml_machine-learning/vocabulary"
)

const defaultSeed int64 = 20260808

type config struct {
	vocabularyDir  string
	imitationDir   string
	checkpointDir  string
	tensorboardDir string
	batchSize      int
	learningRate   float64
	seed           int64
	steps          int
}

func main() {
	if err := run(os.Args[1:], os.Stdout, os.Stderr); err != nil {
		fmt.Fprintf(os.Stderr, "train: %v\n", err)
		os.Exit(1)
	}
}

func run(args []string, stdout, stderr io.Writer) error {
	cfg, err := parseConfig(args, stderr)
	if err != nil {
		return err
	}
	printConfig(stdout, cfg)
	if cfg.steps == 0 {
		return nil
	}

	v, err := vocabulary.Load(cfg.vocabularyDir)
	if err != nil {
		return fmt.Errorf("load vocabulary: %w", err)
	}
	trainData, err := imitationdata.Load(v, cfg.imitationDir, imitationdata.Train)
	if err != nil {
		return fmt.Errorf("load training data: %w", err)
	}
	if cfg.batchSize > trainData.Len() {
		return fmt.Errorf("batch-size %d exceeds %d training records", cfg.batchSize, trainData.Len())
	}
	validationData, err := imitationdata.Load(v, cfg.imitationDir, imitationdata.Validation)
	if err != nil {
		return fmt.Errorf("load validation data: %w", err)
	}

	backend, err := compute.NewWithConfig("xla:cuda")
	if err != nil {
		return fmt.Errorf("create CUDA backend: %w", err)
	}
	defer backend.Finalize()

	session, err := supervised.New(supervised.Config{
		Policy: policy.Config{
			NumSolutions: vocabulary.NumSolutions,
			NumActions:   vocabulary.NumActions,
		},
		LearningRate: cfg.learningRate,
		Seed:         cfg.seed,
	}, backend)
	if err != nil {
		return fmt.Errorf("create supervised session: %w", err)
	}
	checkpoints, err := supervised.NewCheckpoint(session.Store, cfg.checkpointDir)
	if err != nil {
		return fmt.Errorf("open checkpoint: %w", err)
	}
	events, err := tensorboard.New(cfg.tensorboardDir)
	if err != nil {
		return fmt.Errorf("create TensorBoard writer: %w", err)
	}
	defer events.Close()

	if err := session.Trainer.ResetTrainMetrics(); err != nil {
		return fmt.Errorf("reset training metrics: %w", err)
	}
	if err := trainSteps(session, trainData, backend, cfg, events); err != nil {
		return err
	}

	validationBatches := imitationdata.Batch(
		backend,
		validationData.Dataset(cfg.seed),
		cfg.batchSize,
		false,
	)
	validationMetrics, err := session.Trainer.Eval(validationBatches)
	if err != nil {
		return fmt.Errorf("evaluate validation data: %w", err)
	}
	if err := writeEvaluationMetrics(events, "validation", session.Trainer.GlobalStep(), validationMetrics); err != nil {
		return err
	}
	if err := checkpoints.Save(); err != nil {
		return fmt.Errorf("save checkpoint: %w", err)
	}
	fmt.Fprintf(stdout, "completed_steps=%d global_step=%d\n", cfg.steps, session.Trainer.GlobalStep())
	return nil
}

func parseConfig(args []string, stderr io.Writer) (config, error) {
	dataDir := os.Getenv("WORDLEML_DATA_DIR")
	if dataDir == "" {
		dataDir = "../data"
	}
	flags := flag.NewFlagSet("train", flag.ContinueOnError)
	flags.SetOutput(stderr)
	flags.Usage = func() {
		fmt.Fprintln(stderr, "usage: train [flags]")
		flags.PrintDefaults()
	}
	cfg := config{}
	flags.StringVar(&cfg.vocabularyDir, "data-dir", dataDir, "directory containing frozen word lists")
	flags.StringVar(&cfg.imitationDir, "imitation-dir", "", "directory containing WDIT files (default: <data-dir>/imitation)")
	flags.StringVar(&cfg.checkpointDir, "checkpoint-dir", "", "checkpoint directory (default: <data-dir>/checkpoints/first-run)")
	flags.StringVar(&cfg.tensorboardDir, "tensorboard-dir", "", "TensorBoard event directory (default: <data-dir>/tensorboard/first-run)")
	flags.IntVar(&cfg.batchSize, "batch-size", 32, "examples per training batch")
	flags.Float64Var(&cfg.learningRate, "learning-rate", 0.001, "Adam learning rate")
	flags.Int64Var(&cfg.seed, "seed", defaultSeed, "non-zero deterministic seed")
	flags.IntVar(&cfg.steps, "steps", 0, "bounded training steps; zero only prints configuration")
	if err := flags.Parse(args); err != nil {
		return config{}, err
	}
	if flags.NArg() != 0 {
		return config{}, fmt.Errorf("unexpected argument %q", flags.Arg(0))
	}
	if cfg.imitationDir == "" {
		cfg.imitationDir = filepath.Join(cfg.vocabularyDir, "imitation")
	}
	if cfg.checkpointDir == "" {
		cfg.checkpointDir = filepath.Join(cfg.vocabularyDir, "checkpoints", "first-run")
	}
	if cfg.tensorboardDir == "" {
		cfg.tensorboardDir = filepath.Join(cfg.vocabularyDir, "tensorboard", "first-run")
	}
	if cfg.batchSize <= 0 {
		return config{}, errors.New("batch-size must be positive")
	}
	if cfg.learningRate <= 0 || math.IsNaN(cfg.learningRate) || math.IsInf(cfg.learningRate, 0) {
		return config{}, errors.New("learning-rate must be finite and positive")
	}
	if cfg.seed == 0 {
		return config{}, errors.New("seed must not be zero")
	}
	if cfg.steps < 0 {
		return config{}, errors.New("steps must not be negative")
	}
	return cfg, nil
}

func printConfig(w io.Writer, cfg config) {
	fmt.Fprintf(w, "data_dir=%s\n", cfg.vocabularyDir)
	fmt.Fprintf(w, "imitation_dir=%s\n", cfg.imitationDir)
	fmt.Fprintf(w, "checkpoint_dir=%s\n", cfg.checkpointDir)
	fmt.Fprintf(w, "tensorboard_dir=%s\n", cfg.tensorboardDir)
	fmt.Fprintf(w, "batch_size=%d learning_rate=%g seed=%d steps=%d\n", cfg.batchSize, cfg.learningRate, cfg.seed, cfg.steps)
}

func trainSteps(session *supervised.Session, data *imitationdata.Data, backend compute.Backend, cfg config, events *tensorboard.Writer) error {
	completed := 0
	for epoch := int64(0); completed < cfg.steps; epoch++ {
		batches := imitationdata.Batch(backend, data.Dataset(cfg.seed+epoch), cfg.batchSize, true)
		for batch, err := range batches.Iter() {
			if err != nil {
				return fmt.Errorf("iterate training data: %w", err)
			}
			metrics, err := session.Trainer.TrainStep(batch)
			if err != nil {
				return fmt.Errorf("training step %d: %w", completed+1, err)
			}
			if err := writeTrainingMetrics(events, session.Trainer.GlobalStep(), metrics); err != nil {
				return err
			}
			completed++
			if completed == cfg.steps {
				return nil
			}
		}
	}
	return nil
}

func writeTrainingMetrics(events *tensorboard.Writer, step int64, metrics []*tensors.Tensor) error {
	defer finalizeMetrics(metrics)
	if len(metrics) != 5 {
		return fmt.Errorf("training metrics: got %d values, want loss, moving loss, top1, top5, top16", len(metrics))
	}
	return writeSelectedMetrics(events, "train", step, []*tensors.Tensor{metrics[0], metrics[2], metrics[3], metrics[4]})
}

func writeEvaluationMetrics(events *tensorboard.Writer, prefix string, step int64, metrics []*tensors.Tensor) error {
	defer finalizeMetrics(metrics)
	if len(metrics) != 4 {
		return fmt.Errorf("%s metrics: got %d values, want loss, top1, top5, top16", prefix, len(metrics))
	}
	return writeSelectedMetrics(events, prefix, step, metrics)
}

func writeSelectedMetrics(events *tensorboard.Writer, prefix string, step int64, metrics []*tensors.Tensor) error {
	names := [...]string{"loss", "top1", "top5", "top16"}
	scalars := make([]tensorboard.Scalar, len(metrics))
	for index, metric := range metrics {
		value, err := scalar(metric)
		if err != nil {
			return fmt.Errorf("%s %s: %w", prefix, names[index], err)
		}
		scalars[index] = tensorboard.Scalar{Tag: prefix + "/" + names[index], Value: value}
	}
	if err := events.WriteScalars(step, scalars...); err != nil {
		return fmt.Errorf("write %s metrics: %w", prefix, err)
	}
	return nil
}

func finalizeMetrics(metrics []*tensors.Tensor) {
	for _, metric := range metrics {
		if metric != nil {
			_ = metric.FinalizeAll()
		}
	}
}

func scalar(tensor *tensors.Tensor) (float32, error) {
	if tensor == nil || tensor.Size() != 1 {
		return 0, errors.New("metric is not a scalar tensor")
	}
	values, err := tensors.CopyFlatData[float32](tensor)
	if err != nil {
		return 0, err
	}
	return values[0], nil
}

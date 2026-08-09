// Package checkpoints owns the filesystem layout and durable metadata for a
// bounded PPO experiment.
//
// It deliberately does not know how GoMLX checkpoints are encoded or copied.
// The runner writes model payloads into the actor, critic, and actor-only
// directories. This package only publishes the small, durable records which
// say which complete candidate has been accepted. In particular, it never
// reads, copies, or overwrites the supervised actor checkpoint.
package checkpoints

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

const (
	configFilename                    = "config.json"
	metadataFilename                  = "metadata.json"
	acceptedFilename                  = "accepted.json"
	evaluationFilename                = "evaluation.json"
	iterationStateFilename            = "iteration-state.json"
	supervisedBaselineDirectory       = "supervised-baseline"
	ppoDirectory                      = "ppo"
	eventsDirectory                   = "events"
	bestDirectory                     = "best"
	actorCriticDirectory              = "actor-critic"
	actorDirectory                    = "actor"
	criticDirectory                   = "critic"
	actorOnlyDirectory                = "actor-only"
	maximumRunIDLength                = 128
	maximumIteration                  = 999999
	maximumSmallMetadataBytes   int64 = 16 << 20
)

var (
	// ErrAcceptedNotFound means this PPO run has not accepted a candidate
	// checkpoint yet. The supervised baseline remains the deployment fallback.
	ErrAcceptedNotFound = errors.New("accepted PPO checkpoint not found")

	// ErrIterationStateNotFound means a candidate has not published the state
	// needed to resume it yet.
	ErrIterationStateNotFound = errors.New("PPO iteration state not found")
)

// Layout names the artifacts for one PPO experiment. Root is the common
// checkpoints directory, not the PPO run directory. All paths are derived
// from a validated run ID; callers never supply checkpoint-relative paths.
//
// The on-disk structure is:
//
//	Root/
//	  supervised-baseline/metadata.json
//	  ppo/<ID>/
//	    config.json metadata.json events/ accepted.json best/
//	    iter-NNN/{actor-critic/{actor,critic},actor-only,evaluation.json,...}
type Layout struct {
	Root string
	ID   string

	SupervisedBaselineDir          string
	SupervisedBaselineMetadataPath string
	PPODir                         string
	Dir                            string
	ConfigPath                     string
	MetadataPath                   string
	EventsDir                      string
	AcceptedPath                   string
	BestDir                        string
	BestAcceptedPath               string
	BestEvaluationPath             string
	BestIterationStatePath         string
}

// IterationLayout identifies one fresh PPO candidate directory. The model
// runner owns all heavyweight contents under ActorDir, CriticDir, and
// ActorOnlyDir; this package only creates the directories for it.
type IterationLayout struct {
	Run       Layout
	Iteration int
	Name      string
	Dir       string

	ActorCriticDir string
	ActorDir       string
	CriticDir      string
	ActorOnlyDir   string
	EvaluationPath string
	StatePath      string
}

// New returns a planned layout without creating anything.
func New(root, runID string) (Layout, error) {
	if err := ValidateRunID(runID); err != nil {
		return Layout{}, err
	}
	if strings.TrimSpace(root) == "" {
		return Layout{}, errors.New("checkpoints root must not be empty")
	}
	absoluteRoot, err := filepath.Abs(root)
	if err != nil {
		return Layout{}, fmt.Errorf("make checkpoints root absolute: %w", err)
	}
	absoluteRoot = filepath.Clean(absoluteRoot)
	ppoRoot := filepath.Join(absoluteRoot, ppoDirectory)
	runDir := filepath.Join(ppoRoot, runID)
	if !isPathWithin(absoluteRoot, ppoRoot) || !isPathWithin(ppoRoot, runDir) {
		return Layout{}, fmt.Errorf("PPO run ID %q escapes checkpoints root", runID)
	}
	baselineDir := filepath.Join(absoluteRoot, supervisedBaselineDirectory)
	return Layout{
		Root:                           absoluteRoot,
		ID:                             runID,
		SupervisedBaselineDir:          baselineDir,
		SupervisedBaselineMetadataPath: filepath.Join(baselineDir, metadataFilename),
		PPODir:                         ppoRoot,
		Dir:                            runDir,
		ConfigPath:                     filepath.Join(runDir, configFilename),
		MetadataPath:                   filepath.Join(runDir, metadataFilename),
		EventsDir:                      filepath.Join(runDir, eventsDirectory),
		AcceptedPath:                   filepath.Join(runDir, acceptedFilename),
		BestDir:                        filepath.Join(runDir, bestDirectory),
		BestAcceptedPath:               filepath.Join(runDir, bestDirectory, acceptedFilename),
		BestEvaluationPath:             filepath.Join(runDir, bestDirectory, evaluationFilename),
		BestIterationStatePath:         filepath.Join(runDir, bestDirectory, iterationStateFilename),
	}, nil
}

// Create creates a fresh PPO run directory. It never reuses a run ID. It may
// create the empty common supervised-baseline directory, but it never writes
// model data there; WriteSupervisedBaselineMetadata is the only writer for
// its small, immutable identity record.
func Create(root, runID string) (Layout, error) {
	layout, err := New(root, runID)
	if err != nil {
		return Layout{}, err
	}
	if err := ensureDirectory(layout.Root); err != nil {
		return Layout{}, fmt.Errorf("create checkpoints root %q: %w", layout.Root, err)
	}
	for _, directory := range []string{layout.SupervisedBaselineDir, layout.PPODir} {
		if err := ensureDirectory(directory); err != nil {
			return Layout{}, fmt.Errorf("create layout directory %q: %w", directory, err)
		}
	}
	if err := os.Mkdir(layout.Dir, 0o755); err != nil {
		if errors.Is(err, os.ErrExist) {
			return Layout{}, fmt.Errorf("PPO run directory %q already exists", layout.Dir)
		}
		return Layout{}, fmt.Errorf("create PPO run directory %q: %w", layout.Dir, err)
	}
	for _, directory := range []string{layout.EventsDir, layout.BestDir} {
		if err := os.Mkdir(directory, 0o755); err != nil {
			return Layout{}, fmt.Errorf("create PPO run directory %q: %w", directory, err)
		}
	}
	return layout, nil
}

// Open returns an existing PPO run. It does not require an accepted
// checkpoint, since an interrupted pilot may not have reached one.
func Open(root, runID string) (Layout, error) {
	layout, err := New(root, runID)
	if err != nil {
		return Layout{}, err
	}
	if err := requireDirectory(layout.Dir); err != nil {
		return Layout{}, fmt.Errorf("open PPO run directory %q: %w", layout.Dir, err)
	}
	return layout, nil
}

// Iteration returns a planned candidate path without creating it.
func (layout Layout) Iteration(iteration int) (IterationLayout, error) {
	if err := ValidateIteration(iteration); err != nil {
		return IterationLayout{}, err
	}
	name := iterationName(iteration)
	directory := filepath.Join(layout.Dir, name)
	if !isPathWithin(layout.Dir, directory) {
		return IterationLayout{}, fmt.Errorf("iteration %d escapes PPO run directory", iteration)
	}
	actorCriticDir := filepath.Join(directory, actorCriticDirectory)
	return IterationLayout{
		Run:            layout,
		Iteration:      iteration,
		Name:           name,
		Dir:            directory,
		ActorCriticDir: actorCriticDir,
		ActorDir:       filepath.Join(actorCriticDir, actorDirectory),
		CriticDir:      filepath.Join(actorCriticDir, criticDirectory),
		ActorOnlyDir:   filepath.Join(directory, actorOnlyDirectory),
		EvaluationPath: filepath.Join(directory, evaluationFilename),
		StatePath:      filepath.Join(directory, iterationStateFilename),
	}, nil
}

// CreateIteration makes a fresh candidate directory and the exact checkpoint
// destinations the runner may populate. Existing candidates are never reused
// because their trajectories and old-policy statistics are immutable inputs to
// that iteration.
func (layout Layout) CreateIteration(iteration int) (IterationLayout, error) {
	entry, err := layout.Iteration(iteration)
	if err != nil {
		return IterationLayout{}, err
	}
	if err := requireDirectory(layout.Dir); err != nil {
		return IterationLayout{}, fmt.Errorf("open PPO run before making iteration: %w", err)
	}
	if err := os.Mkdir(entry.Dir, 0o755); err != nil {
		if errors.Is(err, os.ErrExist) {
			return IterationLayout{}, fmt.Errorf("PPO iteration directory %q already exists", entry.Dir)
		}
		return IterationLayout{}, fmt.Errorf("create PPO iteration directory %q: %w", entry.Dir, err)
	}
	for _, directory := range []string{entry.ActorCriticDir, entry.ActorDir, entry.CriticDir, entry.ActorOnlyDir} {
		if err := os.Mkdir(directory, 0o755); err != nil {
			return IterationLayout{}, fmt.Errorf("create PPO iteration directory %q: %w", directory, err)
		}
	}
	return entry, nil
}

// OpenIteration returns an existing candidate layout.
func (layout Layout) OpenIteration(iteration int) (IterationLayout, error) {
	entry, err := layout.Iteration(iteration)
	if err != nil {
		return IterationLayout{}, err
	}
	if err := requireDirectory(entry.Dir); err != nil {
		return IterationLayout{}, fmt.Errorf("open PPO iteration directory %q: %w", entry.Dir, err)
	}
	return entry, nil
}

// ValidateRunID accepts portable directory names only. Dot paths, separators,
// whitespace, and shell-sensitive punctuation are intentionally rejected.
func ValidateRunID(runID string) error {
	if runID == "" || len(runID) > maximumRunIDLength {
		return fmt.Errorf("run ID must contain 1 to %d characters", maximumRunIDLength)
	}
	if runID == "." || runID == ".." || strings.HasPrefix(runID, ".") {
		return fmt.Errorf("run ID %q is not a safe directory name", runID)
	}
	for _, character := range runID {
		if (character >= 'a' && character <= 'z') ||
			(character >= 'A' && character <= 'Z') ||
			(character >= '0' && character <= '9') ||
			character == '-' || character == '_' || character == '.' {
			continue
		}
		return fmt.Errorf("run ID %q contains unsupported character %q", runID, character)
	}
	return nil
}

// ValidateIteration bounds the human-readable iter-NNN namespace and rejects
// values which could indicate an accidental counter underflow or corruption.
func ValidateIteration(iteration int) error {
	if iteration < 0 || iteration > maximumIteration {
		return fmt.Errorf("PPO iteration must be in 0..%d, got %d", maximumIteration, iteration)
	}
	return nil
}

func iterationName(iteration int) string {
	return fmt.Sprintf("iter-%03d", iteration)
}

func ensureDirectory(directory string) error {
	if err := os.MkdirAll(directory, 0o755); err != nil {
		return err
	}
	return requireDirectory(directory)
}

func requireDirectory(directory string) error {
	info, err := os.Lstat(directory)
	if err != nil {
		return err
	}
	if info.Mode()&os.ModeSymlink != 0 {
		return fmt.Errorf("refusing symbolic-link directory")
	}
	if !info.IsDir() {
		return errors.New("path is not a directory")
	}
	return nil
}

func isPathWithin(root, path string) bool {
	relative, err := filepath.Rel(root, path)
	if err != nil {
		return false
	}
	return relative != ".." && !strings.HasPrefix(relative, ".."+string(filepath.Separator)) && !filepath.IsAbs(relative)
}

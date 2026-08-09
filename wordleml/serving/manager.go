package serving

import (
	"context"
	"errors"
	"fmt"
	"os"
	"sort"
	"strings"
	"sync"

	"github.com/sam-bee/wordle-ml_machine-learning/inferenceapi"
	"github.com/sam-bee/wordle-ml_machine-learning/proofeval"
	"github.com/sam-bee/wordle-ml_machine-learning/proofrun"
	"github.com/sam-bee/wordle-ml_machine-learning/runstate"
)

// ErrModelNotFound means the requested run is not a completed, compatible
// training run currently available beneath the configured runs directory.
var ErrModelNotFound = inferenceapi.ErrModelNotFound

// ModelSummary describes one best checkpoint which can be selected without
// first loading it onto the CUDA device.
type ModelSummary struct {
	RunID          string         `json:"run_id"`
	Stage          proofrun.Stage `json:"stage"`
	Checkpoint     string         `json:"checkpoint"`
	Update         int64          `json:"update"`
	TrainingCommit string         `json:"training_commit"`
}

type managedRuntime interface {
	Player
	Close()
}

type runtimeLoader func(context.Context, Options) (managedRuntime, error)
type modelCatalogue func() ([]ModelSummary, error)

// Manager owns the currently active runtime and swaps it only after a newly
// selected checkpoint has loaded and warmed successfully. Existing game
// requests can finish against the old runtime while the replacement loads.
type Manager struct {
	options   Options
	load      runtimeLoader
	catalogue modelCatalogue

	switchMu sync.Mutex
	mu       sync.RWMutex
	current  managedRuntime
	closed   bool
}

// LoadManager discovers selectable runs and loads the configured initial run.
func LoadManager(ctx context.Context, options Options) (*Manager, error) {
	if strings.TrimSpace(options.RunID) == "" {
		return nil, errors.New("initial run ID is required")
	}
	catalogue := func() ([]ModelSummary, error) {
		return DiscoverModels(options.DataDir, options.RunsDir)
	}
	models, err := catalogue()
	if err != nil {
		return nil, err
	}
	if !containsModel(models, options.RunID) {
		return nil, fmt.Errorf("initial run %q: %w", options.RunID, ErrModelNotFound)
	}
	loader := func(ctx context.Context, selected Options) (managedRuntime, error) {
		return Load(ctx, selected)
	}
	runtime, err := loader(ctx, options)
	if err != nil {
		return nil, err
	}
	return &Manager{options: options, load: loader, catalogue: catalogue, current: runtime}, nil
}

// DiscoverModels returns completed, passed runs whose best checkpoints and
// immutable inputs are present and compatible with the configured data.
// Unfinished directories and non-run files are deliberately omitted.
func DiscoverModels(dataDir, runsDir string) ([]ModelSummary, error) {
	if strings.TrimSpace(dataDir) == "" || strings.TrimSpace(runsDir) == "" {
		return nil, errors.New("data and runs directories are required")
	}
	entries, err := os.ReadDir(runsDir)
	if err != nil {
		return nil, fmt.Errorf("read runs directory: %w", err)
	}
	models := make([]ModelSummary, 0, len(entries))
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		model, err := inspectModel(dataDir, runsDir, entry.Name())
		if err != nil {
			continue
		}
		models = append(models, model)
	}
	sort.Slice(models, func(first, second int) bool {
		return models[first].RunID < models[second].RunID
	})
	return models, nil
}

func inspectModel(dataDir, runsDir, runID string) (ModelSummary, error) {
	layout, err := runstate.Open(runsDir, runID)
	if err != nil {
		return ModelSummary{}, err
	}
	config, err := proofeval.ReadConfig(layout)
	if err != nil {
		return ModelSummary{}, err
	}
	result, err := readPassedResult(layout, config)
	if err != nil {
		return ModelSummary{}, err
	}
	manifest, err := readCompatibleManifest(layout, dataDir, config)
	if err != nil {
		return ModelSummary{}, err
	}
	checkpointFiles, err := os.ReadDir(layout.BestCheckpointDir)
	if err != nil || len(checkpointFiles) == 0 {
		return ModelSummary{}, errors.New("best checkpoint is absent")
	}
	return ModelSummary{
		RunID:          runID,
		Stage:          config.Stage,
		Checkpoint:     string(proofeval.Best),
		Update:         result.BestValidationStep,
		TrainingCommit: manifest.Repositories.MachineLearning.Commit,
	}, nil
}

// AvailableModels re-scans the runs directory so a newly completed run can be
// selected without restarting the inference service.
func (manager *Manager) AvailableModels() ([]ModelSummary, error) {
	if manager == nil || manager.catalogue == nil {
		return nil, errors.New("model manager is not ready")
	}
	return manager.catalogue()
}

// SelectModel loads and warms the requested best checkpoint, then atomically
// makes it active. A failed load leaves the current runtime untouched.
func (manager *Manager) SelectModel(ctx context.Context, runID string) (ModelIdentity, error) {
	if manager == nil {
		return ModelIdentity{}, errors.New("model manager is not ready")
	}
	manager.switchMu.Lock()
	defer manager.switchMu.Unlock()

	manager.mu.RLock()
	if manager.closed || manager.current == nil {
		manager.mu.RUnlock()
		return ModelIdentity{}, errors.New("model manager is closed")
	}
	identity := manager.current.ModelIdentity()
	manager.mu.RUnlock()
	if runID == identity.RunID {
		return identity, nil
	}
	models, err := manager.AvailableModels()
	if err != nil {
		return ModelIdentity{}, err
	}
	if !containsModel(models, runID) {
		return ModelIdentity{}, fmt.Errorf("run %q: %w", runID, ErrModelNotFound)
	}
	selected := manager.options
	selected.RunID = runID
	replacement, err := manager.load(ctx, selected)
	if err != nil {
		return ModelIdentity{}, fmt.Errorf("load run %q: %w", runID, err)
	}

	manager.mu.Lock()
	if manager.closed || manager.current == nil {
		manager.mu.Unlock()
		replacement.Close()
		return ModelIdentity{}, errors.New("model manager is closed")
	}
	previous := manager.current
	manager.current = replacement
	identity = replacement.ModelIdentity()
	manager.mu.Unlock()
	previous.Close()
	return identity, nil
}

func containsModel(models []ModelSummary, runID string) bool {
	for _, model := range models {
		if model.RunID == runID {
			return true
		}
	}
	return false
}

// ModelIdentity returns the currently active checkpoint identity.
func (manager *Manager) ModelIdentity() ModelIdentity {
	manager.mu.RLock()
	defer manager.mu.RUnlock()
	if manager.current == nil {
		return ModelIdentity{}
	}
	return manager.current.ModelIdentity()
}

// ValidationSolutions returns the fixed validation population accepted by
// every selectable model.
func (manager *Manager) ValidationSolutions() []string {
	manager.mu.RLock()
	defer manager.mu.RUnlock()
	if manager.current == nil {
		return nil
	}
	return manager.current.ValidationSolutions()
}

// PlayGame holds a read lock for the complete game and captures its model
// identity before a selected replacement can become active.
func (manager *Manager) PlayGame(ctx context.Context, solution string) (GameResponse, error) {
	manager.mu.RLock()
	defer manager.mu.RUnlock()
	if manager.current == nil {
		return GameResponse{}, errors.New("model manager is not ready")
	}
	game, err := manager.current.Play(ctx, solution)
	if err != nil {
		return GameResponse{}, err
	}
	return GameResponse{Model: manager.current.ModelIdentity(), GameResult: game}, nil
}

// Close prevents further switches or games and releases the active CUDA
// runtime after any in-flight game has completed.
func (manager *Manager) Close() {
	if manager == nil {
		return
	}
	manager.switchMu.Lock()
	defer manager.switchMu.Unlock()
	manager.mu.Lock()
	if manager.closed {
		manager.mu.Unlock()
		return
	}
	manager.closed = true
	current := manager.current
	manager.current = nil
	manager.mu.Unlock()
	if current != nil {
		current.Close()
	}
}

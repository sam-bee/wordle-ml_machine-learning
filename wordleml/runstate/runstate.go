// Package runstate owns the durable, non-model state for one supervised run.
//
// GoMLX checkpoints hold the model and optimizer variables. This package holds
// the surrounding run identity and the dataset cursor needed to continue that
// checkpoint as the same logical training run.
package runstate

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"strings"

	"github.com/gomlx/gomlx/ml/model"
)

const (
	configFilename          = "config.json"
	metadataFilename        = "metadata.json"
	stateFilename           = "run-state.json"
	finalMetricsFilename    = "final-metrics.json"
	validationGamesFilename = "validation-games.jsonl"
	trainingLogFilename     = "training.log"
	eventsDirectory         = "events"
	checkpointsDirectory    = "checkpoints"
	latestCheckpointDir     = "latest"
	bestCheckpointDir       = "best"
	maximumRunIDLength      = 128
	checkpointStateParam    = "supervised_run_state"
)

// ErrStateNotFound indicates that a run has not saved its first checkpoint
// state yet. It is distinct from malformed state.
var ErrStateNotFound = errors.New("run state not found")

// Layout names every artifact belonging to one run. All paths are derived from
// Root and a validated ID; callers do not supply individual artifact paths.
type Layout struct {
	Root string
	ID   string
	Dir  string

	ConfigPath          string
	MetadataPath        string
	StatePath           string
	EventsDir           string
	CheckpointsDir      string
	LatestCheckpointDir string
	BestCheckpointDir   string
	FinalMetricsPath    string
	ValidationGamesPath string
	TrainingLogPath     string
}

// New returns the planned layout without creating anything on disk.
func New(root, runID string) (Layout, error) {
	if err := ValidateRunID(runID); err != nil {
		return Layout{}, err
	}
	if strings.TrimSpace(root) == "" {
		return Layout{}, errors.New("runs root must not be empty")
	}
	absoluteRoot, err := filepath.Abs(root)
	if err != nil {
		return Layout{}, fmt.Errorf("make runs root absolute: %w", err)
	}
	absoluteRoot = filepath.Clean(absoluteRoot)
	runDir := filepath.Join(absoluteRoot, runID)
	if !isPathWithin(absoluteRoot, runDir) {
		return Layout{}, fmt.Errorf("run ID %q escapes runs root", runID)
	}
	checkpointsDir := filepath.Join(runDir, checkpointsDirectory)
	return Layout{
		Root:                absoluteRoot,
		ID:                  runID,
		Dir:                 runDir,
		ConfigPath:          filepath.Join(runDir, configFilename),
		MetadataPath:        filepath.Join(runDir, metadataFilename),
		StatePath:           filepath.Join(runDir, stateFilename),
		EventsDir:           filepath.Join(runDir, eventsDirectory),
		CheckpointsDir:      checkpointsDir,
		LatestCheckpointDir: filepath.Join(checkpointsDir, latestCheckpointDir),
		BestCheckpointDir:   filepath.Join(checkpointsDir, bestCheckpointDir),
		FinalMetricsPath:    filepath.Join(runDir, finalMetricsFilename),
		ValidationGamesPath: filepath.Join(runDir, validationGamesFilename),
		TrainingLogPath:     filepath.Join(runDir, trainingLogFilename),
	}, nil
}

// Create creates an empty run directory and its required directories. It never
// reuses an existing run ID, avoiding accidental mixing of two logical runs.
func Create(root, runID string) (Layout, error) {
	layout, err := New(root, runID)
	if err != nil {
		return Layout{}, err
	}
	if err := os.MkdirAll(layout.Root, 0o755); err != nil {
		return Layout{}, fmt.Errorf("create runs root %q: %w", layout.Root, err)
	}
	if err := os.Mkdir(layout.Dir, 0o755); err != nil {
		if errors.Is(err, os.ErrExist) {
			return Layout{}, fmt.Errorf("run directory %q already exists", layout.Dir)
		}
		return Layout{}, fmt.Errorf("create run directory %q: %w", layout.Dir, err)
	}
	for _, dir := range []string{layout.EventsDir, layout.CheckpointsDir, layout.LatestCheckpointDir, layout.BestCheckpointDir} {
		if err := os.Mkdir(dir, 0o755); err != nil {
			return Layout{}, fmt.Errorf("create run directory %q: %w", dir, err)
		}
	}
	return layout, nil
}

// Open returns an existing run layout. It intentionally does not require a
// checkpoint: an interrupted run may not have reached its first checkpoint.
func Open(root, runID string) (Layout, error) {
	layout, err := New(root, runID)
	if err != nil {
		return Layout{}, err
	}
	info, err := os.Stat(layout.Dir)
	if err != nil {
		return Layout{}, fmt.Errorf("inspect run directory %q: %w", layout.Dir, err)
	}
	if !info.IsDir() {
		return Layout{}, fmt.Errorf("run path %q is not a directory", layout.Dir)
	}
	return layout, nil
}

// ValidateRunID accepts simple portable directory names only. In particular it
// rejects path separators, dot paths, and hidden names.
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

// BestValidation is the best validation loss/result observed so far. Its
// pointer in State distinguishes no validation result from a valid zero value.
type BestValidation struct {
	Value  float64 `json:"value"`
	Update int64   `json:"update"`
}

// State is embedded in the same GoMLX checkpoint as model and optimizer state.
// Cursor counts records already consumed from the deterministically shuffled
// epoch order; it is not a byte offset. NextRecordIDs is an optional short
// audit trail, normally the next records expected after this checkpoint.
type State struct {
	GlobalUpdate   int64           `json:"global_update"`
	ShuffleSeed    int64           `json:"shuffle_seed"`
	DatasetEpoch   int64           `json:"dataset_epoch"`
	ShuffledCursor int             `json:"shuffled_data_cursor"`
	BestValidation *BestValidation `json:"best_validation,omitempty"`
	NextRecordIDs  []int           `json:"next_record_ids,omitempty"`
}

// Validate checks invariants that do not require opening the dataset.
func (state State) Validate() error {
	if state.GlobalUpdate < 0 {
		return fmt.Errorf("global update must not be negative: %d", state.GlobalUpdate)
	}
	if state.ShuffleSeed == 0 {
		return errors.New("shuffle seed must not be zero")
	}
	if state.DatasetEpoch < 0 {
		return fmt.Errorf("dataset epoch must not be negative: %d", state.DatasetEpoch)
	}
	if state.ShuffledCursor < 0 {
		return fmt.Errorf("shuffled-data cursor must not be negative: %d", state.ShuffledCursor)
	}
	if state.BestValidation != nil {
		if math.IsNaN(state.BestValidation.Value) || math.IsInf(state.BestValidation.Value, 0) {
			return fmt.Errorf("best validation value must be finite: %g", state.BestValidation.Value)
		}
		if state.BestValidation.Update < 0 || state.BestValidation.Update > state.GlobalUpdate {
			return fmt.Errorf("best validation update %d is outside 0..%d", state.BestValidation.Update, state.GlobalUpdate)
		}
	}
	for index, recordID := range state.NextRecordIDs {
		if recordID < 0 {
			return fmt.Errorf("next record ID at position %d must not be negative: %d", index, recordID)
		}
	}
	return nil
}

// SaveCheckpointState installs state as a Store parameter. GoMLX then writes
// it into the same checkpoint JSON as the model, Adam moments, global update,
// and random variables, eliminating model/cursor crash mismatches.
func SaveCheckpointState(store *model.Store, state State) error {
	if store == nil {
		return errors.New("checkpoint store must not be nil")
	}
	if err := state.Validate(); err != nil {
		return err
	}
	contents, err := json.Marshal(state)
	if err != nil {
		return fmt.Errorf("encode checkpoint run state: %w", err)
	}
	store.SetParam(checkpointStateParam, string(contents))
	return nil
}

// LoadCheckpointState reads state restored by a GoMLX checkpoint handler.
func LoadCheckpointState(store *model.Store) (State, error) {
	if store == nil {
		return State{}, errors.New("checkpoint store must not be nil")
	}
	value, found := store.GetParam(checkpointStateParam)
	if !found {
		return State{}, ErrStateNotFound
	}
	contents, ok := value.(string)
	if !ok {
		return State{}, fmt.Errorf("checkpoint run state has type %T, want string", value)
	}
	return decodeState([]byte(contents), "checkpoint")
}

// WriteStateMirror atomically writes a human-readable copy of the state most
// recently embedded in a successful checkpoint. Resume must use
// LoadCheckpointState; this mirror is diagnostics only.
func (layout Layout) WriteStateMirror(state State) error {
	if err := state.Validate(); err != nil {
		return err
	}
	return writeJSONAtomically(layout.StatePath, state)
}

// LoadStateMirror reads the human-readable diagnostic state mirror.
func (layout Layout) LoadStateMirror() (State, error) {
	contents, err := os.ReadFile(layout.StatePath)
	if errors.Is(err, os.ErrNotExist) {
		return State{}, ErrStateNotFound
	}
	if err != nil {
		return State{}, fmt.Errorf("read run state %q: %w", layout.StatePath, err)
	}
	return decodeState(contents, layout.StatePath)
}

// WriteConfig records the complete effective configuration for a newly created
// run. WriteMetadata records its reproducibility manifest. Both writes are
// atomic, but callers should treat them as immutable after creation.
func (layout Layout) WriteConfig(config any) error {
	return writeJSONAtomically(layout.ConfigPath, config)
}

// WriteMetadata records a caller-defined reproducibility manifest.
func (layout Layout) WriteMetadata(metadata any) error {
	return writeJSONAtomically(layout.MetadataPath, metadata)
}

// FileSHA256 returns the lower-case SHA-256 digest of a file's bytes.
func FileSHA256(path string) (string, error) {
	file, err := os.Open(path)
	if err != nil {
		return "", fmt.Errorf("open %q for SHA-256: %w", path, err)
	}
	defer func() { _ = file.Close() }()
	digest := sha256.New()
	if _, err := io.Copy(digest, file); err != nil {
		return "", fmt.Errorf("hash %q with SHA-256: %w", path, err)
	}
	return hex.EncodeToString(digest.Sum(nil)), nil
}

// GitCommit returns the checked-out commit of repositoryDir without invoking a
// shell. It is deliberately a helper rather than implicit metadata collection,
// since a run needs to record three repositories explicitly.
func GitCommit(repositoryDir string) (string, error) {
	if strings.TrimSpace(repositoryDir) == "" {
		return "", errors.New("repository directory must not be empty")
	}
	output, err := exec.Command("git", "-C", repositoryDir, "rev-parse", "HEAD").Output()
	if err != nil {
		return "", fmt.Errorf("read Git commit for %q: %w", repositoryDir, err)
	}
	commit := strings.TrimSpace(string(output))
	if commit == "" {
		return "", fmt.Errorf("Git returned an empty commit for %q", repositoryDir)
	}
	return commit, nil
}

// RuntimeMetadata identifies the Go runtime and pinned GoMLX module available
// to the running binary. Backend, GPU, and CUDA/PJRT details are supplied by
// the training command because this package deliberately has no CUDA dependency.
type RuntimeMetadata struct {
	GoVersion    string `json:"go_version"`
	GOOS         string `json:"goos"`
	GOARCH       string `json:"goarch"`
	GoMLXVersion string `json:"gomlx_version,omitempty"`
}

// CurrentRuntimeMetadata collects stable process metadata without inspecting
// hardware or mutating state.
func CurrentRuntimeMetadata() RuntimeMetadata {
	metadata := RuntimeMetadata{
		GoVersion: runtime.Version(),
		GOOS:      runtime.GOOS,
		GOARCH:    runtime.GOARCH,
	}
	buildInfo, ok := debug.ReadBuildInfo()
	if !ok {
		return metadata
	}
	for _, dependency := range buildInfo.Deps {
		if dependency.Path == "github.com/gomlx/gomlx" {
			metadata.GoMLXVersion = dependency.Version
			break
		}
	}
	return metadata
}

func decodeState(contents []byte, source string) (State, error) {
	var state State
	if err := json.Unmarshal(contents, &state); err != nil {
		return State{}, fmt.Errorf("decode run state from %s: %w", source, err)
	}
	if err := state.Validate(); err != nil {
		return State{}, fmt.Errorf("validate run state from %s: %w", source, err)
	}
	return state, nil
}

func isPathWithin(root, path string) bool {
	relative, err := filepath.Rel(root, path)
	if err != nil {
		return false
	}
	return relative != ".." && !strings.HasPrefix(relative, ".."+string(filepath.Separator)) && !filepath.IsAbs(relative)
}

func writeJSONAtomically(path string, value any) error {
	contents, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return fmt.Errorf("encode JSON for %q: %w", path, err)
	}
	contents = append(contents, '\n')
	directory := filepath.Dir(path)
	temporary, err := os.CreateTemp(directory, "."+filepath.Base(path)+".tmp-*")
	if err != nil {
		return fmt.Errorf("create temporary JSON file beside %q: %w", path, err)
	}
	temporaryPath := temporary.Name()
	defer func() { _ = os.Remove(temporaryPath) }()
	if err := temporary.Chmod(0o644); err != nil {
		_ = temporary.Close()
		return fmt.Errorf("set permissions on temporary JSON file for %q: %w", path, err)
	}
	if _, err := temporary.Write(contents); err != nil {
		_ = temporary.Close()
		return fmt.Errorf("write temporary JSON file for %q: %w", path, err)
	}
	if err := temporary.Sync(); err != nil {
		_ = temporary.Close()
		return fmt.Errorf("sync temporary JSON file for %q: %w", path, err)
	}
	if err := temporary.Close(); err != nil {
		return fmt.Errorf("close temporary JSON file for %q: %w", path, err)
	}
	if err := os.Rename(temporaryPath, path); err != nil {
		return fmt.Errorf("replace JSON file %q: %w", path, err)
	}
	directoryHandle, err := os.Open(directory)
	if err != nil {
		return fmt.Errorf("open JSON directory %q for sync: %w", directory, err)
	}
	defer func() { _ = directoryHandle.Close() }()
	if err := directoryHandle.Sync(); err != nil {
		return fmt.Errorf("sync JSON directory %q: %w", directory, err)
	}
	return nil
}

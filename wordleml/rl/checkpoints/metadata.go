package checkpoints

import (
	"bytes"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// SchemaVersion identifies the durable JSON structures in this package.
const SchemaVersion = 1

// IterationState is the small durable state that accompanies the GoMLX actor,
// critic, and optimiser checkpoint directories. It is intentionally separate
// from their binary representation so an interrupted runner can identify the
// exact rollout seed and optimisation progress it is resuming.
type IterationState struct {
	SchemaVersion  int    `json:"schema_version"`
	Iteration      int    `json:"iteration"`
	RolloutSeed    int64  `json:"rollout_seed"`
	ActorSteps     int64  `json:"actor_steps"`
	CriticSteps    int64  `json:"critic_steps"`
	ActorChecksum  string `json:"actor_checksum"`
	CriticChecksum string `json:"critic_checksum"`
}

// Validate checks structural resume invariants independent of a particular
// run directory. Checksums are SHA-256 digests of the checkpoint generation
// chosen by the runner, typically of its manifest rather than a large binary.
func (state IterationState) Validate() error {
	if state.SchemaVersion != SchemaVersion {
		return fmt.Errorf("unsupported iteration-state schema version %d", state.SchemaVersion)
	}
	if err := ValidateIteration(state.Iteration); err != nil {
		return err
	}
	if state.ActorSteps < 0 || state.CriticSteps < 0 {
		return fmt.Errorf("actor and critic steps must not be negative: %d, %d", state.ActorSteps, state.CriticSteps)
	}
	if err := validateChecksum("actor", state.ActorChecksum); err != nil {
		return err
	}
	if err := validateChecksum("critic", state.CriticChecksum); err != nil {
		return err
	}
	return nil
}

// AcceptedState is the single authoritative pointer to the most recently
// accepted PPO candidate. All paths are relative to Layout.Dir, allowing a
// run directory to be moved as a unit without making its accepted checkpoint
// point outside the experiment.
type AcceptedState struct {
	SchemaVersion int    `json:"schema_version"`
	Status        string `json:"status"`
	Iteration     int    `json:"iteration"`

	ActorCriticCheckpoint string `json:"actor_critic_checkpoint"`
	ActorOnlyCheckpoint   string `json:"actor_only_checkpoint"`
	Evaluation            string `json:"evaluation"`
	IterationState        string `json:"iteration_state"`
}

// NewAcceptedState builds the only accepted pointer permitted for iteration.
// It is exposed to make planned checkpoint paths inspectable without allowing
// callers to construct arbitrary relative paths.
func (layout Layout) NewAcceptedState(iteration int) (AcceptedState, error) {
	entry, err := layout.Iteration(iteration)
	if err != nil {
		return AcceptedState{}, err
	}
	actorCritic, err := relativePath(layout.Dir, entry.ActorCriticDir)
	if err != nil {
		return AcceptedState{}, err
	}
	actorOnly, err := relativePath(layout.Dir, entry.ActorOnlyDir)
	if err != nil {
		return AcceptedState{}, err
	}
	evaluation, err := relativePath(layout.Dir, entry.EvaluationPath)
	if err != nil {
		return AcceptedState{}, err
	}
	state, err := relativePath(layout.Dir, entry.StatePath)
	if err != nil {
		return AcceptedState{}, err
	}
	return AcceptedState{
		SchemaVersion:         SchemaVersion,
		Status:                "accepted",
		Iteration:             iteration,
		ActorCriticCheckpoint: filepath.ToSlash(actorCritic),
		ActorOnlyCheckpoint:   filepath.ToSlash(actorOnly),
		Evaluation:            filepath.ToSlash(evaluation),
		IterationState:        filepath.ToSlash(state),
	}, nil
}

// Validate checks generic accepted-state invariants. Layout-specific path
// validation happens in LoadAccepted and Promote, where the expected
// iteration-derived locations are known.
func (state AcceptedState) Validate() error {
	if state.SchemaVersion != SchemaVersion {
		return fmt.Errorf("unsupported accepted-state schema version %d", state.SchemaVersion)
	}
	if state.Status != "accepted" {
		return fmt.Errorf("accepted state status must be accepted, got %q", state.Status)
	}
	if err := ValidateIteration(state.Iteration); err != nil {
		return err
	}
	for name, path := range map[string]string{
		"actor-critic checkpoint": state.ActorCriticCheckpoint,
		"actor-only checkpoint":   state.ActorOnlyCheckpoint,
		"evaluation":              state.Evaluation,
		"iteration state":         state.IterationState,
	} {
		if !safeRelativePath(path) {
			return fmt.Errorf("%s path %q is not a safe relative path", name, path)
		}
	}
	return nil
}

// WriteConfig records the fully resolved PPO configuration. Configuration is
// immutable: an idempotent retry is accepted, while a differing write fails.
func (layout Layout) WriteConfig(config any) error {
	return publishJSONImmutably(layout.ConfigPath, config)
}

// WriteMetadata records immutable PPO run provenance, such as the split
// manifest identity and the sealed-test declaration.
func (layout Layout) WriteMetadata(metadata any) error {
	return publishJSONImmutably(layout.MetadataPath, metadata)
}

// WriteSupervisedBaselineMetadata records identity only (for example source
// path, checkpoint digest, and producing commit). It never copies, loads, or
// overwrites actor weights. A differing identity is rejected rather than
// silently mutating the known-good supervised baseline record.
func (layout Layout) WriteSupervisedBaselineMetadata(metadata any) error {
	if err := requireDirectory(layout.SupervisedBaselineDir); err != nil {
		return fmt.Errorf("open supervised baseline directory: %w", err)
	}
	return publishJSONImmutably(layout.SupervisedBaselineMetadataPath, metadata)
}

// WriteState atomically publishes a candidate's resume state. The state must
// belong to this exact iteration, so it cannot accidentally journal progress
// for another candidate.
func (layout IterationLayout) WriteState(state IterationState) error {
	if state.Iteration != layout.Iteration {
		return fmt.Errorf("iteration state is for iteration %d, want %d", state.Iteration, layout.Iteration)
	}
	if err := state.Validate(); err != nil {
		return err
	}
	if err := requireDirectory(layout.Dir); err != nil {
		return fmt.Errorf("open PPO iteration directory: %w", err)
	}
	return writeJSONAtomically(layout.StatePath, state)
}

// LoadState loads and validates the small resume state for this candidate.
func (layout IterationLayout) LoadState() (IterationState, error) {
	contents, err := os.ReadFile(layout.StatePath)
	if errors.Is(err, os.ErrNotExist) {
		return IterationState{}, ErrIterationStateNotFound
	}
	if err != nil {
		return IterationState{}, fmt.Errorf("read PPO iteration state %q: %w", layout.StatePath, err)
	}
	var state IterationState
	if err := json.Unmarshal(contents, &state); err != nil {
		return IterationState{}, fmt.Errorf("decode PPO iteration state %q: %w", layout.StatePath, err)
	}
	if state.Iteration != layout.Iteration {
		return IterationState{}, fmt.Errorf("iteration state %q is for iteration %d, want %d", layout.StatePath, state.Iteration, layout.Iteration)
	}
	if err := state.Validate(); err != nil {
		return IterationState{}, fmt.Errorf("validate PPO iteration state %q: %w", layout.StatePath, err)
	}
	return state, nil
}

// WriteEvaluation atomically writes the complete machine-readable evaluation
// result for a candidate. The runner supplies its typed evaluation schema; the
// layout only guarantees that it is valid JSON and remains within the
// candidate directory.
func (layout IterationLayout) WriteEvaluation(evaluation any) error {
	if err := requireDirectory(layout.Dir); err != nil {
		return fmt.Errorf("open PPO iteration directory: %w", err)
	}
	return writeJSONAtomically(layout.EvaluationPath, evaluation)
}

// LoadAccepted returns the authoritative last promoted candidate.
func (layout Layout) LoadAccepted() (AcceptedState, error) {
	contents, err := os.ReadFile(layout.AcceptedPath)
	if errors.Is(err, os.ErrNotExist) {
		return AcceptedState{}, ErrAcceptedNotFound
	}
	if err != nil {
		return AcceptedState{}, fmt.Errorf("read accepted PPO state %q: %w", layout.AcceptedPath, err)
	}
	var state AcceptedState
	if err := json.Unmarshal(contents, &state); err != nil {
		return AcceptedState{}, fmt.Errorf("decode accepted PPO state %q: %w", layout.AcceptedPath, err)
	}
	if err := layout.validateAcceptedState(state); err != nil {
		return AcceptedState{}, fmt.Errorf("validate accepted PPO state %q: %w", layout.AcceptedPath, err)
	}
	return state, nil
}

// Promote publishes iteration as the current accepted PPO checkpoint. Before
// calling it, the runner must have copied its heavyweight actor-critic and
// actor-only checkpoint payloads to BestDir if it wants best/ to contain a
// deployable checkpoint. This method deliberately does not copy those files.
//
// The small iteration state and evaluation are copied into best/ first. Only
// after those durable diagnostic records have been installed does accepted.json
// change, so a failed promotion always leaves the previous accepted pointer
// authoritative and the supervised baseline untouched.
func (layout Layout) Promote(iteration int) (AcceptedState, error) {
	entry, err := layout.OpenIteration(iteration)
	if err != nil {
		return AcceptedState{}, err
	}
	if _, err := entry.LoadState(); err != nil {
		return AcceptedState{}, fmt.Errorf("candidate cannot be promoted without resume state: %w", err)
	}
	if _, err := readJSONFile(entry.EvaluationPath); err != nil {
		return AcceptedState{}, fmt.Errorf("candidate cannot be promoted without evaluation JSON: %w", err)
	}
	accepted, err := layout.NewAcceptedState(iteration)
	if err != nil {
		return AcceptedState{}, err
	}
	if err := layout.validateAcceptedState(accepted); err != nil {
		return AcceptedState{}, err
	}
	if err := copySmallJSONAtomically(entry.StatePath, layout.BestIterationStatePath); err != nil {
		return AcceptedState{}, fmt.Errorf("publish best iteration state: %w", err)
	}
	if err := copySmallJSONAtomically(entry.EvaluationPath, layout.BestEvaluationPath); err != nil {
		return AcceptedState{}, fmt.Errorf("publish best evaluation: %w", err)
	}
	if err := writeJSONAtomically(layout.BestAcceptedPath, accepted); err != nil {
		return AcceptedState{}, fmt.Errorf("publish best accepted state: %w", err)
	}
	if err := writeJSONAtomically(layout.AcceptedPath, accepted); err != nil {
		return AcceptedState{}, fmt.Errorf("publish accepted PPO state: %w", err)
	}
	return accepted, nil
}

func (layout Layout) validateAcceptedState(state AcceptedState) error {
	if err := state.Validate(); err != nil {
		return err
	}
	want, err := layout.NewAcceptedState(state.Iteration)
	if err != nil {
		return err
	}
	if state != want {
		return fmt.Errorf("accepted state checkpoint paths must refer exactly to %s", want.ActorCriticCheckpoint)
	}
	return nil
}

func validateChecksum(name, checksum string) error {
	if len(checksum) != 64 {
		return fmt.Errorf("%s checksum must be a 64-character SHA-256 digest", name)
	}
	decoded, err := hex.DecodeString(checksum)
	if err != nil || len(decoded) != 32 || strings.ToLower(checksum) != checksum {
		return fmt.Errorf("%s checksum must be lowercase hexadecimal SHA-256", name)
	}
	return nil
}

func relativePath(root, path string) (string, error) {
	relative, err := filepath.Rel(root, path)
	if err != nil {
		return "", err
	}
	if !safeRelativePath(relative) {
		return "", fmt.Errorf("path %q is not within %q", path, root)
	}
	return relative, nil
}

func safeRelativePath(path string) bool {
	// Accepted-state paths are stored with slash separators on every platform.
	// Reject backslashes even on Unix, where accepting one would create an
	// unexpected literal filename and make a moved run ambiguous on Windows.
	if path == "" || filepath.IsAbs(path) || strings.Contains(path, "\\") {
		return false
	}
	portable := filepath.FromSlash(path)
	clean := filepath.Clean(portable)
	if clean == "." || clean == ".." || strings.HasPrefix(clean, ".."+string(filepath.Separator)) {
		return false
	}
	return clean == portable
}

func readJSONFile(path string) ([]byte, error) {
	info, err := os.Stat(path)
	if err != nil {
		return nil, err
	}
	if info.Size() > maximumSmallMetadataBytes {
		return nil, fmt.Errorf("metadata file is too large (%d bytes, limit %d)", info.Size(), maximumSmallMetadataBytes)
	}
	contents, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if !json.Valid(contents) {
		return nil, errors.New("file is not valid JSON")
	}
	return contents, nil
}

func copySmallJSONAtomically(source, destination string) error {
	contents, err := readJSONFile(source)
	if err != nil {
		return err
	}
	return writeBytesAtomically(destination, contents)
}

func publishJSONImmutably(path string, value any) error {
	contents, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return fmt.Errorf("encode JSON for %q: %w", path, err)
	}
	return publishBytesImmutably(path, append(contents, '\n'))
}

func writeJSONAtomically(path string, value any) error {
	contents, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return fmt.Errorf("encode JSON for %q: %w", path, err)
	}
	return writeBytesAtomically(path, append(contents, '\n'))
}

func writeBytesAtomically(path string, contents []byte) error {
	directory := filepath.Dir(path)
	temporary, err := os.CreateTemp(directory, "."+filepath.Base(path)+".tmp-*")
	if err != nil {
		return fmt.Errorf("create temporary artifact beside %q: %w", path, err)
	}
	temporaryPath := temporary.Name()
	defer func() { _ = os.Remove(temporaryPath) }()
	if err := temporary.Chmod(0o644); err != nil {
		_ = temporary.Close()
		return fmt.Errorf("set permissions on temporary artifact for %q: %w", path, err)
	}
	if _, err := temporary.Write(contents); err != nil {
		_ = temporary.Close()
		return fmt.Errorf("write temporary artifact for %q: %w", path, err)
	}
	if err := temporary.Sync(); err != nil {
		_ = temporary.Close()
		return fmt.Errorf("sync temporary artifact for %q: %w", path, err)
	}
	if err := temporary.Close(); err != nil {
		return fmt.Errorf("close temporary artifact for %q: %w", path, err)
	}
	if err := os.Rename(temporaryPath, path); err != nil {
		return fmt.Errorf("replace artifact %q: %w", path, err)
	}
	return syncDirectory(directory)
}

func publishBytesImmutably(path string, contents []byte) error {
	if existing, err := os.ReadFile(path); err == nil {
		if bytes.Equal(existing, contents) {
			return nil
		}
		return fmt.Errorf("immutable artifact %q already exists with different contents", path)
	} else if !errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("read immutable artifact %q: %w", path, err)
	}
	directory := filepath.Dir(path)
	temporary, err := os.CreateTemp(directory, "."+filepath.Base(path)+".tmp-*")
	if err != nil {
		return fmt.Errorf("create temporary immutable artifact beside %q: %w", path, err)
	}
	temporaryPath := temporary.Name()
	defer func() { _ = os.Remove(temporaryPath) }()
	if err := temporary.Chmod(0o644); err != nil {
		_ = temporary.Close()
		return err
	}
	if _, err := temporary.Write(contents); err != nil {
		_ = temporary.Close()
		return err
	}
	if err := temporary.Sync(); err != nil {
		_ = temporary.Close()
		return err
	}
	if err := temporary.Close(); err != nil {
		return err
	}
	if err := os.Link(temporaryPath, path); err != nil {
		if errors.Is(err, os.ErrExist) {
			existing, readErr := os.ReadFile(path)
			if readErr != nil {
				return fmt.Errorf("read concurrently published immutable artifact %q: %w", path, readErr)
			}
			if bytes.Equal(existing, contents) {
				return nil
			}
			return fmt.Errorf("immutable artifact %q was concurrently published with different contents", path)
		}
		return fmt.Errorf("publish immutable artifact %q: %w", path, err)
	}
	return syncDirectory(directory)
}

func syncDirectory(directory string) error {
	handle, err := os.Open(directory)
	if err != nil {
		return fmt.Errorf("open artifact directory %q for sync: %w", directory, err)
	}
	defer func() { _ = handle.Close() }()
	if err := handle.Sync(); err != nil {
		return fmt.Errorf("sync artifact directory %q: %w", directory, err)
	}
	return nil
}

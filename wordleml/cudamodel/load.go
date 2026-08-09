package cudamodel

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
)

// Model is a validated portable policy. Its weight storage is immutable from
// the package's point of view: Weights returns a copy so a caller cannot alter
// the evaluator after artifact validation.
type Model struct {
	Manifest Manifest
	weights  []float32
}

// Load reads manifest.json and its fixed little-endian FP32 payload from dir.
// expected must come from the vocabulary currently loaded by the caller; this
// prevents using a model with the same dimensions but different word IDs.
func Load(dir string, expected VocabularyHashes) (*Model, error) {
	if err := validateVocabularyHashes(expected); err != nil {
		return nil, fmt.Errorf("expected vocabulary hashes: %w", err)
	}

	manifestPath := filepath.Join(dir, ManifestFilename)
	manifestData, err := os.ReadFile(manifestPath)
	if err != nil {
		return nil, fmt.Errorf("read manifest %q: %w", manifestPath, err)
	}
	manifest, err := decodeManifest(manifestData)
	if err != nil {
		return nil, fmt.Errorf("decode manifest %q: %w", manifestPath, err)
	}
	if err := ValidateManifest(manifest, expected); err != nil {
		return nil, fmt.Errorf("validate manifest %q: %w", manifestPath, err)
	}

	weightsPath := filepath.Join(dir, WeightsFilename)
	weightBytes, err := os.ReadFile(weightsPath)
	if err != nil {
		return nil, fmt.Errorf("read weights %q: %w", weightsPath, err)
	}
	if len(weightBytes) != WeightsByteCount {
		return nil, fmt.Errorf("weights file has %d bytes, want exactly %d", len(weightBytes), WeightsByteCount)
	}
	digest := sha256.Sum256(weightBytes)
	if got := hex.EncodeToString(digest[:]); got != manifest.WeightsSHA256 {
		return nil, fmt.Errorf("weights SHA-256 is %s, want %s", got, manifest.WeightsSHA256)
	}

	weights := make([]float32, ParameterCount)
	for i := range weights {
		weights[i] = math.Float32frombits(binary.LittleEndian.Uint32(weightBytes[i*4:]))
		if !isFinite(weights[i]) {
			return nil, fmt.Errorf("weight %d is not finite", i)
		}
	}
	return &Model{Manifest: cloneManifest(manifest), weights: weights}, nil
}

// ValidateManifest verifies every fixed field and every tensor-layout detail
// independently of disk I/O. Exporter tests can call it before writing an
// artifact, and runtime code calls it before the weights reach native code.
func ValidateManifest(manifest Manifest, expected VocabularyHashes) error {
	if err := validateVocabularyHashes(expected); err != nil {
		return fmt.Errorf("expected vocabulary hashes: %w", err)
	}
	if manifest.Format != Format {
		return fmt.Errorf("format is %q, want %q", manifest.Format, Format)
	}
	if manifest.Endianness != "little" {
		return fmt.Errorf("endianness is %q, want %q", manifest.Endianness, "little")
	}
	if manifest.DType != "float32" {
		return fmt.Errorf("dtype is %q, want %q", manifest.DType, "float32")
	}
	if strings.TrimSpace(manifest.RunID) == "" {
		return fmt.Errorf("run_id must not be empty")
	}
	if strings.TrimSpace(manifest.Checkpoint) == "" {
		return fmt.Errorf("checkpoint must not be empty")
	}
	if manifest.CheckpointUpdate < 0 {
		return fmt.Errorf("checkpoint_update is %d, want non-negative", manifest.CheckpointUpdate)
	}
	if strings.TrimSpace(manifest.TrainingCommit) == "" {
		return fmt.Errorf("training_commit must not be empty")
	}
	if manifest.NumSolutions != NumSolutions {
		return fmt.Errorf("num_solutions is %d, want %d", manifest.NumSolutions, NumSolutions)
	}
	if manifest.NumActions != NumActions {
		return fmt.Errorf("num_actions is %d, want %d", manifest.NumActions, NumActions)
	}
	if manifest.CandidateStatsSize != CandidateStatsSize {
		return fmt.Errorf("candidate_stats_size is %d, want %d", manifest.CandidateStatsSize, CandidateStatsSize)
	}
	if manifest.NumTurns != NumTurns {
		return fmt.Errorf("num_turns is %d, want %d", manifest.NumTurns, NumTurns)
	}
	if manifest.TrunkSize != TrunkSize {
		return fmt.Errorf("trunk_size is %d, want %d", manifest.TrunkSize, TrunkSize)
	}
	if manifest.ParameterCount != ParameterCount {
		return fmt.Errorf("parameter_count is %d, want %d", manifest.ParameterCount, ParameterCount)
	}
	if manifest.WeightsFile != WeightsFilename {
		return fmt.Errorf("weights_file is %q, want %q", manifest.WeightsFile, WeightsFilename)
	}
	if !isSHA256(manifest.WeightsSHA256) {
		return fmt.Errorf("weights_sha256 must be a lowercase SHA-256 hex digest")
	}
	if !isSHA256(manifest.SolutionVocabularySHA256) {
		return fmt.Errorf("solution_vocabulary_sha256 must be a lowercase SHA-256 hex digest")
	}
	if !isSHA256(manifest.ActionVocabularySHA256) {
		return fmt.Errorf("action_vocabulary_sha256 must be a lowercase SHA-256 hex digest")
	}
	if manifest.SolutionVocabularySHA256 != expected.Solutions {
		return fmt.Errorf("solution vocabulary SHA-256 is %s, want %s", manifest.SolutionVocabularySHA256, expected.Solutions)
	}
	if manifest.ActionVocabularySHA256 != expected.Actions {
		return fmt.Errorf("action vocabulary SHA-256 is %s, want %s", manifest.ActionVocabularySHA256, expected.Actions)
	}
	if len(manifest.Tensors) != TensorCount {
		return fmt.Errorf("tensor table has %d tensors, want %d", len(manifest.Tensors), TensorCount)
	}
	for i, got := range manifest.Tensors {
		want := expectedTensors[i]
		if got.Name != want.Name {
			return fmt.Errorf("tensor %d name is %q, want %q", i, got.Name, want.Name)
		}
		if !sameShape(got.Shape, want.Shape) {
			return fmt.Errorf("tensor %q shape is %v, want %v", got.Name, got.Shape, want.Shape)
		}
		if got.Offset != want.Offset {
			return fmt.Errorf("tensor %q offset is %d, want %d", got.Name, got.Offset, want.Offset)
		}
		if got.Count != want.Count {
			return fmt.Errorf("tensor %q count is %d, want %d", got.Name, got.Count, want.Count)
		}
		if got.SourceName != want.SourceName {
			return fmt.Errorf("tensor %q source_name is %q, want %q", got.Name, got.SourceName, want.SourceName)
		}
	}
	return nil
}

// Weights returns a detached contiguous FP32 copy in ExpectedTensors order.
// It is intended for the one-time native-model creation call.
func (m *Model) Weights() []float32 {
	if m == nil {
		return nil
	}
	return append([]float32(nil), m.weights...)
}

func decodeManifest(data []byte) (Manifest, error) {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	var manifest Manifest
	if err := decoder.Decode(&manifest); err != nil {
		return Manifest{}, err
	}
	if err := decoder.Decode(&struct{}{}); err != io.EOF {
		if err == nil {
			return Manifest{}, fmt.Errorf("multiple JSON values")
		}
		return Manifest{}, fmt.Errorf("trailing data: %w", err)
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(data, &fields); err != nil {
		return Manifest{}, err
	}
	for _, name := range requiredManifestFields {
		if _, found := fields[name]; !found {
			return Manifest{}, fmt.Errorf("missing required field %q", name)
		}
	}
	return manifest, nil
}

var requiredManifestFields = []string{
	"format",
	"endianness",
	"dtype",
	"run_id",
	"checkpoint",
	"checkpoint_update",
	"training_commit",
	"num_solutions",
	"num_actions",
	"candidate_stats_size",
	"num_turns",
	"trunk_size",
	"parameter_count",
	"weights_file",
	"weights_sha256",
	"solution_vocabulary_sha256",
	"action_vocabulary_sha256",
	"tensors",
}

func validateVocabularyHashes(hashes VocabularyHashes) error {
	if !isSHA256(hashes.Solutions) {
		return fmt.Errorf("solutions must be a lowercase SHA-256 hex digest")
	}
	if !isSHA256(hashes.Actions) {
		return fmt.Errorf("actions must be a lowercase SHA-256 hex digest")
	}
	return nil
}

func sameShape(left, right []int) bool {
	if len(left) != len(right) {
		return false
	}
	for i := range left {
		if left[i] != right[i] {
			return false
		}
	}
	return true
}

func isSHA256(value string) bool {
	if len(value) != sha256.Size*2 {
		return false
	}
	for _, char := range value {
		if !(char >= '0' && char <= '9' || char >= 'a' && char <= 'f') {
			return false
		}
	}
	return true
}

func isFinite(value float32) bool {
	return !math.IsNaN(float64(value)) && !math.IsInf(float64(value), 0)
}

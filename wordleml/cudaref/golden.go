package cudaref

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"

	"github.com/sam-bee/wordle-ml_machine-learning/modelstate"
	"github.com/sam-bee/wordle-ml_machine-learning/vocabulary"
)

// WriteGoldenVectors writes a versioned JSON index and one little-endian FP32
// payload below dir.  It is safe to call before publishing the enclosing
// export directory because it never writes outside dir.
func WriteGoldenVectors(dir string, vectors []Vector) (GoldenManifest, error) {
	if len(vectors) == 0 {
		return GoldenManifest{}, errors.New("golden vector set must not be empty")
	}
	values := make([]float32, 0, len(vectors)*(vocabulary.NumSolutions+modelstate.CandidateStatsSize+3*vocabulary.NumActions))
	metadata := make([]VectorMetadata, 0, len(vectors))
	seen := make(map[string]struct{}, len(vectors))
	for _, vector := range vectors {
		if err := ValidateVector(vector); err != nil {
			return GoldenManifest{}, err
		}
		if _, exists := seen[vector.ID]; exists {
			return GoldenManifest{}, fmt.Errorf("golden vector ID %q is repeated", vector.ID)
		}
		seen[vector.ID] = struct{}{}
		metadata = append(metadata, VectorMetadata{
			ID:                  vector.ID,
			Turn:                vector.Inputs.Turn,
			CandidateMask:       appendSpan(&values, vector.Inputs.CandidateMask),
			CandidateStats:      appendSpan(&values, vector.Inputs.CandidateStats),
			RemainingActionMask: appendSpan(&values, vector.Inputs.RemainingActionMask),
			AvailableActionMask: appendSpan(&values, vector.AvailableActionMask),
			RawLogits:           appendSpan(&values, vector.RawLogits),
			RawTopActionID:      vector.RawTopActionID,
			SelectedActionID:    vector.SelectedActionID,
			TopTwoMargin:        vector.TopTwoMargin,
			Provenance:          vector.Provenance,
		})
	}
	payload, err := encodeFloat32LE(values)
	if err != nil {
		return GoldenManifest{}, err
	}
	digest := sha256.Sum256(payload)
	manifest := GoldenManifest{
		Format:       GoldenFormat,
		Endianness:   "little",
		DType:        "float32",
		ValuesFile:   GoldenValuesFilename,
		ValuesSHA256: hex.EncodeToString(digest[:]),
		ValuesCount:  len(values),
		Vectors:      metadata,
	}
	if err := validateManifest(manifest); err != nil {
		return GoldenManifest{}, err
	}
	if err := os.WriteFile(filepath.Join(dir, GoldenValuesFilename), payload, 0o644); err != nil {
		return GoldenManifest{}, fmt.Errorf("write %s: %w", GoldenValuesFilename, err)
	}
	contents, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		return GoldenManifest{}, fmt.Errorf("encode golden manifest: %w", err)
	}
	contents = append(contents, '\n')
	if err := os.WriteFile(filepath.Join(dir, GoldenManifestFilename), contents, 0o644); err != nil {
		return GoldenManifest{}, fmt.Errorf("write %s: %w", GoldenManifestFilename, err)
	}
	return manifest, nil
}

// LoadGoldenVectors validates and decodes both golden sidecars.
func LoadGoldenVectors(dir string) (GoldenSet, error) {
	contents, err := os.ReadFile(filepath.Join(dir, GoldenManifestFilename))
	if err != nil {
		return GoldenSet{}, fmt.Errorf("read %s: %w", GoldenManifestFilename, err)
	}
	var manifest GoldenManifest
	if err := json.Unmarshal(contents, &manifest); err != nil {
		return GoldenSet{}, fmt.Errorf("decode %s: %w", GoldenManifestFilename, err)
	}
	if err := validateManifest(manifest); err != nil {
		return GoldenSet{}, err
	}
	payload, err := os.ReadFile(filepath.Join(dir, manifest.ValuesFile))
	if err != nil {
		return GoldenSet{}, fmt.Errorf("read %s: %w", manifest.ValuesFile, err)
	}
	digest := sha256.Sum256(payload)
	if actual := hex.EncodeToString(digest[:]); actual != manifest.ValuesSHA256 {
		return GoldenSet{}, fmt.Errorf("%s SHA-256 is %s, want %s", manifest.ValuesFile, actual, manifest.ValuesSHA256)
	}
	values, err := decodeFloat32LE(payload)
	if err != nil {
		return GoldenSet{}, fmt.Errorf("decode %s: %w", manifest.ValuesFile, err)
	}
	if len(values) != manifest.ValuesCount {
		return GoldenSet{}, fmt.Errorf("%s has %d float values, want %d", manifest.ValuesFile, len(values), manifest.ValuesCount)
	}
	vectors := make([]Vector, len(manifest.Vectors))
	for index, metadata := range manifest.Vectors {
		vector, err := vectorFromMetadata(values, metadata)
		if err != nil {
			return GoldenSet{}, fmt.Errorf("golden vector %d: %w", index, err)
		}
		vectors[index] = vector
	}
	return GoldenSet{Manifest: manifest, Vectors: vectors}, nil
}

func appendSpan(values *[]float32, additions []float32) Span {
	span := Span{OffsetFloats: len(*values), Count: len(additions)}
	*values = append(*values, additions...)
	return span
}

func vectorFromMetadata(values []float32, metadata VectorMetadata) (Vector, error) {
	candidateMask, err := valuesForSpan(values, metadata.CandidateMask, vocabulary.NumSolutions, "candidate mask")
	if err != nil {
		return Vector{}, err
	}
	candidateStats, err := valuesForSpan(values, metadata.CandidateStats, modelstate.CandidateStatsSize, "candidate stats")
	if err != nil {
		return Vector{}, err
	}
	remaining, err := valuesForSpan(values, metadata.RemainingActionMask, vocabulary.NumActions, "remaining action mask")
	if err != nil {
		return Vector{}, err
	}
	available, err := valuesForSpan(values, metadata.AvailableActionMask, vocabulary.NumActions, "availability mask")
	if err != nil {
		return Vector{}, err
	}
	logits, err := valuesForSpan(values, metadata.RawLogits, vocabulary.NumActions, "raw logits")
	if err != nil {
		return Vector{}, err
	}
	vector := Vector{
		ID: metadata.ID,
		Inputs: modelstate.Inputs{
			CandidateMask: candidateMask, CandidateStats: candidateStats, Turn: metadata.Turn, RemainingActionMask: remaining,
		},
		AvailableActionMask: available,
		RawLogits:           logits,
		RawTopActionID:      metadata.RawTopActionID,
		SelectedActionID:    metadata.SelectedActionID,
		TopTwoMargin:        metadata.TopTwoMargin,
		Provenance:          metadata.Provenance,
	}
	if err := ValidateVector(vector); err != nil {
		return Vector{}, err
	}
	return vector, nil
}

func valuesForSpan(values []float32, span Span, expected int, label string) ([]float32, error) {
	if span.Count != expected {
		return nil, fmt.Errorf("%s span count %d, want %d", label, span.Count, expected)
	}
	if span.OffsetFloats < 0 || span.OffsetFloats > len(values) || span.Count > len(values)-span.OffsetFloats {
		return nil, fmt.Errorf("%s span offset %d/count %d is outside %d values", label, span.OffsetFloats, span.Count, len(values))
	}
	return append([]float32(nil), values[span.OffsetFloats:span.OffsetFloats+span.Count]...), nil
}

func validateManifest(manifest GoldenManifest) error {
	if manifest.Format != GoldenFormat {
		return fmt.Errorf("unsupported golden-vector format %q", manifest.Format)
	}
	if manifest.Endianness != "little" || manifest.DType != "float32" {
		return fmt.Errorf("golden vectors must be little-endian float32, got %s %s", manifest.Endianness, manifest.DType)
	}
	if manifest.ValuesFile != GoldenValuesFilename {
		return fmt.Errorf("golden values file %q, want %q", manifest.ValuesFile, GoldenValuesFilename)
	}
	if len(manifest.ValuesSHA256) != sha256.Size*2 {
		return fmt.Errorf("golden values SHA-256 has %d characters, want %d", len(manifest.ValuesSHA256), sha256.Size*2)
	}
	if _, err := hex.DecodeString(manifest.ValuesSHA256); err != nil {
		return fmt.Errorf("decode golden values SHA-256: %w", err)
	}
	if manifest.ValuesCount <= 0 || len(manifest.Vectors) == 0 {
		return errors.New("golden vectors must contain values and vectors")
	}
	seen := make(map[string]struct{}, len(manifest.Vectors))
	for _, vector := range manifest.Vectors {
		if vector.ID == "" {
			return errors.New("golden vector metadata ID is required")
		}
		if _, duplicate := seen[vector.ID]; duplicate {
			return fmt.Errorf("golden vector metadata repeats %q", vector.ID)
		}
		seen[vector.ID] = struct{}{}
		if vector.Turn < 0 || vector.Turn >= 6 {
			return fmt.Errorf("golden vector %q has invalid turn %d", vector.ID, vector.Turn)
		}
		for _, span := range []Span{vector.CandidateMask, vector.CandidateStats, vector.RemainingActionMask, vector.AvailableActionMask, vector.RawLogits} {
			if span.OffsetFloats < 0 || span.Count <= 0 || span.OffsetFloats > manifest.ValuesCount || span.Count > manifest.ValuesCount-span.OffsetFloats {
				return fmt.Errorf("golden vector %q has invalid span offset=%d count=%d", vector.ID, span.OffsetFloats, span.Count)
			}
		}
	}
	return nil
}

func encodeFloat32LE(values []float32) ([]byte, error) {
	if len(values) > math.MaxInt/4 {
		return nil, errors.New("too many float32 values")
	}
	result := make([]byte, len(values)*4)
	for index, value := range values {
		binary.LittleEndian.PutUint32(result[index*4:], math.Float32bits(value))
	}
	return result, nil
}

func decodeFloat32LE(contents []byte) ([]float32, error) {
	if len(contents)%4 != 0 {
		return nil, fmt.Errorf("payload has %d bytes, not a whole number of float32 values", len(contents))
	}
	values := make([]float32, len(contents)/4)
	for index := range values {
		values[index] = math.Float32frombits(binary.LittleEndian.Uint32(contents[index*4:]))
	}
	return values, nil
}

// WriteFloat32LE writes a deterministic little-endian FP32 file and returns
// its SHA-256 digest.  CUDA model weights use the same representation as the
// golden-vector payload, but have their own manifest table and filename.
func WriteFloat32LE(path string, values []float32) (string, error) {
	contents, err := encodeFloat32LE(values)
	if err != nil {
		return "", err
	}
	if err := os.WriteFile(path, contents, 0o644); err != nil {
		return "", err
	}
	digest := sha256.Sum256(contents)
	return hex.EncodeToString(digest[:]), nil
}

// WriteJSON writes an indented JSON document atomically inside an already
// private staging directory.  The caller publishes that directory atomically.
func WriteJSON(path string, value any) error {
	contents, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return fmt.Errorf("encode JSON: %w", err)
	}
	contents = append(contents, '\n')
	if err := os.WriteFile(path, contents, 0o644); err != nil {
		return err
	}
	return nil
}

// FileSHA256 returns the lowercase SHA-256 of path without retaining a second
// copy of a potentially large export in memory.
func FileSHA256(path string) (string, error) {
	file, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer func() { _ = file.Close() }()
	hash := sha256.New()
	if _, err := io.Copy(hash, file); err != nil {
		return "", err
	}
	return hex.EncodeToString(hash.Sum(nil)), nil
}

package cudamodel

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

var testVocabularyHashes = VocabularyHashes{
	Solutions: strings.Repeat("a", sha256.Size*2),
	Actions:   strings.Repeat("b", sha256.Size*2),
}

func TestExpectedTensorsAreExactContiguousOutputMajorLayout(t *testing.T) {
	tensors := ExpectedTensors()
	if len(tensors) != TensorCount {
		t.Fatalf("got %d tensors, want %d", len(tensors), TensorCount)
	}
	if tensors[0].Name != CandidateProjectionWeight || !sameShape(tensors[0].Shape, []int{96, 2309}) {
		t.Fatalf("first tensor = %+v, want candidate projection output-major matrix", tensors[0])
	}
	if tensors[10].Name != BaseLogitsBias || !sameShape(tensors[10].Shape, []int{4739}) {
		t.Fatalf("eleventh tensor = %+v, want base logits bias", tensors[10])
	}
	if tensors[12].Name != CandidateBonusBias || tensors[12].Offset+tensors[12].Count != ParameterCount {
		t.Fatalf("last tensor = %+v, want final scalar at parameter count %d", tensors[12], ParameterCount)
	}
	for index, tensor := range tensors {
		if got := shapeSize(tensor.Shape); got != tensor.Count {
			t.Fatalf("tensor %q count = %d, shape size = %d", tensor.Name, tensor.Count, got)
		}
		if index > 0 && tensor.Offset != tensors[index-1].Offset+tensors[index-1].Count {
			t.Fatalf("tensor %q starts at %d after previous end %d", tensor.Name, tensor.Offset, tensors[index-1].Offset+tensors[index-1].Count)
		}
		if !strings.HasPrefix(tensor.SourceName, "var:/wordle_policy/") {
			t.Fatalf("tensor %q source name %q is not a checkpoint variable", tensor.Name, tensor.SourceName)
		}
	}

	// Returned layout values must be safe for an exporter to adapt locally.
	tensors[0].Shape[0] = 0
	if ExpectedTensors()[0].Shape[0] != CandidateProjectionSize {
		t.Fatal("ExpectedTensors exposed mutable global layout")
	}
}

func TestLoadAcceptsValidArtifactAndCopiesWeights(t *testing.T) {
	weights := deterministicWeights()
	dir := writeArtifact(t, validManifest(weights), weights)

	model, err := Load(dir, testVocabularyHashes)
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	if model.Manifest.Format != Format || model.Manifest.ParameterCount != ParameterCount {
		t.Fatalf("loaded manifest = %+v", model.Manifest)
	}
	got := model.Weights()
	if len(got) != ParameterCount || got[12345] != weights[12345] {
		t.Fatalf("weights copy has length %d and value %g, want %d and %g", len(got), got[12345], ParameterCount, weights[12345])
	}
	got[12345] = 123
	if model.Weights()[12345] == 123 {
		t.Fatal("Weights() exposed mutable model storage")
	}
}

func TestValidateManifestRejectsWrongVersionAndVocabularyHash(t *testing.T) {
	weights := zeroWeights()
	for name, mutate := range map[string]func(*Manifest){
		"format":              func(manifest *Manifest) { manifest.Format = "wordle-cuda-f32-v2" },
		"endianness":          func(manifest *Manifest) { manifest.Endianness = "big" },
		"dtype":               func(manifest *Manifest) { manifest.DType = "float16" },
		"solution vocabulary": func(manifest *Manifest) { manifest.SolutionVocabularySHA256 = strings.Repeat("c", sha256.Size*2) },
		"action vocabulary":   func(manifest *Manifest) { manifest.ActionVocabularySHA256 = strings.Repeat("c", sha256.Size*2) },
	} {
		t.Run(name, func(t *testing.T) {
			manifest := validManifest(weights)
			mutate(&manifest)
			if err := ValidateManifest(manifest, testVocabularyHashes); err == nil {
				t.Fatal("ValidateManifest() succeeded, want error")
			}
		})
	}
}

func TestValidateManifestRejectsTensorOrderShapeOffsetCountAndSource(t *testing.T) {
	weights := zeroWeights()
	for name, mutate := range map[string]func(*Manifest){
		"order": func(manifest *Manifest) {
			manifest.Tensors[0], manifest.Tensors[1] = manifest.Tensors[1], manifest.Tensors[0]
		},
		"shape":  func(manifest *Manifest) { manifest.Tensors[0].Shape = []int{NumSolutions, CandidateProjectionSize} },
		"offset": func(manifest *Manifest) { manifest.Tensors[5].Offset++ },
		"count":  func(manifest *Manifest) { manifest.Tensors[9].Count-- },
		"source": func(manifest *Manifest) { manifest.Tensors[12].SourceName = "var:/unexpected" },
	} {
		t.Run(name, func(t *testing.T) {
			manifest := validManifest(weights)
			mutate(&manifest)
			if err := ValidateManifest(manifest, testVocabularyHashes); err == nil {
				t.Fatal("ValidateManifest() succeeded, want error")
			}
		})
	}
}

func TestLoadRejectsTruncatedAndOverlongWeights(t *testing.T) {
	weights := zeroWeights()
	for name, mutate := range map[string]func([]byte) []byte{
		"truncated": func(data []byte) []byte { return data[:len(data)-1] },
		"overlong":  func(data []byte) []byte { return append(data, 0) },
	} {
		t.Run(name, func(t *testing.T) {
			dir := writeArtifact(t, validManifest(weights), weights)
			data, err := os.ReadFile(filepath.Join(dir, WeightsFilename))
			if err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(dir, WeightsFilename), mutate(data), 0o600); err != nil {
				t.Fatal(err)
			}
			if _, err := Load(dir, testVocabularyHashes); err == nil {
				t.Fatal("Load() succeeded, want exact-size error")
			}
		})
	}
}

func TestLoadRejectsWeightsHashAndNonFiniteWeight(t *testing.T) {
	t.Run("hash", func(t *testing.T) {
		weights := zeroWeights()
		dir := writeArtifact(t, validManifest(weights), weights)
		path := filepath.Join(dir, WeightsFilename)
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		data[0] = 1
		if err := os.WriteFile(path, data, 0o600); err != nil {
			t.Fatal(err)
		}
		if _, err := Load(dir, testVocabularyHashes); err == nil {
			t.Fatal("Load() succeeded, want SHA-256 error")
		}
	})

	t.Run("non-finite", func(t *testing.T) {
		weights := zeroWeights()
		weights[0] = float32(math.Inf(1))
		dir := writeArtifact(t, validManifest(weights), weights)
		if _, err := Load(dir, testVocabularyHashes); err == nil {
			t.Fatal("Load() succeeded, want non-finite-weight error")
		}
	})
}

func TestLoadRejectsManifestParameterCountAndUnknownJSONField(t *testing.T) {
	weights := zeroWeights()
	manifest := validManifest(weights)
	manifest.ParameterCount--
	if err := ValidateManifest(manifest, testVocabularyHashes); err == nil {
		t.Fatal("ValidateManifest() succeeded with wrong parameter count")
	}

	dir := writeArtifact(t, validManifest(weights), weights)
	path := filepath.Join(dir, "manifest.json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	data = append(data[:len(data)-1], []byte(",\"unrecognized\":true}")...)
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := Load(dir, testVocabularyHashes); err == nil {
		t.Fatal("Load() accepted an unknown manifest JSON field")
	}

	var fields map[string]json.RawMessage
	data, err = json.Marshal(validManifest(weights))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, &fields); err != nil {
		t.Fatal(err)
	}
	delete(fields, "checkpoint_update")
	data, err = json.Marshal(fields)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := Load(dir, testVocabularyHashes); err == nil {
		t.Fatal("Load() accepted a manifest missing checkpoint_update")
	}
}

func TestForwardImplementsOutputMajorPolicyAndExposesActivations(t *testing.T) {
	weights := zeroWeights()
	weights[tensorOffset(CandidateProjectionWeight)] = 2 // candidate 0 -> output 0
	weights[tensorOffset(CandidateProjectionBias)] = -0.5
	weights[tensorOffset(CandidateProjectionWeight)+NumSolutions+1] = 4 // candidate 1 -> output 1
	weights[tensorOffset(CandidateProjectionBias)+1] = -1
	weights[tensorOffset(StatsProjectionWeight)] = 3 // stat 0 -> output 0
	weights[tensorOffset(TurnEmbedding)+2*TurnEmbeddingSize] = 7
	weights[tensorOffset(BaseLogitsWeight)] = 1     // h[0]
	weights[tensorOffset(BaseLogitsWeight)+1] = 5   // h[1]
	weights[tensorOffset(BaseLogitsWeight)+96] = 2  // h[96]
	weights[tensorOffset(BaseLogitsWeight)+144] = 3 // h[144]
	weights[tensorOffset(BaseLogitsBias)] = 4
	weights[tensorOffset(BaseLogitsWeight)+TrunkSize+1] = 11 // output 1, h[1]
	weights[tensorOffset(BaseLogitsBias)+1] = 2
	weights[tensorOffset(ResidualInWeight)] = 2
	weights[tensorOffset(ResidualOutWeight)] = 3
	weights[tensorOffset(CandidateBonusWeight)] = 2
	weights[tensorOffset(CandidateBonusBias)] = 1
	model := &Model{weights: weights}

	inputs := validInputs()
	inputs.CandidateMask[1] = 1 // candidate 0 normalizes to 1/2
	inputs.CandidateStats[0] = 2
	inputs.Turn = 2
	inputs.RemainingActionMask[0] = 1
	inputs.RemainingActionMask[1] = 1

	logits, activations, err := model.ForwardWithActivations(inputs)
	if err != nil {
		t.Fatalf("ForwardWithActivations() error = %v", err)
	}
	if len(logits) != NumActions || logits[0] != 53.5 || logits[1] != 21 {
		t.Fatalf("logits[0:2] = %v, want [53.5 21]", logits[:2])
	}
	if got := activations["candidate_projection_relu"][0]; got != 0.5 {
		t.Fatalf("candidate projection = %g, want 0.5", got)
	}
	if got := activations["candidate_projection_relu"][1]; got != 1 {
		t.Fatalf("second candidate projection = %g, want 1", got)
	}
	if got := activations["stats_projection_relu"][0]; got != 6 {
		t.Fatalf("stats projection = %g, want 6", got)
	}
	if got := activations["turn_embedding"][0]; got != 7 {
		t.Fatalf("turn embedding = %g, want 7", got)
	}
	if got := activations["residual_in_relu"][0]; got != 1 {
		t.Fatalf("residual input = %g, want 1", got)
	}
	if got := activations["residual_out"][0]; got != 3 {
		t.Fatalf("residual output = %g, want 3", got)
	}
	if got := activations["hidden"][0]; got != 3.5 {
		t.Fatalf("hidden[0] = %g, want 3.5", got)
	}
	if got := activations["candidate_bonus"][0]; got != 8 {
		t.Fatalf("candidate bonus = %g, want 8", got)
	}
}

func TestForwardRejectsInvalidInputsAndNonFiniteOutputs(t *testing.T) {
	model := &Model{weights: zeroWeights()}
	for name, mutate := range map[string]func(*Inputs){
		"candidate mask length": func(inputs *Inputs) { inputs.CandidateMask = inputs.CandidateMask[:NumSolutions-1] },
		"stats length":          func(inputs *Inputs) { inputs.CandidateStats = inputs.CandidateStats[:CandidateStatsSize-1] },
		"action mask length":    func(inputs *Inputs) { inputs.RemainingActionMask = inputs.RemainingActionMask[:NumActions-1] },
		"turn":                  func(inputs *Inputs) { inputs.Turn = NumTurns },
		"empty candidates":      func(inputs *Inputs) { inputs.CandidateMask[0] = 0 },
		"nonfinite candidate":   func(inputs *Inputs) { inputs.CandidateMask[1] = float32(math.NaN()) },
		"nonfinite stats":       func(inputs *Inputs) { inputs.CandidateStats[0] = float32(math.Inf(1)) },
		"nonfinite action mask": func(inputs *Inputs) { inputs.RemainingActionMask[0] = float32(math.NaN()) },
	} {
		t.Run(name, func(t *testing.T) {
			inputs := validInputs()
			mutate(&inputs)
			if _, err := model.Forward(inputs); err == nil {
				t.Fatal("Forward() succeeded, want input-validation error")
			}
		})
	}

	weights := zeroWeights()
	weights[tensorOffset(BaseLogitsWeight)+144] = math.MaxFloat32
	model = &Model{weights: weights}
	inputs := validInputs()
	inputs.CandidateMask[1] = 1
	// Candidate features are zero, so make hidden non-zero via the turn embedding.
	weights[tensorOffset(TurnEmbedding)] = math.MaxFloat32
	inputs.Turn = 0
	if _, err := model.Forward(inputs); err == nil {
		t.Fatal("Forward() succeeded despite a non-finite output")
	}
}

func validManifest(weights []float32) Manifest {
	bytes := encodeWeights(weights)
	digest := sha256.Sum256(bytes)
	return Manifest{
		Format:                   Format,
		Endianness:               "little",
		DType:                    "float32",
		RunID:                    "test-run",
		Checkpoint:               "best",
		CheckpointUpdate:         2600,
		TrainingCommit:           "0123456789abcdef",
		NumSolutions:             NumSolutions,
		NumActions:               NumActions,
		CandidateStatsSize:       CandidateStatsSize,
		NumTurns:                 NumTurns,
		TrunkSize:                TrunkSize,
		ParameterCount:           ParameterCount,
		WeightsFile:              WeightsFilename,
		WeightsSHA256:            hex.EncodeToString(digest[:]),
		SolutionVocabularySHA256: testVocabularyHashes.Solutions,
		ActionVocabularySHA256:   testVocabularyHashes.Actions,
		Tensors:                  ExpectedTensors(),
	}
}

func writeArtifact(t *testing.T, manifest Manifest, weights []float32) string {
	t.Helper()
	dir := t.TempDir()
	manifestData, err := json.Marshal(manifest)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, ManifestFilename), manifestData, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, WeightsFilename), encodeWeights(weights), 0o600); err != nil {
		t.Fatal(err)
	}
	return dir
}

func zeroWeights() []float32 {
	return make([]float32, ParameterCount)
}

func deterministicWeights() []float32 {
	weights := zeroWeights()
	for index := range weights {
		weights[index] = float32((index%31)-15) / 100
	}
	return weights
}

func encodeWeights(weights []float32) []byte {
	data := make([]byte, len(weights)*4)
	for index, value := range weights {
		binary.LittleEndian.PutUint32(data[index*4:], math.Float32bits(value))
	}
	return data
}

func validInputs() Inputs {
	inputs := Inputs{
		CandidateMask:       make([]float32, NumSolutions),
		CandidateStats:      make([]float32, CandidateStatsSize),
		RemainingActionMask: make([]float32, NumActions),
	}
	inputs.CandidateMask[0] = 1
	return inputs
}

func shapeSize(shape []int) int {
	size := 1
	for _, dimension := range shape {
		size *= dimension
	}
	return size
}

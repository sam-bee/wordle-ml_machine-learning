// Package cudamodel reads and evaluates the fixed, portable CUDA model format.
//
// It deliberately contains no GoMLX dependency. The evaluator is a test and
// verification aid for the exported FP32 model, not a serving fallback.
package cudamodel

import "github.com/sam-bee/wordle-ml_machine-learning/modelstate"

const (
	// Format is the sole supported portable model-artifact format.
	Format = "wordle-cuda-f32-v1"

	// ManifestFilename is the fixed JSON descriptor name within an artifact.
	ManifestFilename = "manifest.json"

	// WeightsFilename is the fixed name of the little-endian FP32 payload.
	WeightsFilename = "weights.f32le"

	NumSolutions            = 2309
	NumActions              = 4739
	CandidateStatsSize      = 209
	NumTurns                = 6
	CandidateProjectionSize = 96
	StatsProjectionSize     = 48
	TurnEmbeddingSize       = 16
	TrunkSize               = 160
	ParameterCount          = 1046596
	TensorCount             = 13
	WeightsByteCount        = ParameterCount * 4
)

const (
	CandidateProjectionWeight = "candidate_projection.weight"
	CandidateProjectionBias   = "candidate_projection.bias"
	StatsProjectionWeight     = "stats_projection.weight"
	StatsProjectionBias       = "stats_projection.bias"
	TurnEmbedding             = "turn_embedding"
	ResidualInWeight          = "residual_in.weight"
	ResidualInBias            = "residual_in.bias"
	ResidualOutWeight         = "residual_out.weight"
	ResidualOutBias           = "residual_out.bias"
	BaseLogitsWeight          = "base_logits.weight"
	BaseLogitsBias            = "base_logits.bias"
	CandidateBonusWeight      = "candidate_bonus.weight"
	CandidateBonusBias        = "candidate_bonus.bias"
)

// Inputs is the fixed set of host inputs to this policy. It is an alias so
// callers that already use the frozen board-state encoder do not need a copy.
type Inputs = modelstate.Inputs

// VocabularyHashes identifies the two frozen ordered vocabularies that are
// encoded in an exported artifact. Both values must be lowercase SHA-256 hex
// digests supplied by the caller that loaded the local vocabulary.
type VocabularyHashes struct {
	Solutions string
	Actions   string
}

// Manifest describes one version-1 CUDA model payload. Weights are deliberately
// separate from JSON so the native backend can receive one contiguous FP32
// allocation without having to parse this manifest.
type Manifest struct {
	Format                   string   `json:"format"`
	Endianness               string   `json:"endianness"`
	DType                    string   `json:"dtype"`
	RunID                    string   `json:"run_id"`
	Checkpoint               string   `json:"checkpoint"`
	CheckpointUpdate         int      `json:"checkpoint_update"`
	TrainingCommit           string   `json:"training_commit"`
	NumSolutions             int      `json:"num_solutions"`
	NumActions               int      `json:"num_actions"`
	CandidateStatsSize       int      `json:"candidate_stats_size"`
	NumTurns                 int      `json:"num_turns"`
	TrunkSize                int      `json:"trunk_size"`
	ParameterCount           int      `json:"parameter_count"`
	WeightsFile              string   `json:"weights_file"`
	WeightsSHA256            string   `json:"weights_sha256"`
	SolutionVocabularySHA256 string   `json:"solution_vocabulary_sha256"`
	ActionVocabularySHA256   string   `json:"action_vocabulary_sha256"`
	Tensors                  []Tensor `json:"tensors"`
}

// Tensor locates one tensor in the weight payload. Offset and Count are in
// float32 elements, not bytes. SourceName is the exact source checkpoint
// variable name used by the exporter before any dense-weight transposition.
type Tensor struct {
	Name       string `json:"name"`
	Shape      []int  `json:"shape"`
	Offset     int    `json:"offset"`
	Count      int    `json:"count"`
	SourceName string `json:"source_name"`
}

// ExpectedTensors returns the immutable logical layout of every v1 tensor.
// Matrix rows are outputs: a matrix [outputs, inputs] is adjacent in memory by
// input within each output row. Exporters, loaders, and native offset audits
// must use this exact order.
func ExpectedTensors() []Tensor {
	return cloneTensors(expectedTensors)
}

var expectedTensors = []Tensor{
	{
		Name: CandidateProjectionWeight, Shape: []int{CandidateProjectionSize, NumSolutions}, Offset: 0,
		Count:      CandidateProjectionSize * NumSolutions,
		SourceName: "var:/wordle_policy/candidate_projection/dense/weights",
	},
	{
		Name: CandidateProjectionBias, Shape: []int{CandidateProjectionSize}, Offset: CandidateProjectionSize * NumSolutions,
		Count:      CandidateProjectionSize,
		SourceName: "var:/wordle_policy/candidate_projection/dense/biases",
	},
	{
		Name: StatsProjectionWeight, Shape: []int{StatsProjectionSize, CandidateStatsSize}, Offset: 221760,
		Count:      StatsProjectionSize * CandidateStatsSize,
		SourceName: "var:/wordle_policy/stats_projection/dense/weights",
	},
	{
		Name: StatsProjectionBias, Shape: []int{StatsProjectionSize}, Offset: 231792,
		Count:      StatsProjectionSize,
		SourceName: "var:/wordle_policy/stats_projection/dense/biases",
	},
	{
		Name: TurnEmbedding, Shape: []int{NumTurns, TurnEmbeddingSize}, Offset: 231840,
		Count:      NumTurns * TurnEmbeddingSize,
		SourceName: "var:/wordle_policy/turn_embedding/embeddings",
	},
	{
		Name: ResidualInWeight, Shape: []int{TrunkSize, TrunkSize}, Offset: 231936,
		Count:      TrunkSize * TrunkSize,
		SourceName: "var:/wordle_policy/residual_in/dense/weights",
	},
	{
		Name: ResidualInBias, Shape: []int{TrunkSize}, Offset: 257536,
		Count:      TrunkSize,
		SourceName: "var:/wordle_policy/residual_in/dense/biases",
	},
	{
		Name: ResidualOutWeight, Shape: []int{TrunkSize, TrunkSize}, Offset: 257696,
		Count:      TrunkSize * TrunkSize,
		SourceName: "var:/wordle_policy/residual_out/dense/weights",
	},
	{
		Name: ResidualOutBias, Shape: []int{TrunkSize}, Offset: 283296,
		Count:      TrunkSize,
		SourceName: "var:/wordle_policy/residual_out/dense/biases",
	},
	{
		Name: BaseLogitsWeight, Shape: []int{NumActions, TrunkSize}, Offset: 283456,
		Count:      NumActions * TrunkSize,
		SourceName: "var:/wordle_policy/base_logits/dense/weights",
	},
	{
		Name: BaseLogitsBias, Shape: []int{NumActions}, Offset: 1041696,
		Count:      NumActions,
		SourceName: "var:/wordle_policy/base_logits/dense/biases",
	},
	{
		Name: CandidateBonusWeight, Shape: []int{1, TrunkSize}, Offset: 1046435,
		Count:      TrunkSize,
		SourceName: "var:/wordle_policy/candidate_bonus/dense/weights",
	},
	{
		Name: CandidateBonusBias, Shape: []int{1}, Offset: 1046595,
		Count:      1,
		SourceName: "var:/wordle_policy/candidate_bonus/dense/biases",
	},
}

func cloneTensors(tensors []Tensor) []Tensor {
	clone := make([]Tensor, len(tensors))
	for i, tensor := range tensors {
		clone[i] = tensor
		clone[i].Shape = append([]int(nil), tensor.Shape...)
	}
	return clone
}

func cloneManifest(manifest Manifest) Manifest {
	clone := manifest
	clone.Tensors = cloneTensors(manifest.Tensors)
	return clone
}

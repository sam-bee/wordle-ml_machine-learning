// Package cudaref owns the GoMLX-free reference sidecars accompanying an
// exported CUDA model.  It deliberately knows nothing about checkpoints or
// GoMLX: the exporter supplies already materialised inputs and raw logits,
// while CUDA verification later reads the same neutral files.
package cudaref

import (
	"fmt"
	"math"

	"github.com/sam-bee/wordle-ml_machine-learning/gameeval"
	"github.com/sam-bee/wordle-ml_machine-learning/modelstate"
	"github.com/sam-bee/wordle-ml_machine-learning/vocabulary"
)

const (
	// GoldenFormat identifies golden-vectors.json and its companion binary
	// data.  The numeric payload is little-endian float32 values.
	GoldenFormat = "wordle-cuda-golden-v1"
	// GoldenValuesFilename is the compact payload containing all vector arrays.
	GoldenValuesFilename = "golden-vectors.f32le"
	// GoldenManifestFilename describes the vectors and binary spans.
	GoldenManifestFilename = "golden-vectors.json"
	// GoldenGamesFilename records the reference validation-game trajectories.
	GoldenGamesFilename = "golden-games.json"
	// ExportReportFilename records export validation and parity measurements.
	ExportReportFilename = "export-report.json"
)

// Span points at a consecutive set of little-endian float32 values in
// golden-vectors.f32le.  Offsets are in float values rather than bytes so they
// are easy to inspect beside the fixed model tensor layout.
type Span struct {
	OffsetFloats int `json:"offset_floats"`
	Count        int `json:"count"`
}

// Vector is one reference model call plus the Go-owned action selection which
// followed it.  Inputs and logits are only held in memory; WriteGoldenVectors
// stores their large float slices in the companion binary file.
type Vector struct {
	ID                  string
	Inputs              modelstate.Inputs
	AvailableActionMask []float32
	RawLogits           []float32
	RawTopActionID      int
	SelectedActionID    int
	TopTwoMargin        float32
	Provenance          Provenance
}

// Provenance makes a golden position understandable without consulting a
// final-test game.  It intentionally contains only validation gameplay.
type Provenance struct {
	Solution             string                 `json:"solution"`
	Turn                 int                    `json:"turn"`
	CandidateCount       int                    `json:"candidate_count"`
	History              []gameeval.HistoryTurn `json:"history,omitempty"`
	RawTopWasAvailable   bool                   `json:"raw_top_was_available"`
	SelectedWasCandidate bool                   `json:"selected_was_candidate"`
	SelectionKind        string                 `json:"selection_kind"`
	ShortlistSizeBefore  int                    `json:"shortlist_size_before"`
	ShortlistSizeAfter   int                    `json:"shortlist_size_after"`
}

// VectorMetadata is the inspectable JSON representation of Vector.  Array
// values live in the spans within GoldenValuesFilename.
type VectorMetadata struct {
	ID                  string     `json:"id"`
	Turn                int32      `json:"turn"`
	CandidateMask       Span       `json:"candidate_mask"`
	CandidateStats      Span       `json:"candidate_stats"`
	RemainingActionMask Span       `json:"remaining_action_mask"`
	AvailableActionMask Span       `json:"available_action_mask"`
	RawLogits           Span       `json:"raw_logits"`
	RawTopActionID      int        `json:"raw_top_action_id"`
	SelectedActionID    int        `json:"selected_action_id"`
	TopTwoMargin        float32    `json:"top_two_margin"`
	Provenance          Provenance `json:"provenance"`
}

// GoldenManifest is the versioned JSON index for a deterministic vector set.
type GoldenManifest struct {
	Format       string           `json:"format"`
	Endianness   string           `json:"endianness"`
	DType        string           `json:"dtype"`
	ValuesFile   string           `json:"values_file"`
	ValuesSHA256 string           `json:"values_sha256"`
	ValuesCount  int              `json:"values_count"`
	Vectors      []VectorMetadata `json:"vectors"`
}

// GoldenSet holds a decoded golden manifest and its vectors.
type GoldenSet struct {
	Manifest GoldenManifest
	Vectors  []Vector
}

// Games is the reference result for the fixed validation population.  It is
// kept separate from GoldenManifest because it is substantially larger and is
// independently useful for end-to-end CUDA trajectory verification.
type Games struct {
	RunID            string              `json:"run_id"`
	Checkpoint       string              `json:"checkpoint"`
	CheckpointUpdate int64               `json:"checkpoint_update"`
	Evaluation       gameeval.Evaluation `json:"evaluation"`
}

// Comparison summarizes a raw-logit comparison without burying large vectors
// in export-report.json.
type Comparison struct {
	Compared          int     `json:"compared"`
	MaximumAbsolute   float64 `json:"maximum_absolute_error"`
	MeanAbsolute      float64 `json:"mean_absolute_error"`
	WorstVectorID     string  `json:"worst_vector_id,omitempty"`
	WorstActionID     int     `json:"worst_action_id,omitempty"`
	Top1Agreement     int     `json:"top1_agreement"`
	Top5SetAgreement  int     `json:"top5_set_agreement"`
	ActionComparisons int     `json:"action_comparisons"`
}

// ExportReport documents the completed offline conversion and the initial
// GoMLX-to-portable evaluator gate.  All fields are deliberately values that a
// verifier can reproduce from the neutral artifact.
type ExportReport struct {
	Format             string     `json:"format"`
	RunID              string     `json:"run_id"`
	Checkpoint         string     `json:"checkpoint"`
	CheckpointUpdate   int64      `json:"checkpoint_update"`
	CheckpointIdentity string     `json:"checkpoint_identity"`
	TrainingCommit     string     `json:"training_commit"`
	ParameterCount     int        `json:"parameter_count"`
	WeightsSHA256      string     `json:"weights_sha256"`
	GoldenVectorCount  int        `json:"golden_vector_count"`
	GoldenGames        string     `json:"golden_games"`
	PortableComparison Comparison `json:"gomlx_vs_portable_go"`
}

// ValidateVector rejects malformed model calls before they are written.  It
// deliberately does not require masks to be binary, since the neutral model
// artifact records the model contract rather than a second host encoder.
func ValidateVector(vector Vector) error {
	if vector.ID == "" {
		return fmt.Errorf("golden vector ID is required")
	}
	if len(vector.Inputs.CandidateMask) != vocabulary.NumSolutions {
		return fmt.Errorf("vector %q candidate mask has %d values, want %d", vector.ID, len(vector.Inputs.CandidateMask), vocabulary.NumSolutions)
	}
	if len(vector.Inputs.CandidateStats) != modelstate.CandidateStatsSize {
		return fmt.Errorf("vector %q candidate stats has %d values, want %d", vector.ID, len(vector.Inputs.CandidateStats), modelstate.CandidateStatsSize)
	}
	if vector.Inputs.Turn < 0 || vector.Inputs.Turn >= 6 {
		return fmt.Errorf("vector %q turn %d is outside 0..5", vector.ID, vector.Inputs.Turn)
	}
	if len(vector.Inputs.RemainingActionMask) != vocabulary.NumActions {
		return fmt.Errorf("vector %q remaining action mask has %d values, want %d", vector.ID, len(vector.Inputs.RemainingActionMask), vocabulary.NumActions)
	}
	if len(vector.AvailableActionMask) != vocabulary.NumActions {
		return fmt.Errorf("vector %q availability mask has %d values, want %d", vector.ID, len(vector.AvailableActionMask), vocabulary.NumActions)
	}
	if len(vector.RawLogits) != vocabulary.NumActions {
		return fmt.Errorf("vector %q logits have %d values, want %d", vector.ID, len(vector.RawLogits), vocabulary.NumActions)
	}
	if vector.RawTopActionID < 0 || vector.RawTopActionID >= vocabulary.NumActions {
		return fmt.Errorf("vector %q raw top action %d is outside action vocabulary", vector.ID, vector.RawTopActionID)
	}
	if vector.SelectedActionID < 0 || vector.SelectedActionID >= vocabulary.NumActions {
		return fmt.Errorf("vector %q selected action %d is outside action vocabulary", vector.ID, vector.SelectedActionID)
	}
	if vector.Provenance.CandidateCount <= 0 {
		return fmt.Errorf("vector %q candidate count must be positive", vector.ID)
	}
	candidateSum := float32(0)
	for _, values := range [][]float32{vector.Inputs.CandidateMask, vector.Inputs.CandidateStats, vector.Inputs.RemainingActionMask, vector.AvailableActionMask, vector.RawLogits} {
		for _, value := range values {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return fmt.Errorf("vector %q contains a non-finite float", vector.ID)
			}
		}
	}
	for _, value := range vector.Inputs.CandidateMask {
		candidateSum += value
	}
	if candidateSum <= 0 {
		return fmt.Errorf("vector %q candidate mask is empty", vector.ID)
	}
	rawTop, selected, err := SelectAvailableAction(vector.RawLogits, vector.AvailableActionMask)
	if err != nil {
		return fmt.Errorf("vector %q select action: %w", vector.ID, err)
	}
	if vector.RawTopActionID != rawTop || vector.SelectedActionID != selected {
		return fmt.Errorf("vector %q stores raw/selected actions %d/%d, want %d/%d", vector.ID, vector.RawTopActionID, vector.SelectedActionID, rawTop, selected)
	}
	margin, err := TopTwoMargin(vector.RawLogits)
	if err != nil {
		return fmt.Errorf("vector %q top-two margin: %w", vector.ID, err)
	}
	if vector.TopTwoMargin != margin {
		return fmt.Errorf("vector %q stores top-two margin %g, want %g", vector.ID, vector.TopTwoMargin, margin)
	}
	if vector.Provenance.Turn != int(vector.Inputs.Turn) {
		return fmt.Errorf("vector %q provenance turn %d differs from input turn %d", vector.ID, vector.Provenance.Turn, vector.Inputs.Turn)
	}
	return nil
}

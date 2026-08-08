// Package imitationdata loads frozen WDIT v3 teacher records and exposes them
// as the policy inputs used for imitation learning.
package imitationdata

import (
	"encoding/hex"
	"encoding/json"
	"fmt"
	"iter"
	"math/rand"
	"os"
	"path/filepath"
	"sort"

	"github.com/gomlx/compute"
	"github.com/gomlx/gomlx/core/tensors"
	gomlxdataset "github.com/gomlx/gomlx/ml/dataset"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/sam-bee/wordle-ml_game-engine/words"
	"github.com/sam-bee/wordle-ml_machine-learning/modelstate"
	"github.com/sam-bee/wordle-ml_machine-learning/vocabulary"
	synthetic "github.com/sam-bee/wordle-ml_synthetic-data-creation/dataset"
)

// Split identifies one frozen imitation-data file.
type Split string

const (
	Train      Split = "train"
	Validation Split = "validation"
	Test       Split = "test"
	Mini       Split = "mini"
)

// Data owns compact WDIT records. Tensor-sized values are expanded only when
// Example or Dataset needs them.
type Data struct {
	vocabulary *vocabulary.Vocabulary
	encoder    *modelstate.Encoder
	split      Split
	header     synthetic.BinaryHeader
	metadata   synthetic.Metadata
	records    []synthetic.Record
}

// Example is one expanded model state. RemainingActionMask is a learned
// candidate bonus input, while AvailableActionMask only excludes guesses that
// already appear in History.
type Example struct {
	Record              synthetic.Record
	CandidateMask       []float32
	CandidateStats      []float32
	Turn                int32
	RemainingActionMask []float32
	AvailableActionMask []float32
	TeacherTopAction    int32
}

// Load reads and validates the binary dataset and its required JSON sidecar.
// dataDir contains wordle-<split>.bin and wordle-<split>.json.
func Load(v *vocabulary.Vocabulary, dataDir string, split Split) (*Data, error) {
	if v == nil {
		return nil, fmt.Errorf("vocabulary must not be nil")
	}
	syntheticSplit, expectedIDs, err := expectedSplit(v, split)
	if err != nil {
		return nil, err
	}
	stem := "wordle-" + string(split)
	binaryPath := filepath.Join(dataDir, stem+".bin")
	metadataPath := filepath.Join(dataDir, stem+".json")

	decoded, err := synthetic.ReadBinaryFile(binaryPath)
	if err != nil {
		return nil, err
	}
	syntheticVocabulary, err := syntheticVocabulary(v)
	if err != nil {
		return nil, err
	}
	if err := decoded.ValidateSplit(syntheticVocabulary, syntheticSplit); err != nil {
		return nil, fmt.Errorf("validate %s: %w", binaryPath, err)
	}
	contents, err := os.ReadFile(metadataPath)
	if err != nil {
		return nil, fmt.Errorf("read metadata %q: %w", metadataPath, err)
	}
	var metadata synthetic.Metadata
	if err := json.Unmarshal(contents, &metadata); err != nil {
		return nil, fmt.Errorf("decode metadata %q: %w", metadataPath, err)
	}
	if err := validateMetadata(metadata, decoded, split, stem+".bin", expectedIDs); err != nil {
		return nil, fmt.Errorf("validate metadata %q: %w", metadataPath, err)
	}
	encoder, err := modelstate.NewEncoder(v)
	if err != nil {
		return nil, err
	}
	return &Data{vocabulary: v, encoder: encoder, split: split, header: decoded.Header, metadata: metadata, records: decoded.Records}, nil
}

// Split returns the loaded split.
func (d *Data) Split() Split { return d.split }

// Header returns the decoded WDIT v3 header.
func (d *Data) Header() synthetic.BinaryHeader { return d.header }

// Metadata returns the validated JSON sidecar.
func (d *Data) Metadata() synthetic.Metadata { return d.metadata }

// Len returns the number of compact teacher records.
func (d *Data) Len() int { return len(d.records) }

// Record returns one compact WDIT record without model-sized expansion.
func (d *Data) Record(index int) (synthetic.Record, error) {
	if index < 0 || index >= len(d.records) {
		return synthetic.Record{}, fmt.Errorf("record index %d outside 0..%d", index, len(d.records)-1)
	}
	return d.records[index], nil
}

// Example expands one record through the shared model-state encoder.
func (d *Data) Example(index int) (Example, error) {
	record, err := d.Record(index)
	if err != nil {
		return Example{}, err
	}
	inputs, err := d.encoder.Encode(record.CandidateBits[:], int(record.TurnDepth))
	if err != nil {
		return Example{}, fmt.Errorf("encode record %d: %w", index, err)
	}
	available := make([]float32, vocabulary.NumActions)
	for actionID := range available {
		available[actionID] = 1
	}
	for historyIndex := 0; historyIndex < int(record.TurnDepth); historyIndex++ {
		available[record.History[historyIndex].GuessID] = 0
	}
	return Example{
		Record:              record,
		CandidateMask:       inputs.CandidateMask,
		CandidateStats:      inputs.CandidateStats,
		Turn:                inputs.Turn,
		RemainingActionMask: inputs.RemainingActionMask,
		AvailableActionMask: available,
		TeacherTopAction:    int32(record.TopKActionIDs[0]),
	}, nil
}

// IndexOrder returns every record index in a deterministic shuffled order.
func (d *Data) IndexOrder(seed int64) []int {
	order := make([]int, len(d.records))
	for i := range order {
		order[i] = i
	}
	rand.New(rand.NewSource(seed)).Shuffle(len(order), func(i, j int) { order[i], order[j] = order[j], order[i] })
	return order
}

// Dataset returns an unbatched GoMLX dataset in deterministic shuffled order.
func (d *Data) Dataset(seed int64) train.Dataset {
	return &dataset{data: d, order: d.IndexOrder(seed)}
}

// Batch is the thin standard GoMLX batching wrapper for this package's
// unbatched Dataset. Its five inputs have shapes [B,2309], [B,209], [B],
// [B,4739], [B,4739], and its label has shape [B,1].
func Batch(backend compute.Backend, source train.Dataset, batchSize int, dropIncomplete bool) train.Dataset {
	return gomlxdataset.Batch(backend, source, batchSize, true, dropIncomplete)
}

type dataset struct {
	data  *Data
	order []int
}

func (ds *dataset) Name() string { return "wordle-imitation-" + string(ds.data.split) }

func (ds *dataset) Iter() iter.Seq2[train.Batch, error] {
	return func(yield func(train.Batch, error) bool) {
		for _, index := range ds.order {
			example, err := ds.data.Example(index)
			if err != nil {
				yield(train.Batch{}, err)
				return
			}
			batch := train.Batch{
				Inputs: []*tensors.Tensor{
					tensors.FromFlatDataAndDimensions(example.CandidateMask, vocabulary.NumSolutions),
					tensors.FromFlatDataAndDimensions(example.CandidateStats, modelstate.CandidateStatsSize),
					tensors.FromScalar(example.Turn),
					tensors.FromFlatDataAndDimensions(example.RemainingActionMask, vocabulary.NumActions),
					tensors.FromFlatDataAndDimensions(example.AvailableActionMask, vocabulary.NumActions),
				},
				Labels: []*tensors.Tensor{tensors.FromFlatDataAndDimensions([]int32{example.TeacherTopAction}, 1)},
			}
			if !yield(batch, nil) {
				return
			}
		}
	}
}

func expectedSplit(v *vocabulary.Vocabulary, split Split) (synthetic.Split, []uint16, error) {
	var wordsInSplit []string
	var id synthetic.SplitID
	switch split {
	case Train:
		wordsInSplit, id = v.Training(), synthetic.SplitTrain
	case Validation:
		wordsInSplit, id = v.Validation(), synthetic.SplitValidation
	case Test:
		wordsInSplit, id = v.Test(), synthetic.SplitTest
	case Mini:
		training := v.Training()
		wordsInSplit, id = training[:64], synthetic.SplitMini
	default:
		return synthetic.Split{}, nil, fmt.Errorf("unknown imitation-data split %q", split)
	}
	ids := make([]uint16, len(wordsInSplit))
	for i, word := range wordsInSplit {
		solutionID, ok := v.SolutionID(word)
		if !ok {
			return synthetic.Split{}, nil, fmt.Errorf("%s split word %q is not a solution", split, word)
		}
		ids[i] = uint16(solutionID)
	}
	sort.Slice(ids, func(i, j int) bool { return ids[i] < ids[j] })
	return synthetic.Split{ID: id, SolutionIDs: ids}, ids, nil
}

func syntheticVocabulary(v *vocabulary.Vocabulary) (*synthetic.Vocabulary, error) {
	actions := v.Actions()
	solutions := v.Solutions()
	guesses := make([]words.Word, len(actions))
	for i, word := range actions {
		guesses[i] = words.Word(word)
	}
	answers := make([]words.Word, len(solutions))
	for i, word := range solutions {
		answers[i] = words.Word(word)
	}
	return synthetic.NewVocabulary(guesses, answers)
}

func validateMetadata(m synthetic.Metadata, decoded synthetic.BinaryDataset, split Split, basename string, expectedIDs []uint16) error {
	h := decoded.Header
	if m.Version != int(h.Version) || m.Version != synthetic.FormatVersion {
		return fmt.Errorf("version %d does not match header version %d", m.Version, h.Version)
	}
	if m.Split != string(split) || m.SplitID != h.SplitID || m.Split != h.Split {
		return fmt.Errorf("split %q/%d does not match header %q/%d", m.Split, m.SplitID, h.Split, h.SplitID)
	}
	if m.BinaryFile != basename {
		return fmt.Errorf("binary_file %q, want %q", m.BinaryFile, basename)
	}
	if m.RecordCount != int(h.RecordCount) || m.RecordCount != len(decoded.Records) {
		return fmt.Errorf("record_count %d does not match header/records", m.RecordCount)
	}
	if m.HeaderSizeBytes != int(h.HeaderSize) || m.RecordSizeBytes != int(h.RecordSize) ||
		m.TopK != int(h.TopK) || m.MaxTurns != int(h.MaxTurns) ||
		m.GlobalActionCount != int(h.GlobalActionCount) || m.GlobalSolutionCount != int(h.GlobalSolutionCount) ||
		m.SplitSolutionCount != int(h.SplitSolutionCount) {
		return fmt.Errorf("metadata dimensions do not match binary header")
	}
	if m.TopK != synthetic.FixedTopK || m.MaxTurns != synthetic.MaxDepth ||
		m.GlobalActionCount != vocabulary.NumActions || m.GlobalSolutionCount != vocabulary.NumSolutions {
		return fmt.Errorf("metadata dimensions do not match the fixed policy contract")
	}
	if !sort.SliceIsSorted(m.SolutionIDs, func(i, j int) bool { return m.SolutionIDs[i] < m.SolutionIDs[j] }) {
		return fmt.Errorf("solution_ids are not sorted")
	}
	if len(m.SolutionIDs) != len(expectedIDs) {
		return fmt.Errorf("solution_ids has %d IDs, want %d", len(m.SolutionIDs), len(expectedIDs))
	}
	for i, id := range expectedIDs {
		if m.SolutionIDs[i] != id {
			return fmt.Errorf("solution_ids[%d] = %d, want %d", i, m.SolutionIDs[i], id)
		}
	}
	if m.IncludesOpeningState != (countOpenings(decoded.Records) == 1) {
		return fmt.Errorf("includes_opening_state=%t does not match %d opening records", m.IncludesOpeningState, countOpenings(decoded.Records))
	}
	if countOpenings(decoded.Records) > 1 || (split != Train && countOpenings(decoded.Records) != 0) {
		return fmt.Errorf("split %s has invalid opening-record count %d", split, countOpenings(decoded.Records))
	}
	if m.OpeningSolutionID != synthetic.OpeningSolutionID || m.PaddingActionID != synthetic.PaddingActionID || m.PaddingFeedbackValue != synthetic.PaddingFeedbackValue {
		return fmt.Errorf("metadata encoding sentinels do not match WDIT v3")
	}
	if m.ActionVocabularySHA256 != h.ActionVocabularyHash() || m.SolutionVocabularySHA256 != h.SolutionVocabularyHash() || m.SplitMembershipSHA256 != h.SplitMembershipHash() {
		return fmt.Errorf("metadata hashes do not match binary header")
	}
	for name, hash := range map[string]string{
		"action vocabulary":   m.ActionVocabularySHA256,
		"solution vocabulary": m.SolutionVocabularySHA256,
		"split membership":    m.SplitMembershipSHA256,
	} {
		if _, err := hex.DecodeString(hash); err != nil || len(hash) != 64 {
			return fmt.Errorf("invalid %s SHA-256", name)
		}
	}
	return nil
}

func countOpenings(records []synthetic.Record) int {
	count := 0
	for _, record := range records {
		if record.SolutionID == synthetic.OpeningSolutionID {
			count++
		}
	}
	return count
}

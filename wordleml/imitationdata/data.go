// Package imitationdata loads frozen WDIT v3 teacher records and exposes them
// as the policy inputs used for imitation learning.
package imitationdata

import (
	"encoding/hex"
	"encoding/json"
	"errors"
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
	RecordIndex         int
	Record              synthetic.Record
	CandidateMask       []float32
	CandidateStats      []float32
	Turn                int32
	RemainingActionMask []float32
	AvailableActionMask []float32
	TeacherTopAction    int32
	TeacherTopActions   [16]int32
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
	var teacherTopActions [16]int32
	for rank, actionID := range record.TopKActionIDs {
		teacherTopActions[rank] = int32(actionID)
		if actionID >= vocabulary.NumActions {
			return Example{}, fmt.Errorf("record %d teacher action at rank %d is outside the vocabulary: %d", index, rank+1, actionID)
		}
		if available[actionID] == 0 {
			return Example{}, fmt.Errorf("record %d teacher action at rank %d was already guessed: %d", index, rank+1, actionID)
		}
	}
	return Example{
		RecordIndex:         index,
		Record:              record,
		CandidateMask:       inputs.CandidateMask,
		CandidateStats:      inputs.CandidateStats,
		Turn:                inputs.Turn,
		RemainingActionMask: inputs.RemainingActionMask,
		AvailableActionMask: available,
		TeacherTopAction:    int32(record.TopKActionIDs[0]),
		TeacherTopActions:   teacherTopActions,
	}, nil
}

// FindOpening returns the single empty-board training example. Other splits do
// not contain it. A malformed training split with zero or multiple openings is
// rejected rather than silently weakening opening-state sampling.
func (d *Data) FindOpening() (Example, bool, error) {
	openingIndex := -1
	for index, record := range d.records {
		if record.SolutionID != synthetic.OpeningSolutionID {
			continue
		}
		if openingIndex >= 0 {
			return Example{}, false, fmt.Errorf("split %s contains multiple opening records", d.split)
		}
		openingIndex = index
	}
	if openingIndex < 0 {
		if d.split == Train {
			return Example{}, false, errors.New("training split contains no opening record")
		}
		return Example{}, false, nil
	}
	opening, err := d.Example(openingIndex)
	if err != nil {
		return Example{}, false, err
	}
	return opening, true, nil
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
// [B,4739], [B,4739], and its loss label has shape [B,1]. The proof
// runner uses MaterializeBatch when it also needs the teacher's top 16.
func Batch(backend compute.Backend, source train.Dataset, batchSize int, dropIncomplete bool) train.Dataset {
	return gomlxdataset.Batch(backend, source, batchSize, true, dropIncomplete)
}

// Cursor identifies the next non-opening source record in a deterministic
// shuffled epoch. Offset counts records after the source's opening record has
// been removed from the order.
type Cursor struct {
	Epoch  int64 `json:"epoch"`
	Offset int   `json:"offset"`
}

// TrainingSampler builds fixed-size batches with the empty-board example in
// the first slot and shuffled source records in every other slot. Its cursor is
// sufficient to reproduce the exact next examples after a restart.
type TrainingSampler struct {
	source    *Data
	opening   Example
	batchSize int
	seed      int64
	cursor    Cursor
	order     []int
}

// NewTrainingSampler creates a sampler at cursor. The source may be the mini
// split (which has no opening) or the full training split (whose own opening is
// removed before shuffling).
func NewTrainingSampler(source *Data, opening Example, batchSize int, seed int64, cursor Cursor) (*TrainingSampler, error) {
	if source == nil {
		return nil, errors.New("training source must not be nil")
	}
	if source.split != Train && source.split != Mini {
		return nil, fmt.Errorf("split %s cannot be used as a training source", source.split)
	}
	if opening.Record.SolutionID != synthetic.OpeningSolutionID || opening.Record.TurnDepth != 0 {
		return nil, errors.New("opening example is not the empty-board record")
	}
	if len(opening.CandidateMask) != vocabulary.NumSolutions ||
		len(opening.CandidateStats) != modelstate.CandidateStatsSize ||
		len(opening.RemainingActionMask) != vocabulary.NumActions ||
		len(opening.AvailableActionMask) != vocabulary.NumActions {
		return nil, errors.New("opening example has invalid fixed input dimensions")
	}
	if batchSize < 2 {
		return nil, fmt.Errorf("training batch size must be at least 2, got %d", batchSize)
	}
	if seed == 0 {
		return nil, errors.New("training shuffle seed must not be zero")
	}
	if cursor.Epoch < 0 || cursor.Offset < 0 {
		return nil, fmt.Errorf("training cursor must not be negative: %+v", cursor)
	}
	sampler := &TrainingSampler{
		source:    source,
		opening:   opening,
		batchSize: batchSize,
		seed:      seed,
		cursor:    cursor,
	}
	sampler.order = sampler.orderForEpoch(cursor.Epoch)
	if len(sampler.order) < batchSize-1 {
		return nil, fmt.Errorf("training source has %d non-opening records, need at least %d", len(sampler.order), batchSize-1)
	}
	if cursor.Offset > len(sampler.order) {
		return nil, fmt.Errorf("training cursor offset %d exceeds epoch size %d", cursor.Offset, len(sampler.order))
	}
	sampler.normalizeCursor()
	return sampler, nil
}

// Cursor returns the normalized position of the next non-opening example.
func (s *TrainingSampler) Cursor() Cursor {
	return s.cursor
}

// Next returns one batch's examples and advances the sampler. The first value
// is always the opening; the remaining values follow the shuffled source
// sequence, crossing epoch boundaries without dropping records.
func (s *TrainingSampler) Next() ([]Example, error) {
	examples := make([]Example, s.batchSize)
	examples[0] = s.opening
	for index := 1; index < len(examples); index++ {
		recordIndex := s.nextRecordIndex()
		example, err := s.source.Example(recordIndex)
		if err != nil {
			return nil, err
		}
		if example.Record.SolutionID == synthetic.OpeningSolutionID {
			return nil, fmt.Errorf("internal error: shuffled source returned opening record %d", recordIndex)
		}
		examples[index] = example
	}
	return examples, nil
}

// Peek returns the next 20 non-opening source record indices without changing
// the sampler. Checkpoints record this short sequence to audit exact resume.
func (s *TrainingSampler) Peek() []int {
	clone := *s
	clone.order = append([]int(nil), s.order...)
	indices := make([]int, 20)
	for index := range indices {
		indices[index] = clone.nextRecordIndex()
	}
	return indices
}

func (s *TrainingSampler) nextRecordIndex() int {
	s.normalizeCursor()
	recordIndex := s.order[s.cursor.Offset]
	s.cursor.Offset++
	s.normalizeCursor()
	return recordIndex
}

func (s *TrainingSampler) normalizeCursor() {
	for s.cursor.Offset == len(s.order) {
		s.cursor.Epoch++
		s.cursor.Offset = 0
		s.order = s.orderForEpoch(s.cursor.Epoch)
	}
}

func (s *TrainingSampler) orderForEpoch(epoch int64) []int {
	shuffled := s.source.IndexOrder(s.seed + epoch)
	order := make([]int, 0, len(shuffled))
	for _, recordIndex := range shuffled {
		if s.source.records[recordIndex].SolutionID != synthetic.OpeningSolutionID {
			order = append(order, recordIndex)
		}
	}
	return order
}

// MaterializeBatch copies expanded examples into the fixed tensors consumed
// by the policy. It returns independent backing storage because Trainer may
// donate and finalize every tensor in the returned batch.
func MaterializeBatch(examples []Example) (train.Batch, error) {
	if len(examples) == 0 {
		return train.Batch{}, errors.New("cannot materialize an empty batch")
	}
	batchSize := len(examples)
	candidateMasks := make([]float32, batchSize*vocabulary.NumSolutions)
	candidateStats := make([]float32, batchSize*modelstate.CandidateStatsSize)
	turns := make([]int32, batchSize)
	remainingMasks := make([]float32, batchSize*vocabulary.NumActions)
	availableMasks := make([]float32, batchSize*vocabulary.NumActions)
	teacherTop1 := make([]int32, batchSize)
	teacherTop16 := make([]int32, batchSize*16)
	for index, example := range examples {
		if len(example.CandidateMask) != vocabulary.NumSolutions ||
			len(example.CandidateStats) != modelstate.CandidateStatsSize ||
			len(example.RemainingActionMask) != vocabulary.NumActions ||
			len(example.AvailableActionMask) != vocabulary.NumActions {
			return train.Batch{}, fmt.Errorf("example %d has invalid fixed input dimensions", index)
		}
		copy(candidateMasks[index*vocabulary.NumSolutions:], example.CandidateMask)
		copy(candidateStats[index*modelstate.CandidateStatsSize:], example.CandidateStats)
		turns[index] = example.Turn
		copy(remainingMasks[index*vocabulary.NumActions:], example.RemainingActionMask)
		copy(availableMasks[index*vocabulary.NumActions:], example.AvailableActionMask)
		teacherTop1[index] = example.TeacherTopAction
		copy(teacherTop16[index*16:], example.TeacherTopActions[:])
	}
	return train.Batch{
		Inputs: []*tensors.Tensor{
			tensors.FromFlatDataAndDimensions(candidateMasks, batchSize, vocabulary.NumSolutions),
			tensors.FromFlatDataAndDimensions(candidateStats, batchSize, modelstate.CandidateStatsSize),
			tensors.FromFlatDataAndDimensions(turns, batchSize),
			tensors.FromFlatDataAndDimensions(remainingMasks, batchSize, vocabulary.NumActions),
			tensors.FromFlatDataAndDimensions(availableMasks, batchSize, vocabulary.NumActions),
		},
		Labels: []*tensors.Tensor{
			tensors.FromFlatDataAndDimensions(teacherTop1, batchSize, 1),
			tensors.FromFlatDataAndDimensions(teacherTop16, batchSize, 16),
		},
	}, nil
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

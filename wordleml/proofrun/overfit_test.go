package proofrun

import (
	"path/filepath"
	"testing"

	"github.com/sam-bee/wordle-ml_machine-learning/imitationdata"
	"github.com/sam-bee/wordle-ml_machine-learning/vocabulary"
)

func TestSelectOverfitBatchUsesUniqueDiverseStatesAndOpening(t *testing.T) {
	dataDir := filepath.Join("..", "..", "data")
	vocabulary, err := vocabulary.Load(dataDir)
	if err != nil {
		t.Fatalf("load vocabulary: %v", err)
	}
	data, err := imitationdata.Load(vocabulary, filepath.Join(dataDir, "imitation"), imitationdata.Train)
	if err != nil {
		t.Fatalf("load train data: %v", err)
	}
	batch, err := SelectOverfitBatch(data, Seed)
	if err != nil {
		t.Fatalf("SelectOverfitBatch: %v", err)
	}
	if len(batch) != 128 || batch[0].Turn != 0 {
		t.Fatalf("overfit batch length/opening = %d/%d", len(batch), batch[0].Turn)
	}
	seen := make(map[[32]byte]struct{}, len(batch))
	turns, buckets := make(map[int32]struct{}), make(map[int]struct{})
	for _, example := range batch {
		key := encodedStateKey(example)
		if _, found := seen[key]; found {
			t.Fatalf("duplicate encoded state at record %d", example.RecordIndex)
		}
		seen[key] = struct{}{}
		turns[example.Turn] = struct{}{}
		buckets[shortlistBucket(example)] = struct{}{}
	}
	if len(turns) < 2 || len(buckets) < 2 {
		t.Fatalf("turns=%d shortlist buckets=%d, want multiple", len(turns), len(buckets))
	}
}

func TestEncodedStateKeyUsesOnlySemanticModelInputs(t *testing.T) {
	dataDir := filepath.Join("..", "..", "data")
	vocabulary, err := vocabulary.Load(dataDir)
	if err != nil {
		t.Fatalf("load vocabulary: %v", err)
	}
	data, err := imitationdata.Load(vocabulary, filepath.Join(dataDir, "imitation"), imitationdata.Train)
	if err != nil {
		t.Fatalf("load train data: %v", err)
	}

	var example imitationdata.Example
	found := false
	for index := 0; index < data.Len(); index++ {
		candidate, err := data.Example(index)
		if err != nil {
			t.Fatal(err)
		}
		if candidate.Record.TurnDepth >= 2 && int(candidate.Record.TurnDepth) < len(candidate.Record.History) {
			example, found = candidate, true
			break
		}
	}
	if !found {
		t.Fatal("training fixture has no record with used and unused history slots")
	}

	want := encodedStateKey(example)
	modified := example
	// Feedback has already determined CandidateBits, so it is not an additional
	// model input. Likewise, history slots after TurnDepth are not inputs.
	modified.Record.History[0].FeedbackCode ^= 0xff
	unused := int(modified.Record.TurnDepth)
	modified.Record.History[unused].GuessID ^= 0xffff
	modified.Record.History[unused].FeedbackCode ^= 0xff
	if got := encodedStateKey(modified); got != want {
		t.Fatal("feedback or unused history changed the encoded-state key")
	}

	// Availability is a set, so reordering already-guessed actions is also
	// semantically identical.
	modified = example
	modified.Record.History[0].GuessID, modified.Record.History[1].GuessID = modified.Record.History[1].GuessID, modified.Record.History[0].GuessID
	if got := encodedStateKey(modified); got != want {
		t.Fatal("reordered guessed actions changed the encoded-state key")
	}

	modified = example
	used := make(map[uint16]struct{}, modified.Record.TurnDepth)
	for index := 0; index < int(modified.Record.TurnDepth); index++ {
		used[modified.Record.History[index].GuessID] = struct{}{}
	}
	replacement := uint16(0)
	for {
		if _, found := used[replacement]; !found {
			break
		}
		replacement++
	}
	modified.Record.History[0].GuessID = replacement
	if got := encodedStateKey(modified); got == want {
		t.Fatal("a used guessed action did not change the encoded-state key")
	}
}

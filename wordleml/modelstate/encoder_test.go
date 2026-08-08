package modelstate

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/sam-bee/wordle-ml_machine-learning/vocabulary"
)

func TestEncodeHandCheckableABACK(t *testing.T) {
	encoder := testEncoder(t)
	bits := bitset(0)
	inputs, err := encoder.Encode(bits, 3)
	if err != nil {
		t.Fatal(err)
	}
	if inputs.Turn != 3 {
		t.Fatalf("turn = %d, want 3", inputs.Turn)
	}
	if inputs.CandidateMask[0] != 1 || countOnes(inputs.CandidateMask) != 1 {
		t.Fatalf("candidate mask = %v, want only solution 0", inputs.CandidateMask[:2])
	}
	if inputs.RemainingActionMask[1] != 1 || countOnes(inputs.RemainingActionMask) != 1 {
		t.Fatalf("remaining action mask did not contain only ABACK")
	}

	// ABACK: A B A C K. A occurs twice; B, C, and K occur once.
	want := map[int]float32{
		0:   1, // position 0: A
		27:  1, // position 1: B
		52:  1, // position 2: A
		80:  1, // position 3: C
		114: 1, // position 4: K
		130: 1, // A appears at least once
		131: 1, // A appears at least twice
		133: 1, // B appears at least once
		136: 1, // C appears at least once
		160: 1, // K appears at least once
	}
	for i, got := range inputs.CandidateStats {
		if expected := want[i]; got != expected {
			t.Fatalf("stat[%d] = %v, want %v", i, got, expected)
		}
	}
}

func TestEncodeNormalizesTwoCandidates(t *testing.T) {
	encoder := testEncoder(t)
	inputs, err := encoder.Encode(bitset(0, 1), 0) // ABACK, ABASE
	if err != nil {
		t.Fatal(err)
	}
	if got := inputs.CandidateStats[80]; got != 0.5 { // position 3: C in ABACK only
		t.Fatalf("position-3 C fraction = %v, want 0.5", got)
	}
	if got := inputs.CandidateStats[96]; got != 0.5 { // position 3: S in ABASE only
		t.Fatalf("position-3 S fraction = %v, want 0.5", got)
	}
	if got := inputs.CandidateStats[131]; got != 1 { // both words contain two As
		t.Fatalf("two-or-more A fraction = %v, want 1", got)
	}
	wantLog := float32(math.Log(2) / math.Log(vocabulary.NumSolutions))
	if got := inputs.CandidateStats[208]; math.Abs(float64(got-wantLog)) > 1e-7 {
		t.Fatalf("log candidate count = %v, want %v", got, wantLog)
	}
}

func TestEncodeAcceptsEveryCandidateBit(t *testing.T) {
	encoder := testEncoder(t)
	bits := make([]byte, CandidateBitsetBytes)
	for solutionID := 0; solutionID < vocabulary.NumSolutions; solutionID++ {
		bits[solutionID/8] |= 1 << uint(solutionID%8)
	}

	inputs, err := encoder.Encode(bits, 5)
	if err != nil {
		t.Fatal(err)
	}
	if inputs.CandidateMask[vocabulary.NumSolutions-1] != 1 {
		t.Fatal("last valid candidate bit was not encoded")
	}
	if got := inputs.CandidateStats[CandidateStatsSize-1]; got != 1 {
		t.Fatalf("full-set log candidate count = %v, want 1", got)
	}
	for position := 0; position < 5; position++ {
		var total float32
		for letter := 0; letter < 26; letter++ {
			total += inputs.CandidateStats[position*26+letter]
		}
		if math.Abs(float64(total-1)) > 1e-6 {
			t.Fatalf("position %d frequencies sum to %v, want 1", position, total)
		}
	}
}

func TestEncodeRejectsInvalidStates(t *testing.T) {
	encoder := testEncoder(t)
	padding := bitset(0)
	padding[len(padding)-1] |= 1 << 5
	for _, test := range []struct {
		name string
		bits []byte
		turn int
	}{
		{name: "empty", bits: make([]byte, CandidateBitsetBytes), turn: 0},
		{name: "short", bits: make([]byte, CandidateBitsetBytes-1), turn: 0},
		{name: "padding", bits: padding, turn: 0},
		{name: "turn", bits: bitset(0), turn: 6},
	} {
		t.Run(test.name, func(t *testing.T) {
			if _, err := encoder.Encode(test.bits, test.turn); err == nil {
				t.Fatal("Encode succeeded, want error")
			}
		})
	}
}

func testEncoder(t *testing.T) *Encoder {
	t.Helper()
	dataDir := os.Getenv("WORDLEML_DATA_DIR")
	if dataDir == "" {
		dataDir = filepath.Join("..", "..", "data")
	}
	v, err := vocabulary.Load(dataDir)
	if err != nil {
		t.Fatal(err)
	}
	encoder, err := NewEncoder(v)
	if err != nil {
		t.Fatal(err)
	}
	return encoder
}

func bitset(solutionIDs ...int) []byte {
	bits := make([]byte, CandidateBitsetBytes)
	for _, solutionID := range solutionIDs {
		bits[solutionID/8] |= 1 << uint(solutionID%8)
	}
	return bits
}

func countOnes(values []float32) int {
	count := 0
	for _, value := range values {
		if value == 1 {
			count++
		}
	}
	return count
}

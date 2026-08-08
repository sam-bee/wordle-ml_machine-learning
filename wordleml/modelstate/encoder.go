// Package modelstate converts a remaining-solution bitset into policy inputs.
package modelstate

import (
	"fmt"
	"math"

	"github.com/sam-bee/wordle-ml_machine-learning/vocabulary"
)

const (
	// CandidateBitsetBytes stores one LSB-first bit for each fixed solution ID.
	CandidateBitsetBytes = (vocabulary.NumSolutions + 7) / 8
	// CandidateStatsSize is the size of the policy's aggregate candidate features.
	CandidateStatsSize = 209
)

// Inputs are host values for the four inputs accepted by policy.Model.Forward.
type Inputs struct {
	CandidateMask  []float32
	CandidateStats []float32
	Turn           int32
	// RemainingActionMask marks candidate solutions in action-ID space. It is
	// the policy's learned candidate bonus input, not an action-legality mask.
	RemainingActionMask []float32
}

// Encoder uses one frozen vocabulary to encode every model state.
type Encoder struct {
	vocabulary *vocabulary.Vocabulary
}

// NewEncoder creates an encoder for vocabulary.
func NewEncoder(v *vocabulary.Vocabulary) (*Encoder, error) {
	if v == nil {
		return nil, fmt.Errorf("vocabulary must not be nil")
	}
	return &Encoder{vocabulary: v}, nil
}

// Encode converts an exactly 2,309-bit LSB-first candidate set and turn 0..5
// into FP32 masks/statistics and an int32 turn value for the policy model.
func (e *Encoder) Encode(candidateBits []byte, turn int) (Inputs, error) {
	if turn < 0 || turn > 5 {
		return Inputs{}, fmt.Errorf("turn must be from 0 through 5, got %d", turn)
	}
	if len(candidateBits) != CandidateBitsetBytes {
		return Inputs{}, fmt.Errorf("candidate bitset has %d bytes, want %d", len(candidateBits), CandidateBitsetBytes)
	}
	if padding := candidateBits[len(candidateBits)-1] >> (vocabulary.NumSolutions % 8); padding != 0 {
		return Inputs{}, fmt.Errorf("candidate bitset has non-zero padding bits: %08b", padding)
	}

	inputs := Inputs{
		CandidateMask:       make([]float32, vocabulary.NumSolutions),
		CandidateStats:      make([]float32, CandidateStatsSize),
		Turn:                int32(turn),
		RemainingActionMask: make([]float32, vocabulary.NumActions),
	}
	count := 0
	for solutionID := 0; solutionID < vocabulary.NumSolutions; solutionID++ {
		if candidateBits[solutionID/8]&(1<<uint(solutionID%8)) == 0 {
			continue
		}
		count++
		inputs.CandidateMask[solutionID] = 1
		actionID, _ := e.vocabulary.SolutionActionID(solutionID)
		inputs.RemainingActionMask[actionID] = 1
		e.addWordStatistics(inputs.CandidateStats, solutionID)
	}
	if count == 0 {
		return Inputs{}, fmt.Errorf("candidate bitset is empty")
	}

	denominator := float32(count)
	for i := 0; i < CandidateStatsSize-1; i++ {
		inputs.CandidateStats[i] /= denominator
	}
	inputs.CandidateStats[CandidateStatsSize-1] = float32(math.Log(float64(count)) / math.Log(float64(vocabulary.NumSolutions)))
	return inputs, nil
}

func (e *Encoder) addWordStatistics(stats []float32, solutionID int) {
	word, _ := e.vocabulary.SolutionWord(solutionID)
	var copies [26]int
	for position := range word {
		letter := int(word[position] - 'A')
		stats[position*26+letter]++
		copies[letter]++
	}
	for letter, count := range copies {
		for threshold := 1; threshold <= 3 && count >= threshold; threshold++ {
			stats[130+letter*3+threshold-1]++
		}
	}
}

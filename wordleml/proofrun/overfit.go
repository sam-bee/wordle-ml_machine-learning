package proofrun

import (
	"crypto/sha256"
	"fmt"

	"github.com/sam-bee/wordle-ml_machine-learning/imitationdata"
)

// SelectOverfitBatch constructs the deterministic 128-example proof batch.
// It scans all source states, so an equal state with a contradictory top-1
// teacher action cannot be hidden by the selection order.
func SelectOverfitBatch(data *imitationdata.Data, seed int64) ([]imitationdata.Example, error) {
	if data == nil {
		return nil, fmt.Errorf("overfit data must not be nil")
	}
	if data.Split() != imitationdata.Train {
		return nil, fmt.Errorf("overfit data split is %q, want train", data.Split())
	}
	opening, found, err := data.FindOpening()
	if err != nil {
		return nil, err
	}
	if !found {
		return nil, fmt.Errorf("training data has no opening example")
	}
	selected := make([]imitationdata.Example, 0, 128)
	selected = append(selected, opening)
	seen := make(map[[sha256.Size]byte]int32, data.Len())
	seen[imitationdata.ModelStateKey(opening)] = opening.TeacherTopAction
	for _, index := range data.IndexOrder(seed) {
		key, currentLabel, err := data.ModelStateIdentity(index)
		if err != nil {
			return nil, err
		}
		if priorLabel, alreadySeen := seen[key]; alreadySeen {
			if priorLabel != currentLabel {
				return nil, fmt.Errorf("identical encoded state at record %d has top-1 teacher action %d, previously %d", index, currentLabel, priorLabel)
			}
			continue
		}
		seen[key] = currentLabel
		if len(selected) < 128 {
			example, err := data.Example(index)
			if err != nil {
				return nil, err
			}
			selected = append(selected, example)
		}
	}
	if len(selected) != 128 {
		return nil, fmt.Errorf("found only %d unique encoded states, want 128", len(selected))
	}
	turns := make(map[int32]struct{})
	buckets := make(map[int]struct{})
	for _, example := range selected {
		turns[example.Turn] = struct{}{}
		buckets[shortlistBucket(example)] = struct{}{}
	}
	if len(turns) < 2 {
		return nil, fmt.Errorf("overfit batch contains only %d turn group", len(turns))
	}
	if len(buckets) < 2 {
		return nil, fmt.Errorf("overfit batch contains only %d shortlist bucket", len(buckets))
	}
	return selected, nil
}

func shortlistBucket(example imitationdata.Example) int {
	count := 0
	for _, value := range example.CandidateMask {
		if value != 0 {
			count++
		}
	}
	switch {
	case count == 1:
		return 0
	case count <= 5:
		return 1
	case count <= 20:
		return 2
	case count <= 100:
		return 3
	default:
		return 4
	}
}

func encodedStateKey(example imitationdata.Example) [sha256.Size]byte {
	return imitationdata.ModelStateKey(example)
}

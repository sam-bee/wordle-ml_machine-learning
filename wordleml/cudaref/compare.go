package cudaref

import (
	"fmt"
	"math"
	"sort"

	"github.com/sam-bee/wordle-ml_machine-learning/vocabulary"
)

// CompareLogits computes the measurements exported by both the portable-Go
// and CUDA parity commands.  It deliberately uses the same ascending-index,
// strict-greater tie rule as gameeval's action selection for raw top actions.
func CompareLogits(vectors []Vector, actual func(Vector) ([]float32, error)) (Comparison, error) {
	if len(vectors) == 0 {
		return Comparison{}, fmt.Errorf("compare at least one vector")
	}
	if actual == nil {
		return Comparison{}, fmt.Errorf("logit evaluator is required")
	}
	comparison := Comparison{ActionComparisons: len(vectors)}
	var total float64
	for _, vector := range vectors {
		if err := ValidateVector(vector); err != nil {
			return Comparison{}, err
		}
		got, err := actual(vector)
		if err != nil {
			return Comparison{}, fmt.Errorf("evaluate vector %q: %w", vector.ID, err)
		}
		if len(got) != vocabulary.NumActions {
			return Comparison{}, fmt.Errorf("vector %q produced %d logits, want %d", vector.ID, len(got), vocabulary.NumActions)
		}
		for action, value := range got {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return Comparison{}, fmt.Errorf("vector %q action %d produced non-finite logit", vector.ID, action)
			}
			delta := math.Abs(float64(value) - float64(vector.RawLogits[action]))
			total += delta
			comparison.Compared++
			if delta > comparison.MaximumAbsolute {
				comparison.MaximumAbsolute = delta
				comparison.WorstVectorID = vector.ID
				comparison.WorstActionID = action
			}
		}
		if rawTopAction(got) == vector.RawTopActionID {
			comparison.Top1Agreement++
		}
		if sameTopK(got, vector.RawLogits, 5) {
			comparison.Top5SetAgreement++
		}
	}
	comparison.MeanAbsolute = total / float64(comparison.Compared)
	return comparison, nil
}

// RawTopAction returns the first (lowest ID) action whose finite score is
// maximal.  It is public so the exporter can record a raw preference without
// embedding selection logic in CUDA.
func RawTopAction(logits []float32) (int, error) {
	if len(logits) != vocabulary.NumActions {
		return 0, fmt.Errorf("logits have %d values, want %d", len(logits), vocabulary.NumActions)
	}
	for action, value := range logits {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return 0, fmt.Errorf("logit %d is non-finite", action)
		}
	}
	return rawTopAction(logits), nil
}

// SelectAvailableAction applies the Go-side availability mask to raw logits.
// It intentionally uses no numeric masking trick: the caller retains both the
// raw model preference and the accepted action, with equal scores resolving to
// the lower action ID exactly as the authoritative game evaluator does.
func SelectAvailableAction(logits, available []float32) (rawTop, selected int, err error) {
	if len(available) != vocabulary.NumActions {
		return 0, 0, fmt.Errorf("availability mask has %d values, want %d", len(available), vocabulary.NumActions)
	}
	rawTop, err = RawTopAction(logits)
	if err != nil {
		return 0, 0, err
	}
	selected = -1
	for action, score := range logits {
		if available[action] != 0 && (selected < 0 || score > logits[selected]) {
			selected = action
		}
	}
	if selected < 0 {
		return 0, 0, fmt.Errorf("no available action")
	}
	return rawTop, selected, nil
}

// TopTwoMargin returns the finite raw top-one minus top-two score using the
// deterministic lower-ID tie order.  It records how susceptible an action is
// to a genuinely tiny numeric implementation difference.
func TopTwoMargin(logits []float32) (float32, error) {
	if len(logits) != vocabulary.NumActions {
		return 0, fmt.Errorf("logits have %d values, want %d", len(logits), vocabulary.NumActions)
	}
	first, second := -1, -1
	for action, value := range logits {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return 0, fmt.Errorf("logit %d is non-finite", action)
		}
		if first < 0 || value > logits[first] {
			second, first = first, action
		} else if second < 0 || value > logits[second] {
			second = action
		}
	}
	if second < 0 {
		return 0, fmt.Errorf("need at least two logits")
	}
	return logits[first] - logits[second], nil
}

func rawTopAction(logits []float32) int {
	top := 0
	for action := 1; action < len(logits); action++ {
		if logits[action] > logits[top] {
			top = action
		}
	}
	return top
}

func sameTopK(left, right []float32, k int) bool {
	if len(left) != len(right) || k <= 0 || k > len(left) {
		return false
	}
	leftIDs := topK(left, k)
	rightIDs := topK(right, k)
	sort.Ints(leftIDs)
	sort.Ints(rightIDs)
	for index := range leftIDs {
		if leftIDs[index] != rightIDs[index] {
			return false
		}
	}
	return true
}

func topK(values []float32, k int) []int {
	ids := make([]int, len(values))
	for index := range ids {
		ids[index] = index
	}
	sort.SliceStable(ids, func(left, right int) bool {
		if values[ids[left]] == values[ids[right]] {
			return ids[left] < ids[right]
		}
		return values[ids[left]] > values[ids[right]]
	})
	return ids[:k]
}

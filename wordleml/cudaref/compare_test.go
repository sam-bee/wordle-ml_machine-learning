package cudaref

import (
	"testing"

	"github.com/sam-bee/wordle-ml_machine-learning/vocabulary"
)

func TestRawTopActionUsesLowerActionIDForTies(t *testing.T) {
	logits := make([]float32, vocabulary.NumActions)
	logits[11] = 4
	logits[17] = 4
	got, err := RawTopAction(logits)
	if err != nil {
		t.Fatal(err)
	}
	if got != 11 {
		t.Fatalf("RawTopAction=%d, want 11", got)
	}
	margin, err := TopTwoMargin(logits)
	if err != nil {
		t.Fatal(err)
	}
	if margin != 0 {
		t.Fatalf("TopTwoMargin=%g, want 0", margin)
	}
}

func TestSelectAvailableActionSuppressesRawChoiceWithoutChangingTieRule(t *testing.T) {
	logits := make([]float32, vocabulary.NumActions)
	logits[11] = 4
	logits[17] = 4
	available := make([]float32, vocabulary.NumActions)
	available[17] = 1
	raw, selected, err := SelectAvailableAction(logits, available)
	if err != nil {
		t.Fatal(err)
	}
	if raw != 11 || selected != 17 {
		t.Fatalf("raw/selected=%d/%d, want 11/17", raw, selected)
	}
}

func TestCompareLogitsReportsActionAgreementAndError(t *testing.T) {
	vector := testVector("opening", 0)
	vector.RawLogits[100] = 10
	vector.RawTopActionID = 100
	vector.TopTwoMargin = 2.5
	comparison, err := CompareLogits([]Vector{vector}, func(vector Vector) ([]float32, error) {
		got := append([]float32(nil), vector.RawLogits...)
		got[100] += .25
		return got, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if comparison.MaximumAbsolute != .25 || comparison.WorstVectorID != "opening" || comparison.WorstActionID != 100 {
		t.Fatalf("comparison=%+v", comparison)
	}
	if comparison.Top1Agreement != 1 || comparison.Top5SetAgreement != 1 {
		t.Fatalf("agreement=%+v", comparison)
	}
}

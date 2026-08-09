package proofeval

import (
	"strings"
	"testing"
)

func TestSelectionHashCommitsToOrderedTeacherTop1Vector(t *testing.T) {
	first := selectionHash([]bool{true, false, true})
	if len(first) != 64 {
		t.Fatalf("selection hash length = %d", len(first))
	}
	if first != selectionHash([]bool{true, false, true}) {
		t.Fatal("identical selection vectors have different hashes")
	}
	if first == selectionHash([]bool{false, true, true}) || first == selectionHash([]bool{true, false}) {
		t.Fatal("selection hash did not commit to values and order")
	}
	if strings.Trim(first, "0") == "" {
		t.Fatal("selection hash is empty")
	}
}

func TestExactTop1CountUsesFixedValidationPopulation(t *testing.T) {
	if got := exactTop1Count(.5100, 2500); got != 1275 {
		t.Fatalf("top-1 count = %d, want 1275", got)
	}
}

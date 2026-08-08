package vocabulary

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadFrozenVocabularyAndSplit(t *testing.T) {
	v, err := Load(testDataDir(t))
	if err != nil {
		t.Fatal(err)
	}
	if got := len(v.Actions()); got != NumActions {
		t.Fatalf("action count = %d, want %d", got, NumActions)
	}
	if got := len(v.Solutions()); got != NumSolutions {
		t.Fatalf("solution count = %d, want %d", got, NumSolutions)
	}
	if got := len(v.Training()); got != NumTrainingSolutions {
		t.Fatalf("training count = %d, want %d", got, NumTrainingSolutions)
	}
	if got := len(v.Validation()); got != NumValidationSolutions {
		t.Fatalf("validation count = %d, want %d", got, NumValidationSolutions)
	}
	if got := len(v.Test()); got != NumTestSolutions {
		t.Fatalf("test count = %d, want %d", got, NumTestSolutions)
	}

	wantHashes := Hashes{
		Actions:    "992239bab3de16bf51dcca2bc10efe5f81c6d92e01f00172f96574518807f4eb",
		Solutions:  "66d2f19d4543833517c79aafa44a6582251c21618157bffcd9e453daf405d4ff",
		Training:   "70184dfa5c291a73c8576f8ea6bbe041482890b63b68e965e9c94069577f7b78",
		Validation: "3cadd757ce9e6c57676358a8a13de4ef3d12fc2af7ba3033278c9926b867c019",
		Test:       "978a25608a96370b3e26cc8621e9f2cc83ad2d581d07b4b23546b0b4ccdec130",
	}
	if got := v.Hashes(); got != wantHashes {
		t.Fatalf("hashes = %#v, want %#v", got, wantHashes)
	}

	seen := make(map[string]string, NumSolutions)
	for splitName, words := range map[string][]string{
		"training": v.Training(), "validation": v.Validation(), "test": v.Test(),
	} {
		for _, word := range words {
			if earlier, duplicate := seen[word]; duplicate {
				t.Fatalf("%q appears in both %s and %s", word, earlier, splitName)
			}
			seen[word] = splitName
		}
	}
	if len(seen) != NumSolutions {
		t.Fatalf("split covers %d solutions, want %d", len(seen), NumSolutions)
	}
}

func TestSolutionActionMappingPreservesIDs(t *testing.T) {
	v, err := Load(testDataDir(t))
	if err != nil {
		t.Fatal(err)
	}
	if word, ok := v.SolutionWord(0); !ok || word != "ABACK" {
		t.Fatalf("solution ID 0 = %q, %t, want ABACK, true", word, ok)
	}
	if actionID, ok := v.SolutionActionID(0); !ok || actionID != 1 {
		t.Fatalf("action ID for solution ID 0 = %d, %t, want 1, true", actionID, ok)
	}
	for solutionID := 0; solutionID < NumSolutions; solutionID++ {
		word, _ := v.SolutionWord(solutionID)
		actionID, ok := v.SolutionActionID(solutionID)
		if !ok {
			t.Fatalf("solution ID %d has no action mapping", solutionID)
		}
		mappedWord, _ := v.ActionWord(actionID)
		if mappedWord != word {
			t.Fatalf("solution ID %d word %q maps to action %d word %q", solutionID, word, actionID, mappedWord)
		}
	}
}

func testDataDir(t *testing.T) string {
	t.Helper()
	if dataDir := os.Getenv("WORDLEML_DATA_DIR"); dataDir != "" {
		return dataDir
	}
	return filepath.Join("..", "..", "data")
}

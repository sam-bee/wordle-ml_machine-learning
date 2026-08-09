// Package vocabulary loads the frozen Wordle words and solution split.
package vocabulary

import (
	"crypto/sha256"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

const (
	// NumActions is the fixed size of the policy action vocabulary.
	NumActions = 4739
	// NumSolutions is the fixed number of possible Wordle solutions.
	NumSolutions = 2309
	// NumTrainingSolutions is the number of solutions used for training.
	NumTrainingSolutions = 2109
	// NumValidationSolutions is the number of validation solutions.
	NumValidationSolutions = 100
	// NumTestSolutions is the number of final-test solutions.
	NumTestSolutions = 100
)

const (
	actionSpaceFilename  = "wordlist-action-space-4739.csv"
	allSolutionsFilename = "wordlist-valid-solutions-all-2309.csv"
	trainingFilename     = "wordlist-valid-solutions-train-2109.csv"
	validationFilename   = "wordlist-valid-solutions-validation-100.csv"
	testFilename         = "wordlist-valid-solutions-test-100.csv"
)

// Hashes identifies the normalized, ordered contents of each frozen word list.
// Each hash covers uppercase words with exactly one trailing newline per word.
type Hashes struct {
	Actions    string
	Solutions  string
	Training   string
	Validation string
	Test       string
}

// Vocabulary preserves the canonical action and solution IDs from the word lists.
type Vocabulary struct {
	actions          []string
	solutions        []string
	training         []string
	validation       []string
	test             []string
	actionIDByWord   map[string]int
	solutionIDByWord map[string]int
	solutionActionID []int
	hashes           Hashes
}

// Load reads the five fixed vocabulary files from dataDir and validates their
// dimensions, membership, and complete train/validation/test solution split.
func Load(dataDir string) (*Vocabulary, error) {
	return load(dataDir, true)
}

// LoadWithoutFinalTest reads only the action, all-solution, training, and
// validation word lists. It is the sealed-test loader for predeclared
// experiments that must not open the final-test word list even for hashing or
// partition validation.
func LoadWithoutFinalTest(dataDir string) (*Vocabulary, error) {
	return load(dataDir, false)
}

func load(dataDir string, includeFinalTest bool) (*Vocabulary, error) {
	actions, actionHash, err := readWords(filepath.Join(dataDir, actionSpaceFilename), NumActions)
	if err != nil {
		return nil, err
	}
	solutions, solutionHash, err := readWords(filepath.Join(dataDir, allSolutionsFilename), NumSolutions)
	if err != nil {
		return nil, err
	}
	training, trainingHash, err := readWords(filepath.Join(dataDir, trainingFilename), NumTrainingSolutions)
	if err != nil {
		return nil, err
	}
	validation, validationHash, err := readWords(filepath.Join(dataDir, validationFilename), NumValidationSolutions)
	if err != nil {
		return nil, err
	}
	var test []string
	var testHash string
	if includeFinalTest {
		test, testHash, err = readWords(filepath.Join(dataDir, testFilename), NumTestSolutions)
		if err != nil {
			return nil, err
		}
	}

	actionIDByWord, err := indexWords("action space", actions)
	if err != nil {
		return nil, err
	}
	solutionIDByWord, err := indexWords("all solutions", solutions)
	if err != nil {
		return nil, err
	}
	solutionActionID := make([]int, len(solutions))
	for solutionID, word := range solutions {
		actionID, ok := actionIDByWord[word]
		if !ok {
			return nil, fmt.Errorf("solution %q at ID %d is missing from the action space", word, solutionID)
		}
		solutionActionID[solutionID] = actionID
	}

	if err := validateSplit(solutionIDByWord, training, validation, test, includeFinalTest); err != nil {
		return nil, err
	}

	return &Vocabulary{
		actions:          actions,
		solutions:        solutions,
		training:         training,
		validation:       validation,
		test:             test,
		actionIDByWord:   actionIDByWord,
		solutionIDByWord: solutionIDByWord,
		solutionActionID: solutionActionID,
		hashes: Hashes{
			Actions:    actionHash,
			Solutions:  solutionHash,
			Training:   trainingHash,
			Validation: validationHash,
			Test:       testHash,
		},
	}, nil
}

// Actions returns the action words in canonical action-ID order.
func (v *Vocabulary) Actions() []string { return append([]string(nil), v.actions...) }

// Solutions returns the solution words in canonical solution-ID order.
func (v *Vocabulary) Solutions() []string { return append([]string(nil), v.solutions...) }

// Training returns the training solutions in their frozen file order.
func (v *Vocabulary) Training() []string { return append([]string(nil), v.training...) }

// Validation returns the validation solutions in their frozen file order.
func (v *Vocabulary) Validation() []string { return append([]string(nil), v.validation...) }

// Test returns the final-test solutions in their frozen file order.
func (v *Vocabulary) Test() []string { return append([]string(nil), v.test...) }

// ActionWord returns the word at actionID.
func (v *Vocabulary) ActionWord(actionID int) (string, bool) {
	if actionID < 0 || actionID >= len(v.actions) {
		return "", false
	}
	return v.actions[actionID], true
}

// SolutionWord returns the word at solutionID.
func (v *Vocabulary) SolutionWord(solutionID int) (string, bool) {
	if solutionID < 0 || solutionID >= len(v.solutions) {
		return "", false
	}
	return v.solutions[solutionID], true
}

// ActionID returns the canonical action ID for word.
func (v *Vocabulary) ActionID(word string) (int, bool) {
	id, ok := v.actionIDByWord[word]
	return id, ok
}

// SolutionID returns the canonical solution ID for word.
func (v *Vocabulary) SolutionID(word string) (int, bool) {
	id, ok := v.solutionIDByWord[word]
	return id, ok
}

// SolutionActionID returns the action ID corresponding to solutionID.
func (v *Vocabulary) SolutionActionID(solutionID int) (int, bool) {
	if solutionID < 0 || solutionID >= len(v.solutionActionID) {
		return 0, false
	}
	return v.solutionActionID[solutionID], true
}

// SolutionActionIDs returns action IDs indexed by canonical solution ID.
func (v *Vocabulary) SolutionActionIDs() []int { return append([]int(nil), v.solutionActionID...) }

// Hashes returns canonical SHA-256 hashes for the loaded frozen files.
func (v *Vocabulary) Hashes() Hashes { return v.hashes }

func readWords(path string, expectedCount int) ([]string, string, error) {
	contents, err := os.ReadFile(path)
	if err != nil {
		return nil, "", fmt.Errorf("read %s: %w", path, err)
	}
	lines := strings.Split(strings.ReplaceAll(string(contents), "\r\n", "\n"), "\n")
	if len(lines) > 0 && lines[len(lines)-1] == "" {
		lines = lines[:len(lines)-1]
	}
	if len(lines) != expectedCount {
		return nil, "", fmt.Errorf("%s has %d words, want %d", path, len(lines), expectedCount)
	}

	hasher := sha256.New()
	for lineNumber, word := range lines {
		word = strings.ToUpper(strings.TrimSpace(word))
		if !isWord(word) {
			return nil, "", fmt.Errorf("%s line %d: want one five-letter ASCII word, got %q", path, lineNumber+1, lines[lineNumber])
		}
		lines[lineNumber] = word
		_, _ = hasher.Write([]byte(word))
		_, _ = hasher.Write([]byte{'\n'})
	}
	return lines, fmt.Sprintf("%x", hasher.Sum(nil)), nil
}

func isWord(word string) bool {
	if len(word) != 5 {
		return false
	}
	for i := range word {
		if word[i] < 'A' || word[i] > 'Z' {
			return false
		}
	}
	return true
}

func indexWords(name string, words []string) (map[string]int, error) {
	index := make(map[string]int, len(words))
	for id, word := range words {
		if priorID, duplicate := index[word]; duplicate {
			return nil, fmt.Errorf("%s repeats %q at IDs %d and %d", name, word, priorID, id)
		}
		index[word] = id
	}
	return index, nil
}

func validateSplit(solutionIDByWord map[string]int, training, validation, test []string, requireComplete bool) error {
	seen := make(map[string]string, NumSolutions)
	for splitName, words := range map[string][]string{
		"training": training, "validation": validation, "test": test,
	} {
		for _, word := range words {
			if _, found := solutionIDByWord[word]; !found {
				return fmt.Errorf("%s split contains non-solution word %q", splitName, word)
			}
			if previousSplit, duplicate := seen[word]; duplicate {
				return fmt.Errorf("solution %q occurs in both %s and %s splits", word, previousSplit, splitName)
			}
			seen[word] = splitName
		}
	}
	want := NumTrainingSolutions + NumValidationSolutions
	if requireComplete {
		want = NumSolutions
	}
	if len(seen) != want {
		return fmt.Errorf("loaded solution splits contain %d distinct solutions, want %d", len(seen), want)
	}
	return nil
}

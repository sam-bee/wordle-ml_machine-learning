// Command split-solutions creates reproducible training, validation, and test
// solution vocabularies from the complete Wordle solution list.
package main

import (
	"bytes"
	"crypto/sha256"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

const (
	defaultInput          = "../data/wordlist-valid-solutions-all-2309.csv"
	defaultOutputDir      = "../data"
	defaultSeed           = "wordle-ml-solution-split-v1"
	defaultValidationSize = 100
	defaultTestSize       = 100
)

type solutionSplits struct {
	training   []string
	validation []string
	test       []string
}

type rankedWord struct {
	word string
	rank [sha256.Size]byte
}

func main() {
	if err := run(os.Args[1:], os.Stdout); err != nil {
		fmt.Fprintf(os.Stderr, "split solutions: %v\n", err)
		os.Exit(1)
	}
}

func run(args []string, stdout io.Writer) error {
	flags := flag.NewFlagSet("split-solutions", flag.ContinueOnError)
	flags.SetOutput(io.Discard)

	input := flags.String("input", defaultInput, "complete solution wordlist")
	outputDir := flags.String("output-dir", defaultOutputDir, "directory for split wordlists")
	seed := flags.String("seed", defaultSeed, "stable split seed")
	validationSize := flags.Int("validation-size", defaultValidationSize, "number of validation solutions")
	testSize := flags.Int("test-size", defaultTestSize, "number of final test solutions")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if flags.NArg() != 0 {
		return fmt.Errorf("unexpected positional arguments: %s", strings.Join(flags.Args(), " "))
	}

	contents, err := os.ReadFile(*input)
	if err != nil {
		return fmt.Errorf("read %s: %w", *input, err)
	}
	words, err := parseWordlist(contents)
	if err != nil {
		return fmt.Errorf("parse %s: %w", *input, err)
	}
	splits, err := splitSolutions(words, *validationSize, *testSize, *seed)
	if err != nil {
		return err
	}

	outputs := []struct {
		name  string
		words []string
	}{
		{name: fmt.Sprintf("wordlist-valid-solutions-train-%d.csv", len(splits.training)), words: splits.training},
		{name: fmt.Sprintf("wordlist-valid-solutions-validation-%d.csv", len(splits.validation)), words: splits.validation},
		{name: fmt.Sprintf("wordlist-valid-solutions-test-%d.csv", len(splits.test)), words: splits.test},
	}
	for _, output := range outputs {
		path := filepath.Join(*outputDir, output.name)
		contents := strings.Join(output.words, "\n") + "\n"
		if err := os.WriteFile(path, []byte(contents), 0o644); err != nil {
			return fmt.Errorf("write %s: %w", path, err)
		}
	}

	fmt.Fprintf(
		stdout,
		"Split %d solutions with seed %q: train=%d validation=%d test=%d\n",
		len(words), *seed, len(splits.training), len(splits.validation), len(splits.test),
	)
	return nil
}

func parseWordlist(contents []byte) ([]string, error) {
	text := strings.TrimSuffix(string(contents), "\n")
	if text == "" {
		return nil, errors.New("wordlist is empty")
	}

	lines := strings.Split(text, "\n")
	words := make([]string, len(lines))
	for i, line := range lines {
		word := strings.TrimSuffix(line, "\r")
		if len(word) != 5 {
			return nil, fmt.Errorf("line %d: %q is not five ASCII characters", i+1, word)
		}
		for _, letter := range []byte(word) {
			if letter < 'A' || letter > 'Z' {
				return nil, fmt.Errorf("line %d: %q is not uppercase ASCII", i+1, word)
			}
		}
		if i > 0 && words[i-1] >= word {
			return nil, fmt.Errorf("line %d: %q is duplicated or out of order", i+1, word)
		}
		words[i] = word
	}
	return words, nil
}

func splitSolutions(words []string, validationSize, testSize int, seed string) (solutionSplits, error) {
	if validationSize < 0 || testSize < 0 {
		return solutionSplits{}, errors.New("validation and test sizes must not be negative")
	}
	if validationSize+testSize >= len(words) {
		return solutionSplits{}, fmt.Errorf(
			"validation (%d) and test (%d) sizes leave no training solutions out of %d",
			validationSize, testSize, len(words),
		)
	}

	ranked := make([]rankedWord, len(words))
	for i, word := range words {
		ranked[i] = rankedWord{
			word: word,
			rank: sha256.Sum256([]byte(seed + "\n" + word)),
		}
	}
	sort.Slice(ranked, func(i, j int) bool {
		if comparison := bytes.Compare(ranked[i].rank[:], ranked[j].rank[:]); comparison != 0 {
			return comparison < 0
		}
		return ranked[i].word < ranked[j].word
	})

	splits := solutionSplits{
		test:       make([]string, testSize),
		validation: make([]string, validationSize),
		training:   make([]string, len(words)-testSize-validationSize),
	}
	for i := range testSize {
		splits.test[i] = ranked[i].word
	}
	for i := range validationSize {
		splits.validation[i] = ranked[testSize+i].word
	}
	for i := range splits.training {
		splits.training[i] = ranked[testSize+validationSize+i].word
	}

	sort.Strings(splits.training)
	sort.Strings(splits.validation)
	sort.Strings(splits.test)
	return splits, nil
}

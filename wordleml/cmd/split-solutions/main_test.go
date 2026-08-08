package main

import (
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"
)

func TestRunWritesCountedFilenamesWithTrailingNewlines(t *testing.T) {
	inputDir := t.TempDir()
	input := filepath.Join(inputDir, "solutions.csv")
	contents := "ALPHA\nBRAVO\nCHARM\nDELTA\nEAGER\nFLAME\nGIANT\nHONEY\nIDEAL\nJELLY\n"
	if err := os.WriteFile(input, []byte(contents), 0o644); err != nil {
		t.Fatalf("write input: %v", err)
	}

	outputDir := t.TempDir()
	var stdout strings.Builder
	if err := run([]string{
		"-input", input,
		"-output-dir", outputDir,
		"-validation-size", "2",
		"-test-size", "2",
	}, &stdout); err != nil {
		t.Fatalf("run() error = %v", err)
	}

	wantCounts := map[string]int{
		"wordlist-valid-solutions-train-6.csv":      6,
		"wordlist-valid-solutions-validation-2.csv": 2,
		"wordlist-valid-solutions-test-2.csv":       2,
	}
	entries, err := os.ReadDir(outputDir)
	if err != nil {
		t.Fatalf("read output directory: %v", err)
	}
	if len(entries) != len(wantCounts) {
		t.Fatalf("output file count = %d, want %d", len(entries), len(wantCounts))
	}
	for _, entry := range entries {
		wantCount, ok := wantCounts[entry.Name()]
		if !ok {
			t.Errorf("unexpected output file %q", entry.Name())
			continue
		}
		output, err := os.ReadFile(filepath.Join(outputDir, entry.Name()))
		if err != nil {
			t.Errorf("read %s: %v", entry.Name(), err)
			continue
		}
		if !strings.HasSuffix(string(output), "\n") {
			t.Errorf("%s does not end with a newline", entry.Name())
		}
		words, err := parseWordlist(output)
		if err != nil {
			t.Errorf("parse %s: %v", entry.Name(), err)
			continue
		}
		if len(words) != wantCount {
			t.Errorf("%s word count = %d, want %d", entry.Name(), len(words), wantCount)
		}
	}
}

func TestParseWordlist(t *testing.T) {
	words, err := parseWordlist([]byte("ABACK\nABASE\nABATE\n"))
	if err != nil {
		t.Fatalf("parseWordlist() error = %v", err)
	}
	want := []string{"ABACK", "ABASE", "ABATE"}
	if !reflect.DeepEqual(words, want) {
		t.Fatalf("parseWordlist() = %v, want %v", words, want)
	}
}

func TestParseWordlistRejectsInvalidInput(t *testing.T) {
	tests := []struct {
		name     string
		contents string
	}{
		{name: "empty", contents: ""},
		{name: "lowercase", contents: "aback"},
		{name: "wrong length", contents: "WORD"},
		{name: "duplicate", contents: "ABACK\nABACK"},
		{name: "unsorted", contents: "ABASE\nABACK"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if _, err := parseWordlist([]byte(test.contents)); err == nil {
				t.Fatal("parseWordlist() error = nil, want an error")
			}
		})
	}
}

func TestSplitSolutionsIsDeterministicAndDisjoint(t *testing.T) {
	words := []string{
		"ALPHA", "BRAVO", "CHARM", "DELTA", "EAGER",
		"FLAME", "GIANT", "HONEY", "IDEAL", "JELLY",
	}
	first, err := splitSolutions(words, 2, 2, "test-seed")
	if err != nil {
		t.Fatalf("splitSolutions() error = %v", err)
	}
	second, err := splitSolutions(words, 2, 2, "test-seed")
	if err != nil {
		t.Fatalf("splitSolutions() second error = %v", err)
	}
	if !reflect.DeepEqual(first, second) {
		t.Fatalf("splitSolutions() is not deterministic: %#v != %#v", first, second)
	}
	if len(first.training) != 6 || len(first.validation) != 2 || len(first.test) != 2 {
		t.Fatalf(
			"split sizes = train %d, validation %d, test %d; want 6, 2, 2",
			len(first.training), len(first.validation), len(first.test),
		)
	}

	all := append([]string{}, first.training...)
	all = append(all, first.validation...)
	all = append(all, first.test...)
	sort.Strings(all)
	if !reflect.DeepEqual(all, words) {
		t.Fatalf("combined splits = %v, want %v", all, words)
	}
	for name, split := range map[string][]string{
		"training": first.training, "validation": first.validation, "test": first.test,
	} {
		if !sort.StringsAreSorted(split) {
			t.Fatalf("%s split is not sorted: %v", name, split)
		}
	}
}

func TestSplitSolutionsRejectsInvalidSizes(t *testing.T) {
	words := strings.Fields("ALPHA BRAVO CHARM")
	if _, err := splitSolutions(words, 2, 1, "test-seed"); err == nil {
		t.Fatal("splitSolutions() error = nil, want an error")
	}
	if _, err := splitSolutions(words, -1, 1, "test-seed"); err == nil {
		t.Fatal("splitSolutions() negative-size error = nil, want an error")
	}
}

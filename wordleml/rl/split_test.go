package rl

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/sam-bee/wordle-ml_machine-learning/vocabulary"
)

func TestGeneratePPOV1SplitUsesOnlyFrozenTrainingPopulation(t *testing.T) {
	v := testVocabulary(t)
	if got := len(v.Test()); got != 0 {
		t.Fatalf("LoadWithoutFinalTest exposed %d final-test words", got)
	}
	first, firstManifest, err := GeneratePPOV1Split(v)
	if err != nil {
		t.Fatal(err)
	}
	second, secondManifest, err := GeneratePPOV1Split(v)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(first, second) || !reflect.DeepEqual(firstManifest, secondManifest) {
		t.Fatal("PPO v1 split generation was not deterministic")
	}
	if got, want := len(first.Development), 200; got != want {
		t.Fatalf("development count = %d, want %d", got, want)
	}
	if got, want := len(first.Rollout), 1909; got != want {
		t.Fatalf("rollout count = %d, want %d", got, want)
	}
	training := make(map[string]struct{}, vocabulary.NumTrainingSolutions)
	for _, word := range v.Training() {
		training[word] = struct{}{}
	}
	seen := make(map[string]struct{}, vocabulary.NumTrainingSolutions)
	for _, split := range [][]string{first.Development, first.Rollout} {
		for _, word := range split {
			if _, found := training[word]; !found {
				t.Fatalf("split word %q is not in supervised training", word)
			}
			if _, duplicate := seen[word]; duplicate {
				t.Fatalf("split repeats %q", word)
			}
			seen[word] = struct{}{}
		}
	}
	if got := len(seen); got != vocabulary.NumTrainingSolutions {
		t.Fatalf("split covers %d training words, want %d", got, vocabulary.NumTrainingSolutions)
	}
	if firstManifest.Source.SHA256 != v.Hashes().Training {
		t.Fatalf("manifest source hash = %q, want vocabulary training hash %q", firstManifest.Source.SHA256, v.Hashes().Training)
	}
	if firstManifest.Development.Count != 200 || firstManifest.Rollout.Count != 1909 {
		t.Fatalf("manifest counts = development=%d rollout=%d", firstManifest.Development.Count, firstManifest.Rollout.Count)
	}
}

func TestCommittedPPOV1ManifestVerifiesAgainstSealedTestVocabulary(t *testing.T) {
	dataDir := testDataDir()
	v := testVocabulary(t)
	split, manifest, err := LoadPPOV1Split(DefaultPPOV1ManifestPath(dataDir), v)
	if err != nil {
		t.Fatal(err)
	}
	if len(split.Development) != 200 || len(split.Rollout) != 1909 {
		t.Fatalf("loaded split counts = development=%d rollout=%d", len(split.Development), len(split.Rollout))
	}
	if manifest.Validation == "" || manifest.FinalTest == "" {
		t.Fatalf("manifest omission protections = validation=%q final_test=%q", manifest.Validation, manifest.FinalTest)
	}
	contents, err := os.ReadFile(DefaultPPOV1ManifestPath(dataDir))
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(contents), "wordlist-valid-solutions-test") {
		t.Fatal("PPO manifest must not contain a final-test path")
	}
}

func TestLoadPPOV1SplitRejectsAlteredAudit(t *testing.T) {
	v := testVocabulary(t)
	_, manifest, err := GeneratePPOV1Split(v)
	if err != nil {
		t.Fatal(err)
	}
	manifest.Rollout.SHA256 = "not-the-derived-hash"
	path := filepath.Join(t.TempDir(), "ppo-split-v1.json")
	contents, err := json.Marshal(manifest)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, contents, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, _, err := LoadPPOV1Split(path, v); err == nil {
		t.Fatal("LoadPPOV1Split accepted altered rollout audit")
	}
}

func testVocabulary(t *testing.T) *vocabulary.Vocabulary {
	t.Helper()
	v, err := vocabulary.LoadWithoutFinalTest(testDataDir())
	if err != nil {
		t.Fatal(err)
	}
	return v
}

func testDataDir() string {
	if dataDir := os.Getenv("WORDLEML_DATA_DIR"); dataDir != "" {
		return dataDir
	}
	return filepath.Join("..", "..", "data")
}

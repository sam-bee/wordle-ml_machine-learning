// Package rl contains the deliberately small shared pieces used by the
// optional reinforcement-learning experiments. It does not contain a second
// Wordle implementation: environment transitions remain in the game engine.
package rl

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"

	"github.com/sam-bee/wordle-ml_machine-learning/vocabulary"
)

const (
	// PPOV1SplitVersion identifies the immutable initial PPO data split.
	PPOV1SplitVersion = "ppo-split-v1"
	// PPOV1SplitDomain separates this deterministic split from all other uses
	// of SHA-256 over a Wordle word.
	PPOV1SplitDomain    = "wordleml-ppo-rl-split-v1\n"
	ppoDevelopmentCount = 200
)

// Split is the deterministic division of the supervised-training population
// into PPO rollout and PPO development populations. Neither validation nor
// final-test solutions are read to generate or validate this split.
type Split struct {
	Development []string
	Rollout     []string
}

// Manifest is the on-disk, versioned audit record for Split. The lists are
// intentionally represented by their normalized hashes: membership is derived
// solely from the frozen supervised-training list and the public domain string,
// so this small manifest is both reproducible and easy to review.
//
// No field identifies, opens, or contains a final-test solution.
type Manifest struct {
	Version     string    `json:"version"`
	Ranking     string    `json:"ranking"`
	HashDomain  string    `json:"hash_domain"`
	Source      ListAudit `json:"source_training"`
	Development ListAudit `json:"development"`
	Rollout     ListAudit `json:"rollout"`
	Validation  string    `json:"validation"`
	FinalTest   string    `json:"final_test"`
}

// ListAudit is a count and normalized ordered-list hash used to make a split
// manifest independently checkable without duplicating its word lists.
type ListAudit struct {
	Count  int    `json:"count"`
	SHA256 string `json:"sha256"`
}

// DefaultPPOV1ManifestPath returns the repository-data location of the
// committed PPO split manifest for a supplied data directory.
func DefaultPPOV1ManifestPath(dataDir string) string {
	return filepath.Join(dataDir, "rl", "ppo-split-v1.json")
}

// GeneratePPOV1Split derives the initial PPO split from precisely the frozen
// supervised-training solution list. Words are ranked by ascending SHA-256 of
// PPOV1SplitDomain followed by the uppercase word; the first 200 are reserved
// for development evaluation and never enter PPO rollouts or updates.
func GeneratePPOV1Split(v *vocabulary.Vocabulary) (Split, Manifest, error) {
	if v == nil {
		return Split{}, Manifest{}, errors.New("vocabulary must not be nil")
	}
	training := v.Training()
	if len(training) != vocabulary.NumTrainingSolutions {
		return Split{}, Manifest{}, fmt.Errorf("training solution count = %d, want %d", len(training), vocabulary.NumTrainingSolutions)
	}
	if len(training) <= ppoDevelopmentCount {
		return Split{}, Manifest{}, fmt.Errorf("training solution count = %d, need more than %d", len(training), ppoDevelopmentCount)
	}

	type rankedWord struct {
		word string
		hash [sha256.Size]byte
	}
	ranked := make([]rankedWord, len(training))
	for i, word := range training {
		ranked[i] = rankedWord{word: word, hash: sha256.Sum256([]byte(PPOV1SplitDomain + word))}
	}
	sort.Slice(ranked, func(i, j int) bool {
		if comparison := bytes.Compare(ranked[i].hash[:], ranked[j].hash[:]); comparison != 0 {
			return comparison < 0
		}
		return ranked[i].word < ranked[j].word
	})

	split := Split{
		Development: make([]string, ppoDevelopmentCount),
		Rollout:     make([]string, len(ranked)-ppoDevelopmentCount),
	}
	for i, entry := range ranked {
		if i < len(split.Development) {
			split.Development[i] = entry.word
		} else {
			split.Rollout[i-len(split.Development)] = entry.word
		}
	}

	manifest := Manifest{
		Version:    PPOV1SplitVersion,
		Ranking:    "ascending SHA-256 bytes of hash_domain followed by uppercase training word; word lexical order breaks an impossible digest tie",
		HashDomain: PPOV1SplitDomain,
		Source: ListAudit{
			Count:  len(training),
			SHA256: hashWords(training),
		},
		Development: ListAudit{
			Count:  len(split.Development),
			SHA256: hashWords(split.Development),
		},
		Rollout: ListAudit{
			Count:  len(split.Rollout),
			SHA256: hashWords(split.Rollout),
		},
		Validation: "untouched: existing 100-solution validation population is excluded from PPO generation, rollouts, updates, and development evaluation",
		FinalTest:  "sealed and untouched: excluded from PPO generation, rollouts, updates, and development evaluation",
	}
	return split, manifest, nil
}

// LoadPPOV1Split loads the committed manifest then regenerates and verifies
// every count and normalized-list hash against the supplied vocabulary. Callers
// should load v with vocabulary.LoadWithoutFinalTest for PPO work.
func LoadPPOV1Split(manifestPath string, v *vocabulary.Vocabulary) (Split, Manifest, error) {
	contents, err := os.ReadFile(manifestPath)
	if err != nil {
		return Split{}, Manifest{}, fmt.Errorf("read PPO split manifest %s: %w", manifestPath, err)
	}
	var committed Manifest
	if err := json.Unmarshal(contents, &committed); err != nil {
		return Split{}, Manifest{}, fmt.Errorf("decode PPO split manifest %s: %w", manifestPath, err)
	}
	derived, expected, err := GeneratePPOV1Split(v)
	if err != nil {
		return Split{}, Manifest{}, err
	}
	if err := compareManifest(committed, expected); err != nil {
		return Split{}, Manifest{}, fmt.Errorf("verify PPO split manifest %s: %w", manifestPath, err)
	}
	return derived, committed, nil
}

func compareManifest(got, want Manifest) error {
	if got.Version != PPOV1SplitVersion {
		return fmt.Errorf("version = %q, want %q", got.Version, PPOV1SplitVersion)
	}
	if got.Ranking != want.Ranking {
		return errors.New("ranking description differs from the PPO v1 definition")
	}
	if got.HashDomain != PPOV1SplitDomain {
		return fmt.Errorf("hash domain = %q, want %q", got.HashDomain, PPOV1SplitDomain)
	}
	for _, check := range []struct {
		name string
		got  ListAudit
		want ListAudit
	}{
		{name: "source training", got: got.Source, want: want.Source},
		{name: "development", got: got.Development, want: want.Development},
		{name: "rollout", got: got.Rollout, want: want.Rollout},
	} {
		if check.got.Count != check.want.Count || check.got.SHA256 != check.want.SHA256 {
			return fmt.Errorf("%s audit = %#v, want %#v", check.name, check.got, check.want)
		}
	}
	if got.Validation != want.Validation {
		return errors.New("validation protection declaration differs from the PPO v1 definition")
	}
	if got.FinalTest != want.FinalTest {
		return errors.New("final-test protection declaration differs from the PPO v1 definition")
	}
	return nil
}

func hashWords(words []string) string {
	hasher := sha256.New()
	for _, word := range words {
		_, _ = hasher.Write([]byte(word))
		_, _ = hasher.Write([]byte{'\n'})
	}
	return fmt.Sprintf("%x", hasher.Sum(nil))
}

package experiment

import (
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestVerifyGitProvenanceAcceptsConfiguredBranchWithAncestorBase(t *testing.T) {
	var calls [][]string
	git := func(args ...string) (string, error) {
		calls = append(calls, append([]string(nil), args...))
		switch strings.Join(args, " ") {
		case "symbolic-ref --quiet --short HEAD":
			return "experiment/ppo-rl\n", nil
		case "rev-parse --verify --end-of-options base^{commit}":
			return "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n", nil
		case "rev-parse --verify HEAD^{commit}":
			return "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\n", nil
		case "merge-base --is-ancestor aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb":
			return "", nil
		default:
			t.Fatalf("unexpected Git command: %q", args)
			return "", nil
		}
	}
	if err := verifyGitProvenanceWith(git, Config{Branch: "experiment/ppo-rl", BaseCommit: "base"}); err != nil {
		t.Fatal(err)
	}
	want := [][]string{
		{"symbolic-ref", "--quiet", "--short", "HEAD"},
		{"rev-parse", "--verify", "--end-of-options", "base^{commit}"},
		{"rev-parse", "--verify", "HEAD^{commit}"},
		{"merge-base", "--is-ancestor", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"},
	}
	if !reflect.DeepEqual(calls, want) {
		t.Fatalf("Git calls = %#v, want %#v", calls, want)
	}
}

func TestVerifyGitProvenanceRejectsWrongBranchAndUnrelatedBase(t *testing.T) {
	wrongBranch := func(args ...string) (string, error) {
		if reflect.DeepEqual(args, []string{"symbolic-ref", "--quiet", "--short", "HEAD"}) {
			return "master", nil
		}
		t.Fatal("wrong-branch validation made an unnecessary Git call")
		return "", nil
	}
	if err := verifyGitProvenanceWith(wrongBranch, Config{Branch: "experiment/ppo-rl", BaseCommit: "base"}); err == nil || !strings.Contains(err.Error(), "current branch is \"master\"") {
		t.Fatalf("wrong-branch error = %v", err)
	}

	notAncestor := func(args ...string) (string, error) {
		switch strings.Join(args, " ") {
		case "symbolic-ref --quiet --short HEAD":
			return "experiment/ppo-rl", nil
		case "rev-parse --verify --end-of-options base^{commit}":
			return "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", nil
		case "rev-parse --verify HEAD^{commit}":
			return "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", nil
		case "merge-base --is-ancestor aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb":
			return "", errors.New("exit status 1")
		default:
			t.Fatalf("unexpected Git command: %q", args)
			return "", nil
		}
	}
	if err := verifyGitProvenanceWith(notAncestor, Config{Branch: "experiment/ppo-rl", BaseCommit: "base"}); err == nil || !strings.Contains(err.Error(), "not an ancestor") {
		t.Fatalf("unrelated-base error = %v", err)
	}
}

func TestValidateGeneratedRunDirectoryUsesIgnoredRunsChildAndRejectsOverlap(t *testing.T) {
	repositoryRoot := t.TempDir()
	runsRoot := filepath.Join(repositoryRoot, "runs")
	if err := os.Mkdir(runsRoot, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(runsRoot, ".gitignore"), []byte(runsIgnoreContents+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	baseline := filepath.Join(repositoryRoot, "selected-baseline")
	if err := os.Mkdir(baseline, 0o755); err != nil {
		t.Fatal(err)
	}
	want := filepath.Join(runsRoot, "ppo-pilot")
	got, err := validateGeneratedRunDirectory(repositoryRoot, want, baseline)
	if err != nil {
		t.Fatal(err)
	}
	if got != want {
		t.Fatalf("run directory = %q, want %q", got, want)
	}
	if _, err := os.Stat(got); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("validation created run directory: %v", err)
	}

	if _, err := validateGeneratedRunDirectory(repositoryRoot, filepath.Join(repositoryRoot, "outside"), baseline); err == nil || !strings.Contains(err.Error(), "ignored runs directory") {
		t.Fatalf("outside run directory error = %v", err)
	}
	if _, err := validateGeneratedRunDirectory(repositoryRoot, want, runsRoot); err == nil || !strings.Contains(err.Error(), "must be isolated") {
		t.Fatalf("overlapping supervised checkpoint error = %v", err)
	}
}

func TestValidateGeneratedRunDirectoryRequiresAllArtifactIgnorePolicy(t *testing.T) {
	repositoryRoot := t.TempDir()
	runsRoot := filepath.Join(repositoryRoot, "runs")
	if err := os.Mkdir(runsRoot, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(runsRoot, ".gitignore"), []byte("events/\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	baseline := filepath.Join(repositoryRoot, "selected-baseline")
	if err := os.Mkdir(baseline, 0o755); err != nil {
		t.Fatal(err)
	}
	if _, err := validateGeneratedRunDirectory(repositoryRoot, filepath.Join(runsRoot, "ppo-pilot"), baseline); err == nil || !strings.Contains(err.Error(), "all-artifacts ignore policy") {
		t.Fatalf("ignore-policy error = %v", err)
	}
}

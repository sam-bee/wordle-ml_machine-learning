package experiment

import (
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

const runsIgnoreContents = "*\n!.gitignore"

// gitCommand is intentionally tiny so provenance validation can be tested
// without consulting the caller's worktree. The production implementation
// invokes only read-only Git commands.
type gitCommand func(args ...string) (string, error)

// repositoryRootForConfig obtains the worktree root from the configuration
// location, rather than the process working directory. This keeps an invoked
// binary from accidentally writing run artifacts beside an unrelated cwd.
func repositoryRootForConfig(configPath string) (string, error) {
	absoluteConfig, err := filepath.Abs(configPath)
	if err != nil {
		return "", fmt.Errorf("resolve PPO config path: %w", err)
	}
	root, err := gitOutput(filepath.Dir(absoluteConfig), "rev-parse", "--show-toplevel")
	if err != nil {
		return "", fmt.Errorf("find Git worktree for PPO config: %w", err)
	}
	canonicalRoot, err := filepath.EvalSymlinks(root)
	if err != nil {
		return "", fmt.Errorf("resolve Git worktree root: %w", err)
	}
	canonicalConfig, err := filepath.EvalSymlinks(absoluteConfig)
	if err != nil {
		return "", fmt.Errorf("resolve PPO config: %w", err)
	}
	if !pathWithin(canonicalRoot, canonicalConfig) {
		return "", fmt.Errorf("PPO config %q is outside Git worktree %q", canonicalConfig, canonicalRoot)
	}
	return canonicalRoot, nil
}

// verifyGitProvenance prevents the config from being used as a mere label for
// a run made from another branch or unrelated history. BaseCommit is expected
// to be the branch start, not HEAD, so it must be an ancestor of HEAD.
func verifyGitProvenance(repositoryRoot string, config Config) error {
	return verifyGitProvenanceWith(func(args ...string) (string, error) {
		return gitOutput(repositoryRoot, args...)
	}, config)
}

func verifyGitProvenanceWith(git gitCommand, config Config) error {
	if git == nil {
		return errors.New("Git command runner must not be nil")
	}
	branch, err := git("symbolic-ref", "--quiet", "--short", "HEAD")
	if err != nil {
		return fmt.Errorf("read current Git branch: %w", err)
	}
	branch = strings.TrimSpace(branch)
	if branch != config.Branch {
		return fmt.Errorf("PPO config requires Git branch %q, current branch is %q", config.Branch, branch)
	}
	base, err := git("rev-parse", "--verify", "--end-of-options", config.BaseCommit+"^{commit}")
	if err != nil {
		return fmt.Errorf("resolve configured PPO base commit %q: %w", config.BaseCommit, err)
	}
	base = strings.TrimSpace(base)
	head, err := git("rev-parse", "--verify", "HEAD^{commit}")
	if err != nil {
		return fmt.Errorf("resolve current Git HEAD: %w", err)
	}
	head = strings.TrimSpace(head)
	if _, err := git("merge-base", "--is-ancestor", base, head); err != nil {
		return fmt.Errorf("configured PPO base commit %s is not an ancestor of HEAD %s: %w", base, head, err)
	}
	return nil
}

// validateGeneratedRunDirectory limits large, disposable PPO artifacts to
// the repository's intentionally ignored runs/ tree. It also rejects a run
// whose canonical path overlaps the immutable supervised checkpoint.
//
// A run is deliberately one fresh child directory of runs/: this lets us
// resolve its existing parent before creation and makes the Git-ignore policy
// auditable instead of accepting arbitrary user-owned locations.
func validateGeneratedRunDirectory(repositoryRoot, requestedRunDir, supervisedCheckpoint string) (string, error) {
	if strings.TrimSpace(repositoryRoot) == "" || strings.TrimSpace(requestedRunDir) == "" || strings.TrimSpace(supervisedCheckpoint) == "" {
		return "", errors.New("repository root, PPO run directory, and supervised checkpoint are required")
	}
	canonicalRoot, err := filepath.EvalSymlinks(repositoryRoot)
	if err != nil {
		return "", fmt.Errorf("resolve repository root: %w", err)
	}
	runsRoot, err := filepath.EvalSymlinks(filepath.Join(canonicalRoot, "runs"))
	if err != nil {
		return "", fmt.Errorf("resolve repository runs directory: %w", err)
	}
	ignore, err := os.ReadFile(filepath.Join(runsRoot, ".gitignore"))
	if err != nil {
		return "", fmt.Errorf("read repository runs ignore policy: %w", err)
	}
	if strings.TrimSpace(string(ignore)) != runsIgnoreContents {
		return "", fmt.Errorf("repository runs directory %q does not have the required all-artifacts ignore policy", runsRoot)
	}

	absoluteRun, err := filepath.Abs(requestedRunDir)
	if err != nil {
		return "", fmt.Errorf("resolve PPO run directory: %w", err)
	}
	if filepath.Base(absoluteRun) == "." || filepath.Base(absoluteRun) == string(filepath.Separator) {
		return "", errors.New("PPO run directory must name one fresh child of runs")
	}
	canonicalParent, err := filepath.EvalSymlinks(filepath.Dir(absoluteRun))
	if err != nil {
		return "", fmt.Errorf("resolve PPO run parent: %w", err)
	}
	if canonicalParent != runsRoot {
		return "", fmt.Errorf("PPO run directory %q must be a fresh direct child of ignored runs directory %q", absoluteRun, runsRoot)
	}
	runCanonical := filepath.Join(canonicalParent, filepath.Base(absoluteRun))
	if !pathWithin(runsRoot, runCanonical) || runCanonical == runsRoot {
		return "", fmt.Errorf("PPO run directory %q escapes ignored runs directory %q", runCanonical, runsRoot)
	}
	baselineCanonical, err := filepath.EvalSymlinks(supervisedCheckpoint)
	if err != nil {
		return "", fmt.Errorf("resolve supervised checkpoint: %w", err)
	}
	if pathsOverlap(baselineCanonical, runCanonical) {
		return "", fmt.Errorf("PPO run directory %q must be isolated from supervised checkpoint %q", runCanonical, baselineCanonical)
	}
	return runCanonical, nil
}

func gitOutput(directory string, args ...string) (string, error) {
	command := exec.Command("git", append([]string{"-C", directory}, args...)...)
	output, err := command.CombinedOutput()
	if err != nil {
		message := strings.TrimSpace(string(output))
		if message == "" {
			return "", err
		}
		return "", fmt.Errorf("%w: %s", err, message)
	}
	return strings.TrimSpace(string(output)), nil
}

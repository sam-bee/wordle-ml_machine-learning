// Package runmetadata collects the immutable provenance for a training run.
//
// The package deliberately does not initialise a backend. A training command
// supplies the backend and PJRT/GoMLX environment details it actually used;
// Collect and the optional probe perform read-only inspection only.
package runmetadata

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"sort"
	"strings"
	"time"
)

// SchemaVersion identifies the JSON structure written by this package.
const SchemaVersion = 1

// Artifact is one input file, named relative to RepositoryRoot when it lies
// below that directory. SHA256 is lowercase hexadecimal.
type Artifact struct {
	Path   string `json:"path"`
	SHA256 string `json:"sha256"`
}

// Repository identifies the exact checked-out Git commit used by a run.
type Repository struct {
	Path   string `json:"path"`
	Commit string `json:"commit"`
}

// Repositories identifies the three repositories that define a proof run.
type Repositories struct {
	MachineLearning Repository `json:"machine_learning"`
	SyntheticData   Repository `json:"synthetic_data"`
	GameEngine      Repository `json:"game_engine"`
}

// Dataset describes the frozen teacher data loaded by the training command.
type Dataset struct {
	Format  string     `json:"format"`
	Version string     `json:"version"`
	Files   []Artifact `json:"files"`
}

// Splits holds the canonical train, validation, and test split files.
type Splits struct {
	Training   []Artifact `json:"training"`
	Validation []Artifact `json:"validation"`
	Test       []Artifact `json:"test"`
}

// RuntimeMetadata describes the process and the GoMLX version selected for it.
// Detail maps preserve exact device, driver, CUDA, PJRT, and GoMLX facts
// exposed by the backend or its environment.
type RuntimeMetadata struct {
	GoVersion    string            `json:"go_version"`
	GOOS         string            `json:"goos"`
	GOARCH       string            `json:"goarch"`
	GoMLXVersion string            `json:"gomlx_version"`
	GoMLXDetails map[string]string `json:"gomlx_details,omitempty"`
	Backend      string            `json:"backend"`
	GPUDetails   map[string]string `json:"gpu_details,omitempty"`
	CUDADetails  map[string]string `json:"cuda_details,omitempty"`
	PJRTDetails  map[string]string `json:"pjrt_details,omitempty"`
}

// Manifest is the complete provenance record for one training run. All paths
// use slash separators, which makes manifests portable and stable in diffs.
type Manifest struct {
	SchemaVersion       int             `json:"schema_version"`
	CollectedAt         time.Time       `json:"collected_at"`
	Repositories        Repositories    `json:"repositories"`
	Dataset             Dataset         `json:"dataset"`
	Vocabulary          []Artifact      `json:"vocabulary"`
	Splits              Splits          `json:"splits"`
	ModelParameterCount int64           `json:"model_parameter_count"`
	Runtime             RuntimeMetadata `json:"runtime"`
	Seed                int64           `json:"seed"`
	EffectiveConfig     json.RawMessage `json:"effective_config"`
}

// CollectOptions supplies the factual inputs used to build a manifest.
// RepositoryRoot is used only to make paths portable; when empty, the machine
// learning repository is used. EffectiveConfig must contain the complete,
// already-resolved training configuration as valid JSON.
type CollectOptions struct {
	RepositoryRoot            string
	MachineLearningRepository string
	SyntheticDataRepository   string
	GameEngineRepository      string

	DatasetFormat   string
	DatasetVersion  string
	DatasetFiles    []string
	Vocabulary      []string
	TrainingSplit   []string
	ValidationSplit []string
	TestSplit       []string

	ModelParameterCount int64
	Runtime             RuntimeMetadata
	Seed                int64
	EffectiveConfig     json.RawMessage
	CollectedAt         time.Time
}

// Collect hashes the supplied loaded files and reads HEAD from each supplied
// repository. It does not modify Git state, files, or the execution backend.
func Collect(options CollectOptions) (Manifest, error) {
	root, err := collectionRoot(options.RepositoryRoot, options.MachineLearningRepository)
	if err != nil {
		return Manifest{}, err
	}
	repositories, err := collectRepositories(root, options)
	if err != nil {
		return Manifest{}, err
	}
	datasetFiles, err := collectArtifacts(root, options.DatasetFiles, "dataset")
	if err != nil {
		return Manifest{}, err
	}
	vocabulary, err := collectArtifacts(root, options.Vocabulary, "vocabulary")
	if err != nil {
		return Manifest{}, err
	}
	training, err := collectArtifacts(root, options.TrainingSplit, "training split")
	if err != nil {
		return Manifest{}, err
	}
	validation, err := collectArtifacts(root, options.ValidationSplit, "validation split")
	if err != nil {
		return Manifest{}, err
	}
	test, err := collectArtifacts(root, options.TestSplit, "test split")
	if err != nil {
		return Manifest{}, err
	}

	collectedAt := options.CollectedAt
	if collectedAt.IsZero() {
		collectedAt = time.Now()
	}
	runtimeMetadata, err := mergeRuntimeMetadata(options.Runtime, options.MachineLearningRepository)
	if err != nil {
		return Manifest{}, err
	}
	manifest := Manifest{
		SchemaVersion: SchemaVersion,
		CollectedAt:   collectedAt.UTC(),
		Repositories:  repositories,
		Dataset: Dataset{
			Format:  options.DatasetFormat,
			Version: options.DatasetVersion,
			Files:   datasetFiles,
		},
		Vocabulary:          vocabulary,
		Splits:              Splits{Training: training, Validation: validation, Test: test},
		ModelParameterCount: options.ModelParameterCount,
		Runtime:             runtimeMetadata,
		Seed:                options.Seed,
		EffectiveConfig:     append(json.RawMessage(nil), options.EffectiveConfig...),
	}
	if err := manifest.Validate(); err != nil {
		return Manifest{}, err
	}
	return manifest, nil
}

// Validate checks that a manifest contains every datum needed to reproduce a
// proof run. It also checks deterministic ordering of every artifact list.
func (manifest Manifest) Validate() error {
	if manifest.SchemaVersion != SchemaVersion {
		return fmt.Errorf("unsupported manifest schema version %d", manifest.SchemaVersion)
	}
	if manifest.CollectedAt.IsZero() || manifest.CollectedAt.UTC().Format(time.RFC3339Nano) != manifest.CollectedAt.Format(time.RFC3339Nano) {
		return errors.New("collected_at must be a UTC timestamp")
	}
	if err := validateRepositories(manifest.Repositories); err != nil {
		return err
	}
	if strings.TrimSpace(manifest.Dataset.Format) == "" || strings.TrimSpace(manifest.Dataset.Version) == "" {
		return errors.New("dataset format and version must not be empty")
	}
	if err := validateArtifacts("dataset files", manifest.Dataset.Files); err != nil {
		return err
	}
	if err := validateArtifacts("vocabulary", manifest.Vocabulary); err != nil {
		return err
	}
	if err := validateArtifacts("training split", manifest.Splits.Training); err != nil {
		return err
	}
	if err := validateArtifacts("validation split", manifest.Splits.Validation); err != nil {
		return err
	}
	if err := validateArtifacts("test split", manifest.Splits.Test); err != nil {
		return err
	}
	if manifest.ModelParameterCount <= 0 {
		return fmt.Errorf("model parameter count must be positive, got %d", manifest.ModelParameterCount)
	}
	if err := validateRuntime(manifest.Runtime); err != nil {
		return err
	}
	if manifest.Seed == 0 {
		return errors.New("seed must not be zero")
	}
	if !json.Valid(manifest.EffectiveConfig) {
		return errors.New("effective configuration is not valid JSON")
	}
	var configuration map[string]json.RawMessage
	if err := json.Unmarshal(manifest.EffectiveConfig, &configuration); err != nil || configuration == nil {
		return errors.New("effective configuration must be a JSON object")
	}
	return nil
}

// JSON returns canonical, indented JSON suitable for metadata.json. Artifact
// slices have already been ordered by Collect and map keys are ordered by the
// standard JSON encoder.
func (manifest Manifest) JSON() ([]byte, error) {
	if err := manifest.Validate(); err != nil {
		return nil, err
	}
	contents, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		return nil, fmt.Errorf("encode manifest JSON: %w", err)
	}
	return append(contents, '\n'), nil
}

// VerifyEvaluationInputs checks every DataDir input that an independent
// evaluation is allowed to consume before that code opens vocabulary or WDIT
// records. It intentionally excludes training and test WDIT files.
func VerifyEvaluationInputs(manifest Manifest, dataDir string) error {
	if err := manifest.Validate(); err != nil {
		return fmt.Errorf("validate immutable metadata: %w", err)
	}
	if strings.TrimSpace(dataDir) == "" {
		return errors.New("evaluation data directory must not be empty")
	}
	checks := []struct {
		artifacts []Artifact
		basename  string
		path      string
	}{
		{manifest.Dataset.Files, "wordle-validation.bin", filepath.Join(dataDir, "imitation", "wordle-validation.bin")},
		{manifest.Dataset.Files, "wordle-validation.json", filepath.Join(dataDir, "imitation", "wordle-validation.json")},
		{manifest.Vocabulary, "wordlist-action-space-4739.csv", filepath.Join(dataDir, "wordlist-action-space-4739.csv")},
		{manifest.Vocabulary, "wordlist-valid-solutions-all-2309.csv", filepath.Join(dataDir, "wordlist-valid-solutions-all-2309.csv")},
		{manifest.Splits.Validation, "wordlist-valid-solutions-validation-100.csv", filepath.Join(dataDir, "wordlist-valid-solutions-validation-100.csv")},
	}
	for _, check := range checks {
		expected, err := artifactByBaseName(check.artifacts, check.basename)
		if err != nil {
			return err
		}
		actual, err := FileSHA256(check.path)
		if err != nil {
			return fmt.Errorf("hash evaluation input %q: %w", check.path, err)
		}
		if actual != expected.SHA256 {
			return fmt.Errorf("evaluation input %q hash differs from immutable metadata", check.basename)
		}
	}
	return nil
}

// VerifyEvaluationRepositories checks that the three repositories used to
// evaluate a checkpoint remain at the exact commits recorded for its run.
func VerifyEvaluationRepositories(manifest Manifest, machineLearning, syntheticData, gameEngine string) error {
	if err := manifest.Validate(); err != nil {
		return err
	}
	for _, check := range []struct {
		label, path string
		expected    Repository
	}{
		{"machine learning", machineLearning, manifest.Repositories.MachineLearning},
		{"synthetic data", syntheticData, manifest.Repositories.SyntheticData},
		{"game engine", gameEngine, manifest.Repositories.GameEngine},
	} {
		if strings.TrimSpace(check.path) == "" {
			return fmt.Errorf("%s repository path is required for evaluation", check.label)
		}
		if err := requireCleanGitWorktree(check.path); err != nil {
			return fmt.Errorf("%s evaluation repository: %w", check.label, err)
		}
		actual, err := gitCommit(check.path)
		if err != nil {
			return fmt.Errorf("read %s evaluation repository commit: %w", check.label, err)
		}
		if actual != check.expected.Commit {
			return fmt.Errorf("%s evaluation repository commit %s differs from immutable metadata %s", check.label, actual, check.expected.Commit)
		}
	}
	return nil
}

// VerifyEvaluationRuntime checks the backend identity available before any
// inference graph is executed. Hardware details are already fixed by the
// container's one-visible-GPU configuration and recorded in the manifest.
func VerifyEvaluationRuntime(manifest Manifest, backendName, backendDescription string) error {
	if manifest.Runtime.Backend != "xla:cuda" {
		return fmt.Errorf("immutable run backend %q is not xla:cuda", manifest.Runtime.Backend)
	}
	if expected := manifest.Runtime.GoMLXDetails["backend_name"]; expected != "" && expected != backendName {
		return fmt.Errorf("evaluation backend name %q differs from immutable metadata %q", backendName, expected)
	}
	if expected := manifest.Runtime.PJRTDetails["backend_description"]; expected != "" && expected != backendDescription {
		return fmt.Errorf("evaluation backend description differs from immutable metadata")
	}
	return nil
}

func artifactByBaseName(artifacts []Artifact, basename string) (Artifact, error) {
	var matched *Artifact
	for index := range artifacts {
		if filepath.Base(artifacts[index].Path) != basename {
			continue
		}
		if matched != nil {
			return Artifact{}, fmt.Errorf("immutable metadata contains multiple artifacts named %q", basename)
		}
		matched = &artifacts[index]
	}
	if matched == nil {
		return Artifact{}, fmt.Errorf("immutable metadata lacks required evaluation artifact %q", basename)
	}
	return *matched, nil
}

func collectionRoot(configuredRoot, machineLearningRepository string) (string, error) {
	root := configuredRoot
	if strings.TrimSpace(root) == "" {
		root = machineLearningRepository
	}
	if strings.TrimSpace(root) == "" {
		return "", errors.New("repository root and machine learning repository must not both be empty")
	}
	abs, err := filepath.Abs(root)
	if err != nil {
		return "", fmt.Errorf("make repository root absolute: %w", err)
	}
	return filepath.Clean(abs), nil
}

func collectRepositories(root string, options CollectOptions) (Repositories, error) {
	ml, err := collectRepository(root, options.MachineLearningRepository, "machine learning repository")
	if err != nil {
		return Repositories{}, err
	}
	synthetic, err := collectRepository(root, options.SyntheticDataRepository, "synthetic data repository")
	if err != nil {
		return Repositories{}, err
	}
	game, err := collectRepository(root, options.GameEngineRepository, "game engine repository")
	if err != nil {
		return Repositories{}, err
	}
	return Repositories{MachineLearning: ml, SyntheticData: synthetic, GameEngine: game}, nil
}

func collectRepository(root, path, label string) (Repository, error) {
	if strings.TrimSpace(path) == "" {
		return Repository{}, fmt.Errorf("%s path must not be empty", label)
	}
	abs, err := filepath.Abs(path)
	if err != nil {
		return Repository{}, fmt.Errorf("make %s path absolute: %w", label, err)
	}
	if err := requireCleanGitWorktree(abs); err != nil {
		return Repository{}, fmt.Errorf("%s: %w", label, err)
	}
	commit, err := gitCommit(abs)
	if err != nil {
		return Repository{}, err
	}
	return Repository{Path: displayPath(root, abs), Commit: commit}, nil
}

// requireCleanGitWorktree prevents a manifest from claiming that a run is
// reproducible from HEAD when source differs from that commit. Generated run
// artifacts remain irrelevant because the repository ignores runs/.
func requireCleanGitWorktree(repositoryDir string) error {
	output, err := exec.Command(
		"git", "-C", repositoryDir, "status", "--porcelain", "--untracked-files=normal",
	).Output()
	if err != nil {
		return fmt.Errorf("inspect Git worktree %q: %w", repositoryDir, err)
	}
	if strings.TrimSpace(string(output)) != "" {
		return fmt.Errorf("Git worktree %q has uncommitted changes; commit them before starting a proof run", repositoryDir)
	}
	return nil
}

func collectArtifacts(root string, paths []string, label string) ([]Artifact, error) {
	if len(paths) == 0 {
		return nil, fmt.Errorf("%s files must not be empty", label)
	}
	artifacts := make([]Artifact, 0, len(paths))
	for _, path := range paths {
		if strings.TrimSpace(path) == "" {
			return nil, fmt.Errorf("%s file path must not be empty", label)
		}
		abs, err := filepath.Abs(path)
		if err != nil {
			return nil, fmt.Errorf("make %s file path absolute: %w", label, err)
		}
		digest, err := FileSHA256(abs)
		if err != nil {
			return nil, err
		}
		artifacts = append(artifacts, Artifact{Path: displayPath(root, abs), SHA256: digest})
	}
	sort.Slice(artifacts, func(i, j int) bool { return artifacts[i].Path < artifacts[j].Path })
	for index := 1; index < len(artifacts); index++ {
		if artifacts[index-1].Path == artifacts[index].Path {
			return nil, fmt.Errorf("%s lists %q more than once", label, artifacts[index].Path)
		}
	}
	return artifacts, nil
}

func gitCommit(repositoryDir string) (string, error) {
	output, err := exec.Command("git", "-C", repositoryDir, "rev-parse", "HEAD").Output()
	if err != nil {
		return "", fmt.Errorf("read Git commit for %q: %w", repositoryDir, err)
	}
	commit := strings.TrimSpace(string(output))
	if commit == "" {
		return "", fmt.Errorf("Git returned an empty commit for %q", repositoryDir)
	}
	return commit, nil
}

// FileSHA256 returns the SHA-256 digest of path without loading it into memory.
func FileSHA256(path string) (string, error) {
	file, err := os.Open(path)
	if err != nil {
		return "", fmt.Errorf("open %q for SHA-256: %w", path, err)
	}
	defer func() { _ = file.Close() }()
	digest := sha256.New()
	if _, err := io.Copy(digest, file); err != nil {
		return "", fmt.Errorf("hash %q with SHA-256: %w", path, err)
	}
	return hex.EncodeToString(digest.Sum(nil)), nil
}

func mergeRuntimeMetadata(supplied RuntimeMetadata, repository string) (RuntimeMetadata, error) {
	metadata := RuntimeMetadata{GoVersion: runtime.Version(), GOOS: runtime.GOOS, GOARCH: runtime.GOARCH}
	if buildInfo, ok := debug.ReadBuildInfo(); ok {
		for _, dependency := range buildInfo.Deps {
			if dependency.Path == "github.com/gomlx/gomlx" {
				metadata.GoMLXVersion = dependency.Version
				break
			}
		}
	}
	if supplied.GoVersion != "" {
		metadata.GoVersion = supplied.GoVersion
	}
	if supplied.GOOS != "" {
		metadata.GOOS = supplied.GOOS
	}
	if supplied.GOARCH != "" {
		metadata.GOARCH = supplied.GOARCH
	}
	if supplied.GoMLXVersion != "" {
		metadata.GoMLXVersion = supplied.GoMLXVersion
	}
	metadata.Backend = supplied.Backend
	metadata.GoMLXDetails = cloneDetails(supplied.GoMLXDetails)
	metadata.GPUDetails = cloneDetails(supplied.GPUDetails)
	metadata.CUDADetails = cloneDetails(supplied.CUDADetails)
	metadata.PJRTDetails = cloneDetails(supplied.PJRTDetails)
	if metadata.GoMLXVersion == "" {
		version, err := goMLXVersionFromModule(repository)
		if err != nil {
			return RuntimeMetadata{}, err
		}
		metadata.GoMLXVersion = version
	}
	return metadata, nil
}

func goMLXVersionFromModule(repository string) (string, error) {
	for _, path := range []string{filepath.Join(repository, "go.mod"), filepath.Join(repository, "wordleml", "go.mod")} {
		contents, err := os.ReadFile(path)
		if errors.Is(err, os.ErrNotExist) {
			continue
		}
		if err != nil {
			return "", fmt.Errorf("read Go module file %q: %w", path, err)
		}
		for _, line := range strings.Split(string(contents), "\n") {
			fields := strings.Fields(line)
			if len(fields) >= 3 && fields[0] == "require" {
				fields = fields[1:]
			}
			if len(fields) >= 2 && fields[0] == "github.com/gomlx/gomlx" {
				return fields[1], nil
			}
		}
	}
	return "", fmt.Errorf("find github.com/gomlx/gomlx version in %q", repository)
}

func cloneDetails(details map[string]string) map[string]string {
	if len(details) == 0 {
		return nil
	}
	copy := make(map[string]string, len(details))
	for key, value := range details {
		copy[key] = value
	}
	return copy
}

func displayPath(root, path string) string {
	relative, err := filepath.Rel(root, path)
	if err == nil && relative != ".." && !strings.HasPrefix(relative, ".."+string(filepath.Separator)) && !filepath.IsAbs(relative) {
		return filepath.ToSlash(relative)
	}
	return filepath.ToSlash(path)
}

func validateRepositories(repositories Repositories) error {
	for _, repository := range []struct {
		name  string
		value Repository
	}{
		{"machine learning repository", repositories.MachineLearning},
		{"synthetic data repository", repositories.SyntheticData},
		{"game engine repository", repositories.GameEngine},
	} {
		if strings.TrimSpace(repository.value.Path) == "" || strings.TrimSpace(repository.value.Commit) == "" {
			return fmt.Errorf("%s path and commit must not be empty", repository.name)
		}
	}
	return nil
}

func validateArtifacts(label string, artifacts []Artifact) error {
	if len(artifacts) == 0 {
		return fmt.Errorf("%s must not be empty", label)
	}
	lastPath := ""
	for _, artifact := range artifacts {
		if strings.TrimSpace(artifact.Path) == "" {
			return fmt.Errorf("%s contains an empty path", label)
		}
		if len(artifact.SHA256) != sha256.Size*2 {
			return fmt.Errorf("%s hash for %q has length %d, want %d", label, artifact.Path, len(artifact.SHA256), sha256.Size*2)
		}
		if _, err := hex.DecodeString(artifact.SHA256); err != nil || artifact.SHA256 != strings.ToLower(artifact.SHA256) {
			return fmt.Errorf("%s hash for %q is not lowercase hexadecimal", label, artifact.Path)
		}
		if lastPath >= artifact.Path {
			return fmt.Errorf("%s are not deterministically ordered", label)
		}
		lastPath = artifact.Path
	}
	return nil
}

func validateRuntime(metadata RuntimeMetadata) error {
	if strings.TrimSpace(metadata.GoVersion) == "" || strings.TrimSpace(metadata.GOOS) == "" || strings.TrimSpace(metadata.GOARCH) == "" || strings.TrimSpace(metadata.GoMLXVersion) == "" {
		return errors.New("Go and GoMLX runtime metadata must not be empty")
	}
	if strings.TrimSpace(metadata.Backend) == "" {
		return errors.New("execution backend must not be empty")
	}
	detailSections := []struct {
		name    string
		details map[string]string
	}{
		{"GPUDetails", metadata.GPUDetails},
		{"CUDADetails", metadata.CUDADetails},
		{"PJRTDetails", metadata.PJRTDetails},
		{"GoMLXDetails", metadata.GoMLXDetails},
	}
	if metadata.Backend == "xla:cuda" {
		for _, section := range detailSections {
			if len(section.details) == 0 {
				return fmt.Errorf("xla:cuda backend requires non-empty %s", section.name)
			}
		}
	}
	for _, section := range detailSections {
		details := section.details
		for key, value := range details {
			if strings.TrimSpace(key) == "" || strings.TrimSpace(value) == "" {
				return fmt.Errorf("%s must not contain empty keys or values", section.name)
			}
		}
	}
	return nil
}

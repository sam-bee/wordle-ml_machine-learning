package runmetadata

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestCollectRecordsCompleteDeterministicManifest(t *testing.T) {
	root := t.TempDir()
	mlRepository := makeGitRepository(t, root, "ml")
	syntheticRepository := makeGitRepository(t, root, "synthetic")
	gameRepository := makeGitRepository(t, root, "game")
	writeFile(t, filepath.Join(mlRepository, "go.mod"), "module example.test/ml\n\nrequire github.com/gomlx/gomlx v0.28.0\n")
	runGit(t, mlRepository, "add", "go.mod")
	runGit(t, mlRepository, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "add module")

	datasetB := writeFile(t, filepath.Join(mlRepository, "data", "dataset-b.bin"), "dataset-b")
	datasetA := writeFile(t, filepath.Join(mlRepository, "data", "dataset-a.json"), "dataset-a")
	vocabulary := writeFile(t, filepath.Join(mlRepository, "vocabulary.txt"), "crane\nslate\n")
	training := writeFile(t, filepath.Join(mlRepository, "splits", "train.txt"), "crane\n")
	validation := writeFile(t, filepath.Join(mlRepository, "splits", "validation.txt"), "slate\n")
	test := writeFile(t, filepath.Join(mlRepository, "splits", "test.txt"), "other\n")
	runGit(t, mlRepository, "add", ".")
	runGit(t, mlRepository, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "add proof inputs")

	manifest, err := Collect(CollectOptions{
		RepositoryRoot:            mlRepository,
		MachineLearningRepository: mlRepository,
		SyntheticDataRepository:   syntheticRepository,
		GameEngineRepository:      gameRepository,
		DatasetFormat:             "WDIT",
		DatasetVersion:            "3",
		DatasetFiles:              []string{datasetB, datasetA},
		Vocabulary:                []string{vocabulary},
		TrainingSplit:             []string{training},
		ValidationSplit:           []string{validation},
		TestSplit:                 []string{test},
		ModelParameterCount:       4739,
		Runtime: RuntimeMetadata{
			Backend:      "xla:cuda",
			GPUDetails:   map[string]string{"name": "RTX 5070 Ti", "compute_capability": "12.0"},
			CUDADetails:  map[string]string{"runtime_version": "13.1"},
			PJRTDetails:  map[string]string{"plugin": "cuda"},
			GoMLXDetails: map[string]string{"compute_backend": "xla:cuda"},
		},
		Seed:            20260808,
		EffectiveConfig: json.RawMessage(`{"batch_size":256,"learning_rate":0.0003}`),
		CollectedAt:     time.Date(2026, time.August, 8, 12, 0, 0, 0, time.FixedZone("BST", 3600)),
	})
	if err != nil {
		t.Fatalf("Collect: %v", err)
	}
	if got, want := manifest.CollectedAt.Location(), time.UTC; got != want {
		t.Fatalf("CollectedAt location = %v, want %v", got, want)
	}
	if manifest.Runtime.GoMLXVersion != "v0.28.0" {
		t.Fatalf("GoMLX version = %q, want v0.28.0", manifest.Runtime.GoMLXVersion)
	}
	if manifest.Repositories.MachineLearning.Path != "." {
		t.Fatalf("machine learning repository path = %q, want .", manifest.Repositories.MachineLearning.Path)
	}
	if manifest.Repositories.SyntheticData.Path != filepath.ToSlash(syntheticRepository) {
		t.Fatalf("synthetic repository path = %q, want absolute path", manifest.Repositories.SyntheticData.Path)
	}
	if got, want := manifest.Dataset.Files[0].Path, "data/dataset-a.json"; got != want {
		t.Fatalf("first dataset path = %q, want %q", got, want)
	}
	if got, want := manifest.Dataset.Files[1].Path, "data/dataset-b.bin"; got != want {
		t.Fatalf("second dataset path = %q, want %q", got, want)
	}
	if len(manifest.Dataset.Files[0].SHA256) != 64 {
		t.Fatalf("dataset hash = %q, want SHA-256", manifest.Dataset.Files[0].SHA256)
	}

	contents, err := manifest.JSON()
	if err != nil {
		t.Fatalf("JSON: %v", err)
	}
	if !strings.HasSuffix(string(contents), "\n") {
		t.Fatal("manifest JSON has no trailing newline")
	}
	var decoded Manifest
	if err := json.Unmarshal(contents, &decoded); err != nil {
		t.Fatalf("decode manifest JSON: %v", err)
	}
	if err := decoded.Validate(); err != nil {
		t.Fatalf("Validate decoded manifest: %v", err)
	}

}

func TestCollectRejectsMissingRequiredInput(t *testing.T) {
	repository := makeGitRepository(t, t.TempDir(), "repo")
	file := writeFile(t, filepath.Join(repository, "input"), "input")
	runGit(t, repository, "add", "input")
	runGit(t, repository, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "add input")
	_, err := Collect(CollectOptions{
		MachineLearningRepository: repository,
		SyntheticDataRepository:   repository,
		GameEngineRepository:      repository,
		DatasetFormat:             "WDIT",
		DatasetVersion:            "3",
		DatasetFiles:              []string{file},
		Vocabulary:                []string{file},
		TrainingSplit:             []string{file},
		ValidationSplit:           []string{file},
		ModelParameterCount:       1,
		Runtime: RuntimeMetadata{
			GoMLXVersion: "v0.28.0",
			Backend:      "xla:cuda",
		},
		Seed:            1,
		EffectiveConfig: json.RawMessage(`{}`),
	})
	if err == nil || !strings.Contains(err.Error(), "test split") {
		t.Fatalf("Collect missing test split error = %v, want test split error", err)
	}
}

func TestCollectRejectsTrackedChanges(t *testing.T) {
	repository := makeGitRepository(t, t.TempDir(), "repo")
	file := filepath.Join(repository, "input")
	writeFile(t, file, "input")
	runGit(t, repository, "add", "input")
	runGit(t, repository, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "add input")
	writeFile(t, file, "changed")

	_, err := Collect(CollectOptions{
		MachineLearningRepository: repository,
		SyntheticDataRepository:   repository,
		GameEngineRepository:      repository,
		DatasetFormat:             "WDIT",
		DatasetVersion:            "3",
		DatasetFiles:              []string{file},
		Vocabulary:                []string{file},
		TrainingSplit:             []string{file},
		ValidationSplit:           []string{file},
		TestSplit:                 []string{file},
		ModelParameterCount:       1,
		Runtime: RuntimeMetadata{
			GoMLXVersion: "v0.28.0",
			Backend:      "xla:cuda",
		},
		Seed:            1,
		EffectiveConfig: json.RawMessage(`{}`),
	})
	if err == nil || !strings.Contains(err.Error(), "uncommitted changes") {
		t.Fatalf("Collect dirty repository error = %v, want uncommitted-changes error", err)
	}
}

func TestValidateRejectsNonCanonicalManifest(t *testing.T) {
	manifest := validManifest()
	manifest.Dataset.Files = []Artifact{
		{Path: "z", SHA256: strings.Repeat("a", 64)},
		{Path: "a", SHA256: strings.Repeat("b", 64)},
	}
	if err := manifest.Validate(); err == nil || !strings.Contains(err.Error(), "deterministically ordered") {
		t.Fatalf("Validate unordered files error = %v", err)
	}
	manifest = validManifest()
	manifest.CollectedAt = manifest.CollectedAt.In(time.FixedZone("BST", 3600))
	if err := manifest.Validate(); err == nil || !strings.Contains(err.Error(), "UTC") {
		t.Fatalf("Validate non-UTC timestamp error = %v", err)
	}
	manifest = validManifest()
	manifest.EffectiveConfig = json.RawMessage(`[]`)
	if err := manifest.Validate(); err == nil || !strings.Contains(err.Error(), "JSON object") {
		t.Fatalf("Validate non-object config error = %v", err)
	}
}

func TestValidateRejectsIncompleteCUDARuntimeDetails(t *testing.T) {
	for name, clear := range map[string]func(*RuntimeMetadata){
		"GPUDetails":   func(runtime *RuntimeMetadata) { runtime.GPUDetails = nil },
		"CUDADetails":  func(runtime *RuntimeMetadata) { runtime.CUDADetails = nil },
		"PJRTDetails":  func(runtime *RuntimeMetadata) { runtime.PJRTDetails = nil },
		"GoMLXDetails": func(runtime *RuntimeMetadata) { runtime.GoMLXDetails = nil },
	} {
		t.Run(name, func(t *testing.T) {
			manifest := validManifest()
			clear(&manifest.Runtime)
			err := manifest.Validate()
			if err == nil || !strings.Contains(err.Error(), "requires non-empty "+name) {
				t.Fatalf("Validate missing %s error = %v", name, err)
			}
		})
	}
}

func TestManifestAcceptsSealedFinalTestWithJSONNull(t *testing.T) {
	manifest := validManifest()
	manifest.FinalTestSealed = true
	manifest.Splits.Test = nil
	contents, err := json.Marshal(manifest)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(contents), `"final_test_sealed_unopened":true`) || !strings.Contains(string(contents), `"test":null`) {
		t.Fatalf("sealed manifest JSON = %s", contents)
	}
	var decoded Manifest
	if err := json.Unmarshal(contents, &decoded); err != nil {
		t.Fatal(err)
	}
	if !decoded.FinalTestSealed || decoded.Splits.Test != nil {
		t.Fatalf("decoded sealed state = %#v", decoded)
	}
	if err := decoded.Validate(); err != nil {
		t.Fatalf("Validate sealed manifest: %v", err)
	}
	decoded.Splits.Test = []Artifact{{Path: "test", SHA256: strings.Repeat("a", 64)}}
	if err := decoded.Validate(); err == nil || !strings.Contains(err.Error(), "must not contain") {
		t.Fatalf("sealed manifest with test artifact error = %v", err)
	}
}

func TestVerifyEvaluationInputsRejectsAlteredValidationAndVocabularyBeforeLoad(t *testing.T) {
	dataDir := t.TempDir()
	validationBin := writeFile(t, filepath.Join(dataDir, "imitation", "wordle-validation.bin"), "validation binary")
	validationJSON := writeFile(t, filepath.Join(dataDir, "imitation", "wordle-validation.json"), "validation json")
	actions := writeFile(t, filepath.Join(dataDir, "wordlist-action-space-4739.csv"), "actions")
	solutions := writeFile(t, filepath.Join(dataDir, "wordlist-valid-solutions-all-2309.csv"), "solutions")
	validationSplit := writeFile(t, filepath.Join(dataDir, "wordlist-valid-solutions-validation-100.csv"), "validation split")
	manifest := validManifest()
	manifest.Dataset.Files = []Artifact{testArtifact(t, validationBin), testArtifact(t, validationJSON)}
	manifest.Vocabulary = []Artifact{testArtifact(t, actions), testArtifact(t, solutions)}
	manifest.Splits.Validation = []Artifact{testArtifact(t, validationSplit)}
	if err := VerifyEvaluationInputs(manifest, dataDir); err != nil {
		t.Fatalf("VerifyEvaluationInputs: %v", err)
	}
	for _, name := range []string{"validation binary", "validation JSON", "canonical vocabulary", "validation split"} {
		t.Run(name, func(t *testing.T) {
			// Each case starts from a fresh immutable manifest and data files to
			// prove exactly the changed evaluation input is detected.
			data := t.TempDir()
			bin := writeFile(t, filepath.Join(data, "imitation", "wordle-validation.bin"), "validation binary")
			js := writeFile(t, filepath.Join(data, "imitation", "wordle-validation.json"), "validation json")
			act := writeFile(t, filepath.Join(data, "wordlist-action-space-4739.csv"), "actions")
			sol := writeFile(t, filepath.Join(data, "wordlist-valid-solutions-all-2309.csv"), "solutions")
			split := writeFile(t, filepath.Join(data, "wordlist-valid-solutions-validation-100.csv"), "validation split")
			current := validManifest()
			current.Dataset.Files = []Artifact{testArtifact(t, bin), testArtifact(t, js)}
			current.Vocabulary = []Artifact{testArtifact(t, act), testArtifact(t, sol)}
			current.Splits.Validation = []Artifact{testArtifact(t, split)}
			switch name {
			case "validation binary":
				writeFile(t, bin, "changed validation binary")
			case "validation JSON":
				writeFile(t, js, "changed validation json")
			case "canonical vocabulary":
				writeFile(t, act, "changed actions")
			case "validation split":
				writeFile(t, split, "changed split")
			}
			if err := VerifyEvaluationInputs(current, data); err == nil {
				t.Fatal("altered evaluation input was accepted")
			}
		})
	}
}

func TestVerifyEvaluationRepositoriesRejectsDirtyWorktree(t *testing.T) {
	root := t.TempDir()
	ml := makeGitRepository(t, root, "ml")
	synthetic := makeGitRepository(t, root, "synthetic")
	game := makeGitRepository(t, root, "game")
	manifest := validManifest()
	for repository, destination := range map[string]*Repository{
		ml: &manifest.Repositories.MachineLearning, synthetic: &manifest.Repositories.SyntheticData, game: &manifest.Repositories.GameEngine,
	} {
		commit, err := gitCommit(repository)
		if err != nil {
			t.Fatal(err)
		}
		destination.Commit = commit
	}
	if err := VerifyEvaluationRepositories(manifest, ml, synthetic, game); err != nil {
		t.Fatalf("verify clean repositories: %v", err)
	}
	writeFile(t, filepath.Join(game, "uncommitted.txt"), "changed evaluation code")
	if err := VerifyEvaluationRepositories(manifest, ml, synthetic, game); err == nil {
		t.Fatal("dirty evaluation repository was accepted")
	}
}

func testArtifact(t *testing.T, path string) Artifact {
	t.Helper()
	digest, err := FileSHA256(path)
	if err != nil {
		t.Fatal(err)
	}
	return Artifact{Path: filepath.Base(path), SHA256: digest}
}

func validManifest() Manifest {
	artifact := Artifact{Path: "input", SHA256: strings.Repeat("a", 64)}
	return Manifest{
		SchemaVersion: SchemaVersion,
		CollectedAt:   time.Date(2026, time.August, 8, 12, 0, 0, 0, time.UTC),
		Repositories: Repositories{
			MachineLearning: Repository{Path: ".", Commit: "a"},
			SyntheticData:   Repository{Path: "synthetic", Commit: "b"},
			GameEngine:      Repository{Path: "game", Commit: "c"},
		},
		Dataset:             Dataset{Format: "WDIT", Version: "3", Files: []Artifact{artifact}},
		Vocabulary:          []Artifact{artifact},
		Splits:              Splits{Training: []Artifact{artifact}, Validation: []Artifact{artifact}, Test: []Artifact{artifact}},
		ModelParameterCount: 1,
		Runtime: RuntimeMetadata{
			GoVersion:    "go1.test",
			GOOS:         "linux",
			GOARCH:       "amd64",
			GoMLXVersion: "v0.28.0",
			Backend:      "xla:cuda",
			GPUDetails:   map[string]string{"name": "RTX 5070 Ti"},
			CUDADetails:  map[string]string{"nvcc": "available"},
			PJRTDetails:  map[string]string{"plugin": "cuda"},
			GoMLXDetails: map[string]string{"compute_backend": "xla:cuda"},
		},
		Seed:            1,
		EffectiveConfig: json.RawMessage(`{"seed":1}`),
	}
}

func makeGitRepository(t *testing.T, root, name string) string {
	t.Helper()
	directory := filepath.Join(root, name)
	if err := os.MkdirAll(directory, 0o755); err != nil {
		t.Fatalf("create repository directory: %v", err)
	}
	runGit(t, directory, "init")
	writeFile(t, filepath.Join(directory, "README"), name)
	runGit(t, directory, "add", "README")
	runGit(t, directory, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "initial")
	return directory
}

func runGit(t *testing.T, directory string, arguments ...string) {
	t.Helper()
	command := exec.Command("git", append([]string{"-C", directory}, arguments...)...)
	if output, err := command.CombinedOutput(); err != nil {
		t.Fatalf("git %s: %v\n%s", strings.Join(arguments, " "), err, output)
	}
}

func writeFile(t *testing.T, path, contents string) string {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatalf("create parent directory for %q: %v", path, err)
	}
	if err := os.WriteFile(path, []byte(contents), 0o644); err != nil {
		t.Fatalf("write %q: %v", path, err)
	}
	return path
}

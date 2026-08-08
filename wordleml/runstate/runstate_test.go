package runstate

import (
	"errors"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/model/checkpoint"
)

func TestCreateMakesCompleteSelfContainedLayout(t *testing.T) {
	root := filepath.Join(t.TempDir(), "runs")
	layout, err := Create(root, "mini-proof-001")
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	if layout.Root != root {
		t.Fatalf("layout root = %q, want %q", layout.Root, root)
	}
	for _, path := range []string{layout.Dir, layout.EventsDir, layout.CheckpointsDir, layout.LatestCheckpointDir, layout.BestCheckpointDir, layout.InitialCheckpointDir, layout.EvaluationsDir} {
		info, err := os.Stat(path)
		if err != nil {
			t.Errorf("required directory %q is missing: %v", path, err)
			continue
		}
		if !info.IsDir() {
			t.Errorf("required directory %q is not a directory", path)
		}
	}
	for _, path := range []string{layout.ConfigPath, layout.MetadataPath, layout.StatePath, layout.FinalMetricsPath, layout.ValidationGamesPath, layout.TrainingLogPath} {
		if filepath.Dir(path) != layout.Dir {
			t.Errorf("artifact %q is not directly inside run directory %q", path, layout.Dir)
		}
	}
	if _, err := Create(root, "mini-proof-001"); err == nil {
		t.Fatal("Create reused an existing run ID")
	}
	opened, err := Open(root, "mini-proof-001")
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	if opened != layout {
		t.Fatalf("Open layout = %#v, want %#v", opened, layout)
	}
}

func TestEvaluationArtifactsAreAtomicAndNamed(t *testing.T) {
	layout, err := Create(t.TempDir(), "evaluation-proof")
	if err != nil {
		t.Fatal(err)
	}
	if err := layout.WriteEvaluationJSON("best", "games100", map[string]int{"games": 100}); err != nil {
		t.Fatal(err)
	}
	if err := layout.WriteEvaluationGamesJSONL("best", "games100", []byte("{\"solution\":\"ABCDE\"}\n")); err != nil {
		t.Fatal(err)
	}
	if err := layout.WriteValidationGamesJSONL([]byte("{\"solution\":\"ABCDE\"}\n")); err != nil {
		t.Fatal(err)
	}
	for _, path := range []string{
		filepath.Join(layout.EvaluationsDir, "best-games100.json"),
		filepath.Join(layout.EvaluationsDir, "best-games100.jsonl"),
		layout.ValidationGamesPath,
	} {
		contents, err := os.ReadFile(path)
		if err != nil || len(contents) == 0 {
			t.Fatalf("artifact %q = %q, %v", path, contents, err)
		}
	}
	if !strings.HasSuffix(string(mustRead(t, filepath.Join(layout.EvaluationsDir, "best-games100.json"))), "\n") {
		t.Fatal("evaluation JSON lacks trailing newline")
	}
	for _, name := range []string{"", "../bad", "UPPER"} {
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("unsafe evaluation name %q did not panic", name)
				}
			}()
			_ = layout.evaluationPath(name, "games100", ".json")
		}()
	}
}

func TestEvaluationArtifactsAreImmutableButIdempotent(t *testing.T) {
	layout, err := Create(t.TempDir(), "immutable-evaluation")
	if err != nil {
		t.Fatal(err)
	}
	value := map[string]any{"stage": "full", "score": 1}
	if err := layout.WriteEvaluationJSON("best", "games100", value); err != nil {
		t.Fatal(err)
	}
	if err := layout.WriteEvaluationJSON("best", "games100", value); err != nil {
		t.Fatalf("identical evaluation JSON retry: %v", err)
	}
	if err := layout.WriteEvaluationJSON("best", "games100", map[string]any{"stage": "full", "score": 2}); err == nil {
		t.Fatal("different evaluation JSON replaced immutable artifact")
	}
	jsonl := []byte("{\"solution\":\"ABCDE\"}\n")
	if err := layout.WriteEvaluationGamesJSONL("best", "games100", jsonl); err != nil {
		t.Fatal(err)
	}
	if err := layout.WriteEvaluationGamesJSONL("best", "games100", jsonl); err != nil {
		t.Fatalf("identical JSONL retry: %v", err)
	}
	if err := layout.WriteEvaluationGamesJSONL("best", "games100", []byte("{\"solution\":\"OTHER\"}\n")); err == nil {
		t.Fatal("different JSONL replaced immutable artifact")
	}
	if err := layout.WriteValidationGamesJSONL([]byte("first\n")); err != nil {
		t.Fatal(err)
	}
	if err := layout.WriteValidationGamesJSONL([]byte("second\n")); err != nil {
		t.Fatal(err)
	}
	if got := string(mustRead(t, layout.ValidationGamesPath)); got != "second\n" {
		t.Fatalf("canonical games = %q, want replacement", got)
	}
}

func mustRead(t *testing.T, path string) []byte {
	t.Helper()
	contents, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return contents
}

func TestValidateRunIDRejectsUnsafeNames(t *testing.T) {
	for _, runID := range []string{"", ".", "..", ".hidden", "with/slash", "with\\backslash", "with space", "bad:colon", strings.Repeat("a", maximumRunIDLength+1)} {
		if err := ValidateRunID(runID); err == nil {
			t.Errorf("ValidateRunID(%q) succeeded", runID)
		}
	}
	for _, runID := range []string{"proof-01", "mini_20260808", "run.2"} {
		if err := ValidateRunID(runID); err != nil {
			t.Errorf("ValidateRunID(%q): %v", runID, err)
		}
	}
}

func TestStateRoundTripIsValidatedAndReplaced(t *testing.T) {
	layout, err := Create(t.TempDir(), "resume-proof")
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	if _, err := layout.LoadStateMirror(); !errors.Is(err, ErrStateNotFound) {
		t.Fatalf("LoadStateMirror before first save = %v, want ErrStateNotFound", err)
	}
	first := State{
		GlobalUpdate:   500,
		ShuffleSeed:    71,
		DatasetEpoch:   1,
		ShuffledCursor: 11392,
		BestValidation: &BestValidation{Value: 1.25, Update: 500},
		NextRecordIDs:  []int{42, 17, 8},
	}
	if err := layout.WriteStateMirror(first); err != nil {
		t.Fatalf("WriteStateMirror first: %v", err)
	}
	second := State{GlobalUpdate: 600, ShuffleSeed: 71, DatasetEpoch: 1, ShuffledCursor: 24192}
	if err := layout.WriteStateMirror(second); err != nil {
		t.Fatalf("WriteStateMirror replacement: %v", err)
	}
	got, err := layout.LoadStateMirror()
	if err != nil {
		t.Fatalf("LoadState: %v", err)
	}
	if got.GlobalUpdate != second.GlobalUpdate || got.DatasetEpoch != second.DatasetEpoch || got.ShuffledCursor != second.ShuffledCursor || got.BestValidation != nil || len(got.NextRecordIDs) != 0 {
		t.Fatalf("loaded state = %#v, want %#v", got, second)
	}
	leftovers, err := filepath.Glob(filepath.Join(layout.Dir, "."+stateFilename+".tmp-*"))
	if err != nil {
		t.Fatalf("Glob temporary state files: %v", err)
	}
	if len(leftovers) != 0 {
		t.Fatalf("temporary state files left behind: %v", leftovers)
	}
}

func TestCheckpointStateIsSavedWithModelStore(t *testing.T) {
	want := State{
		GlobalUpdate:   0,
		ShuffleSeed:    71,
		DatasetEpoch:   2,
		ShuffledCursor: 123,
		NextRecordIDs:  []int{8, 5, 3},
	}
	store := model.NewStore()
	if err := SaveCheckpointState(store, want); err != nil {
		t.Fatalf("SaveCheckpointState: %v", err)
	}
	handler, err := checkpoint.Build(store).Dir(t.TempDir()).Keep(1).Done()
	if err != nil {
		t.Fatalf("build checkpoint: %v", err)
	}
	if err := handler.Save(); err != nil {
		t.Fatalf("save checkpoint: %v", err)
	}

	restoredStore := model.NewStore()
	if _, err := checkpoint.Build(restoredStore).Dir(handler.Dir()).Keep(1).Done(); err != nil {
		t.Fatalf("restore checkpoint: %v", err)
	}
	got, err := LoadCheckpointState(restoredStore)
	if err != nil {
		t.Fatalf("LoadCheckpointState: %v", err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("checkpoint state = %#v, want %#v", got, want)
	}
}

func TestStateValidation(t *testing.T) {
	for name, state := range map[string]State{
		"negative update":         {GlobalUpdate: -1, ShuffleSeed: 1},
		"zero shuffle seed":       {},
		"negative epoch":          {ShuffleSeed: 1, DatasetEpoch: -1},
		"negative cursor":         {ShuffleSeed: 1, ShuffledCursor: -1},
		"best update after state": {GlobalUpdate: 4, ShuffleSeed: 1, BestValidation: &BestValidation{Value: 1, Update: 5}},
		"non-finite best value":   {ShuffleSeed: 1, BestValidation: &BestValidation{Value: math.Inf(1)}},
		"negative record ID":      {ShuffleSeed: 1, NextRecordIDs: []int{0, -1}},
	} {
		t.Run(name, func(t *testing.T) {
			if err := state.Validate(); err == nil {
				t.Fatalf("Validate(%#v) succeeded", state)
			}
		})
	}
}

func TestManifestAndMetadataWrites(t *testing.T) {
	layout, err := Create(t.TempDir(), "metadata-proof")
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	if err := layout.WriteConfig(struct {
		BatchSize int `json:"batch_size"`
	}{BatchSize: 128}); err != nil {
		t.Fatalf("WriteConfig: %v", err)
	}
	if err := layout.WriteMetadata(map[string]string{"dataset_format": "WDIT v3"}); err != nil {
		t.Fatalf("WriteMetadata: %v", err)
	}
	for _, path := range []string{layout.ConfigPath, layout.MetadataPath} {
		contents, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("read %q: %v", path, err)
		}
		if !strings.HasSuffix(string(contents), "\n") {
			t.Errorf("JSON file %q has no trailing newline", path)
		}
	}

}

package runstate

import (
	"errors"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
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
	for _, path := range []string{layout.Dir, layout.EventsDir, layout.CheckpointsDir, layout.LatestCheckpointDir, layout.BestCheckpointDir} {
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

func TestManifestAndMetadataHelpers(t *testing.T) {
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

	hashPath := filepath.Join(layout.Dir, "hash-input")
	if err := os.WriteFile(hashPath, []byte("wordle"), 0o644); err != nil {
		t.Fatalf("WriteFile hash input: %v", err)
	}
	gotHash, err := FileSHA256(hashPath)
	if err != nil {
		t.Fatalf("FileSHA256: %v", err)
	}
	const wantHash = "ebe054f08821294feee7bc442014fdd38b4836d83781d8ba99d38eb50d0c9d85"
	if gotHash != wantHash {
		t.Fatalf("FileSHA256 = %q, want %q", gotHash, wantHash)
	}

	metadata := CurrentRuntimeMetadata()
	if metadata.GoVersion != runtime.Version() || metadata.GOOS != runtime.GOOS || metadata.GOARCH != runtime.GOARCH {
		t.Fatalf("runtime metadata = %#v", metadata)
	}
}

package proofrun

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/gomlx/gomlx/ml/model/checkpoint"
)

type checkpointSourceStub struct {
	dir         string
	checkpoints []string
	err         error
}

func (s checkpointSourceStub) ListCheckpoints() ([]string, error) { return s.checkpoints, s.err }
func (s checkpointSourceStub) Dir() string                        { return s.dir }

func TestCopyLatestCheckpointAtomicallyCopiesNewestAndPublishesJSONLast(t *testing.T) {
	sourceDir := t.TempDir()
	destinationDir := t.TempDir()
	writeCheckpointArtifact(t, sourceDir, "checkpoint-old", checkpoint.BinDataSuffix, "old binary")
	writeCheckpointArtifact(t, sourceDir, "checkpoint-old", checkpoint.JsonNameSuffix, "old metadata")
	writeCheckpointArtifact(t, sourceDir, "checkpoint-new", checkpoint.BinDataSuffix, "new binary")
	writeCheckpointArtifact(t, sourceDir, "checkpoint-new", checkpoint.JsonNameSuffix, "new metadata")

	// This models an interrupted earlier attempt after publishing its BIN. The
	// retry must replace it, and must not publish JSON until that replacement is
	// complete and durable.
	writeCheckpointArtifact(t, destinationDir, "checkpoint-new", checkpoint.BinDataSuffix, "partial binary")
	var jsonPublished bool
	ops := osCheckpointFileOps()
	baseRename := ops.rename
	ops.rename = func(oldPath, newPath string) error {
		if strings.HasSuffix(newPath, checkpoint.JsonNameSuffix) {
			jsonPublished = true
			contents, err := os.ReadFile(filepath.Join(destinationDir, "checkpoint-new"+checkpoint.BinDataSuffix))
			if err != nil {
				t.Fatalf("read BIN while publishing JSON: %v", err)
			}
			if got, want := string(contents), "new binary"; got != want {
				t.Fatalf("BIN visible with JSON = %q, want %q", got, want)
			}
			if _, err := os.Stat(newPath); !errors.Is(err, os.ErrNotExist) {
				t.Fatalf("JSON existed before its publication rename: %v", err)
			}
		}
		return baseRename(oldPath, newPath)
	}

	if err := copyLatestCheckpointWithFileOps(checkpointSourceStub{
		dir:         sourceDir,
		checkpoints: []string{"checkpoint-old", "checkpoint-new"},
	}, destinationDir, ops); err != nil {
		t.Fatalf("copyLatestCheckpointWithFileOps: %v", err)
	}
	if !jsonPublished {
		t.Fatal("JSON publication rename was not observed")
	}
	assertFileContents(t, filepath.Join(destinationDir, "checkpoint-new"+checkpoint.BinDataSuffix), "new binary")
	assertFileContents(t, filepath.Join(destinationDir, "checkpoint-new"+checkpoint.JsonNameSuffix), "new metadata")
	if _, err := os.Stat(filepath.Join(destinationDir, "checkpoint-old"+checkpoint.JsonNameSuffix)); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("older checkpoint was copied: %v", err)
	}
}

func TestCopyLatestCheckpointAtomicallyRetryCleansPartialTemporaryFile(t *testing.T) {
	sourceDir := t.TempDir()
	destinationDir := t.TempDir()
	base := "checkpoint-retry"
	if err := os.Mkdir(filepath.Join(sourceDir, base+checkpoint.BinDataSuffix), 0o755); err != nil {
		t.Fatalf("make unreadable BIN source: %v", err)
	}
	writeCheckpointArtifact(t, sourceDir, base, checkpoint.JsonNameSuffix, "metadata")
	source := checkpointSourceStub{dir: sourceDir, checkpoints: []string{base}}

	if err := copyLatestCheckpointAtomically(source, destinationDir); err == nil {
		t.Fatal("copy with invalid BIN source succeeded")
	}
	if _, err := os.Stat(filepath.Join(destinationDir, base+checkpoint.JsonNameSuffix)); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("JSON was published after BIN copy failure: %v", err)
	}
	entries, err := os.ReadDir(destinationDir)
	if err != nil {
		t.Fatalf("read destination: %v", err)
	}
	if len(entries) != 0 {
		t.Fatalf("partial temporary artifact was retained: %v", entries)
	}

	if err := os.Remove(filepath.Join(sourceDir, base+checkpoint.BinDataSuffix)); err != nil {
		t.Fatalf("remove invalid BIN source: %v", err)
	}
	writeCheckpointArtifact(t, sourceDir, base, checkpoint.BinDataSuffix, "recovered binary")
	if err := copyLatestCheckpointAtomically(source, destinationDir); err != nil {
		t.Fatalf("retry checkpoint copy: %v", err)
	}
	assertFileContents(t, filepath.Join(destinationDir, base+checkpoint.BinDataSuffix), "recovered binary")
	assertFileContents(t, filepath.Join(destinationDir, base+checkpoint.JsonNameSuffix), "metadata")
}

func TestCopyLatestCheckpointAtomicallyRejectsEmptyAndUnsafeSources(t *testing.T) {
	destinationDir := t.TempDir()
	if err := copyLatestCheckpointAtomically(checkpointSourceStub{dir: t.TempDir()}, destinationDir); err == nil {
		t.Fatal("empty source was accepted")
	}
	if err := copyLatestCheckpointAtomically(checkpointSourceStub{dir: t.TempDir(), checkpoints: []string{"../checkpoint"}}, destinationDir); err == nil {
		t.Fatal("unsafe checkpoint base name was accepted")
	}
}

func writeCheckpointArtifact(t *testing.T, dir, base, suffix, contents string) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(dir, base+suffix), []byte(contents), 0o644); err != nil {
		t.Fatalf("write checkpoint artifact: %v", err)
	}
}

func assertFileContents(t *testing.T, path, want string) {
	t.Helper()
	contents, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %q: %v", path, err)
	}
	if got := string(contents); got != want {
		t.Fatalf("contents of %q = %q, want %q", path, got, want)
	}
}

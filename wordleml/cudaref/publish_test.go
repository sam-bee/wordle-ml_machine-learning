package cudaref

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
)

func TestPublishDirectoryPublishesOnlyCompleteOutput(t *testing.T) {
	destination := filepath.Join(t.TempDir(), "best")
	if err := PublishDirectory(destination, func(staging string) error {
		return os.WriteFile(filepath.Join(staging, "complete"), []byte("yes"), 0o644)
	}); err != nil {
		t.Fatalf("PublishDirectory: %v", err)
	}
	contents, err := os.ReadFile(filepath.Join(destination, "complete"))
	if err != nil || string(contents) != "yes" {
		t.Fatalf("published file = %q, %v", contents, err)
	}
	if err := PublishDirectory(destination, func(string) error { return nil }); err == nil {
		t.Fatal("existing destination was overwritten")
	}
}

func TestPublishDirectoryRemovesFailedStagingDirectory(t *testing.T) {
	parent := t.TempDir()
	destination := filepath.Join(parent, "best")
	if err := PublishDirectory(destination, func(staging string) error {
		if err := os.WriteFile(filepath.Join(staging, "partial"), []byte("no"), 0o644); err != nil {
			return err
		}
		return errors.New("stop")
	}); err == nil {
		t.Fatal("failing export population succeeded")
	}
	if _, err := os.Stat(destination); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("destination exists after failed population: %v", err)
	}
	entries, err := os.ReadDir(parent)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 0 {
		t.Fatalf("staging output leaked: %v", entries)
	}
}

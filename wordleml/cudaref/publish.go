package cudaref

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
)

// PublishDirectory builds one export in a private sibling directory, then
// publishes it with a single rename.  It refuses to replace a prior export:
// a repeat run must use a distinct output path or explicitly remove the old
// generated artifact outside this API.
func PublishDirectory(destination string, populate func(string) error) (err error) {
	if destination == "" {
		return errors.New("export destination is required")
	}
	if populate == nil {
		return errors.New("export population callback is required")
	}
	parent := filepath.Dir(destination)
	base := filepath.Base(destination)
	if base == "." || base == string(filepath.Separator) {
		return fmt.Errorf("unsafe export destination %q", destination)
	}
	if err := os.MkdirAll(parent, 0o755); err != nil {
		return fmt.Errorf("create export parent %q: %w", parent, err)
	}
	if _, err := os.Lstat(destination); err == nil {
		return fmt.Errorf("export destination %q already exists", destination)
	} else if !errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("inspect export destination %q: %w", destination, err)
	}
	staging, err := os.MkdirTemp(parent, "."+base+".tmp-")
	if err != nil {
		return fmt.Errorf("create export staging directory: %w", err)
	}
	committed := false
	defer func() {
		if !committed {
			_ = os.RemoveAll(staging)
		}
	}()
	if err := populate(staging); err != nil {
		return err
	}
	if err := syncDirectory(staging); err != nil {
		return fmt.Errorf("sync export staging directory: %w", err)
	}
	if err := os.Rename(staging, destination); err != nil {
		return fmt.Errorf("publish export directory: %w", err)
	}
	committed = true
	if err := syncDirectory(parent); err != nil {
		return fmt.Errorf("sync export parent directory: %w", err)
	}
	return nil
}

func syncDirectory(path string) (err error) {
	directory, err := os.Open(path)
	if err != nil {
		return err
	}
	defer func() {
		if closeErr := directory.Close(); closeErr != nil && err == nil {
			err = closeErr
		}
	}()
	return directory.Sync()
}

package cudacheck

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// WriteJSON writes an indented report atomically, creating the report parent
// directory when necessary.
func WriteJSON(path string, value any) error {
	if path == "" {
		return fmt.Errorf("report path is required")
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("create report directory: %w", err)
	}
	contents, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return fmt.Errorf("encode report: %w", err)
	}
	contents = append(contents, '\n')
	temporary, err := os.CreateTemp(filepath.Dir(path), ".report-*.tmp")
	if err != nil {
		return fmt.Errorf("create temporary report: %w", err)
	}
	temporaryPath := temporary.Name()
	defer func() { _ = os.Remove(temporaryPath) }()
	if _, err := temporary.Write(contents); err != nil {
		_ = temporary.Close()
		return fmt.Errorf("write temporary report: %w", err)
	}
	if err := temporary.Chmod(0o644); err != nil {
		_ = temporary.Close()
		return fmt.Errorf("set temporary report permissions: %w", err)
	}
	if err := temporary.Close(); err != nil {
		return fmt.Errorf("close temporary report: %w", err)
	}
	if err := os.Rename(temporaryPath, path); err != nil {
		return fmt.Errorf("publish report: %w", err)
	}
	return nil
}

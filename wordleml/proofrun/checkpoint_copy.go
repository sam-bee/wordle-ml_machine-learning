package proofrun

import (
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/gomlx/gomlx/ml/model/checkpoint"
)

// checkpointSource is the small part of a GoMLX checkpoint handler needed to
// copy one immutable checkpoint generation.
type checkpointSource interface {
	ListCheckpoints() ([]string, error)
	Dir() string
}

// copyLatestCheckpointAtomically copies the newest generation from source to
// destinationDir. It commits the binary data before the metadata, so a GoMLX
// loader can never discover the new JSON before its complete BIN is present.
//
// A previous interrupted copy can leave a temporary file or an orphan BIN.
// Temporary files are ignored and an orphan BIN is replaced on retry before
// the JSON is published.
func copyLatestCheckpointAtomically(source checkpointSource, destinationDir string) error {
	return copyLatestCheckpointWithFileOps(source, destinationDir, osCheckpointFileOps())
}

type checkpointFileOps struct {
	open       func(string) (*os.File, error)
	createTemp func(string, string) (*os.File, error)
	rename     func(string, string) error
	openDir    func(string) (*os.File, error)
}

func osCheckpointFileOps() checkpointFileOps {
	return checkpointFileOps{
		open:       os.Open,
		createTemp: os.CreateTemp,
		rename:     os.Rename,
		openDir:    os.Open,
	}
}

func copyLatestCheckpointWithFileOps(source checkpointSource, destinationDir string, ops checkpointFileOps) error {
	checkpoints, err := source.ListCheckpoints()
	if err != nil {
		return fmt.Errorf("list source checkpoints: %w", err)
	}
	if len(checkpoints) == 0 {
		return errors.New("checkpoint source has no checkpoint to copy")
	}
	base := checkpoints[len(checkpoints)-1]
	if base != filepath.Base(base) || base == "." || base == string(filepath.Separator) || strings.Contains(base, string(filepath.Separator)) {
		return fmt.Errorf("unsafe checkpoint base name %q", base)
	}

	for _, suffix := range []string{checkpoint.BinDataSuffix, checkpoint.JsonNameSuffix} {
		sourcePath := filepath.Join(source.Dir(), base+suffix)
		destinationPath := filepath.Join(destinationDir, base+suffix)
		if err := copyFileAtomically(sourcePath, destinationPath, ops); err != nil {
			return fmt.Errorf("copy checkpoint artifact %q: %w", base+suffix, err)
		}
	}
	return nil
}

// copyFileAtomically makes the fully written file visible with one rename,
// then syncs its directory so that rename is durable. destination must be in
// an existing directory; proof-run layouts create checkpoint directories
// before this function is called.
func copyFileAtomically(sourcePath, destinationPath string, ops checkpointFileOps) (err error) {
	input, err := ops.open(sourcePath)
	if err != nil {
		return fmt.Errorf("open source %q: %w", sourcePath, err)
	}
	defer func() {
		if closeErr := input.Close(); closeErr != nil && err == nil {
			err = fmt.Errorf("close source %q: %w", sourcePath, closeErr)
		}
	}()

	destinationDir := filepath.Dir(destinationPath)
	temp, err := ops.createTemp(destinationDir, "."+filepath.Base(destinationPath)+".tmp-*")
	if err != nil {
		return fmt.Errorf("create temporary destination for %q: %w", destinationPath, err)
	}
	tempPath := temp.Name()
	committed := false
	defer func() {
		if !committed {
			_ = temp.Close()
			_ = os.Remove(tempPath)
		}
	}()

	if _, err := io.Copy(temp, input); err != nil {
		return fmt.Errorf("copy data to temporary destination %q: %w", tempPath, err)
	}
	if err := temp.Sync(); err != nil {
		return fmt.Errorf("sync temporary destination %q: %w", tempPath, err)
	}
	if err := temp.Close(); err != nil {
		return fmt.Errorf("close temporary destination %q: %w", tempPath, err)
	}
	if err := ops.rename(tempPath, destinationPath); err != nil {
		return fmt.Errorf("rename temporary destination %q to %q: %w", tempPath, destinationPath, err)
	}
	committed = true
	if err := syncDirectory(destinationDir, ops); err != nil {
		return fmt.Errorf("sync destination directory %q: %w", destinationDir, err)
	}
	return nil
}

func syncDirectory(dir string, ops checkpointFileOps) (err error) {
	directory, err := ops.openDir(dir)
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

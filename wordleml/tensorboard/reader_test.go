package tensorboard_test

import (
	"encoding/binary"
	"errors"
	"hash/crc32"
	"os"
	"path/filepath"
	"testing"

	"github.com/sam-bee/wordle-ml_machine-learning/tensorboard"
)

func TestInspectDirReadsWriterScalarAndHistogramRecords(t *testing.T) {
	logDir := t.TempDir()
	first, err := tensorboard.New(logDir)
	if err != nil {
		t.Fatal(err)
	}
	if err := first.WriteScalars(500, tensorboard.Scalar{Tag: "train/loss", Value: 0.5}); err != nil {
		t.Fatal(err)
	}
	if err := first.WriteHistograms(500, tensorboard.Histogram{Tag: "model/logits", Values: []float64{-1, 0, 1}}); err != nil {
		t.Fatal(err)
	}
	if err := first.Close(); err != nil {
		t.Fatal(err)
	}
	second, err := tensorboard.New(logDir)
	if err != nil {
		t.Fatal(err)
	}
	if err := second.WriteScalars(510, tensorboard.Scalar{Tag: "train/loss", Value: 0.4}); err != nil {
		t.Fatal(err)
	}
	if err := second.Close(); err != nil {
		t.Fatal(err)
	}

	inspection, err := tensorboard.InspectDir(logDir)
	if err != nil {
		t.Fatalf("InspectDir() error = %v", err)
	}
	if got, want := inspection.Scalars, 2; len(got) != want {
		t.Fatalf("scalar records = %#v, want %d records", got, want)
	}
	scalarsByStep := make(map[int64]tensorboard.ScalarRecord, len(inspection.Scalars))
	for _, scalar := range inspection.Scalars {
		scalarsByStep[scalar.Step] = scalar
	}
	for step, value := range map[int64]float32{500: 0.5, 510: 0.4} {
		record, found := scalarsByStep[step]
		if !found || record.Tag != "train/loss" || record.Value != value || record.SourceFile == "" {
			t.Errorf("scalar record at step %d = %#v, want train/loss=%v with a source file", step, record, value)
		}
	}
	if got, want := inspection.Histograms, []tensorboard.HistogramRecord{{Tag: "model/logits", Step: 500, Count: 3}}; len(got) != len(want) || got[0].Tag != want[0].Tag || got[0].Step != want[0].Step || got[0].Count != want[0].Count || got[0].SourceFile == "" {
		t.Errorf("histogram records = %#v, want %#v with a source file", got, want)
	}
}

func TestInspectDirRejectsCorruptTFRecord(t *testing.T) {
	logDir := t.TempDir()
	writer, err := tensorboard.New(logDir)
	if err != nil {
		t.Fatal(err)
	}
	if err := writer.WriteScalars(10, tensorboard.Scalar{Tag: "train/loss", Value: 1}); err != nil {
		t.Fatal(err)
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	paths, err := filepath.Glob(filepath.Join(logDir, "events.out.tfevents.*"))
	if err != nil || len(paths) != 1 {
		t.Fatalf("event paths = %v, %v", paths, err)
	}
	contents, err := os.ReadFile(paths[0])
	if err != nil {
		t.Fatal(err)
	}
	contents[len(contents)-1] ^= 0xff
	if err := os.WriteFile(paths[0], contents, 0o644); err != nil {
		t.Fatal(err)
	}
	_, err = tensorboard.InspectDir(logDir)
	if !errors.Is(err, tensorboard.ErrInvalidRecord) {
		t.Fatalf("InspectDir() error = %v, want ErrInvalidRecord", err)
	}
}

func TestInspectDirRejectsMalformedEventPayload(t *testing.T) {
	logDir := t.TempDir()
	path := filepath.Join(logDir, "events.out.tfevents.malformed")
	if err := os.WriteFile(path, tfRecord([]byte{0}), 0o644); err != nil {
		t.Fatal(err)
	}
	_, err := tensorboard.InspectDir(logDir)
	if !errors.Is(err, tensorboard.ErrMalformedEvent) {
		t.Fatalf("InspectDir() error = %v, want ErrMalformedEvent", err)
	}
}

func TestInspectDirRequiresEventFiles(t *testing.T) {
	_, err := tensorboard.InspectDir(t.TempDir())
	if !errors.Is(err, tensorboard.ErrNoEventFiles) {
		t.Fatalf("InspectDir() error = %v, want ErrNoEventFiles", err)
	}
}

func tfRecord(payload []byte) []byte {
	contents := make([]byte, 8+4+len(payload)+4)
	binary.LittleEndian.PutUint64(contents[:8], uint64(len(payload)))
	binary.LittleEndian.PutUint32(contents[8:12], testMaskedCRC(contents[:8]))
	copy(contents[12:], payload)
	binary.LittleEndian.PutUint32(contents[12+len(payload):], testMaskedCRC(payload))
	return contents
}

func testMaskedCRC(data []byte) uint32 {
	checksum := crc32.Checksum(data, crc32.MakeTable(crc32.Castagnoli))
	return (checksum>>15 | checksum<<17) + 0xa282ead8
}

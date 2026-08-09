package tensorboard_test

import (
	"encoding/binary"
	"errors"
	"hash/crc32"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/sam-bee/wordle-ml_machine-learning/tensorboard"
)

func TestWriterWritesTensorBoardScalarEvents(t *testing.T) {
	logDir := t.TempDir()
	w, err := tensorboard.New(logDir)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	t.Cleanup(func() {
		if err := w.Close(); err != nil && !errors.Is(err, tensorboard.ErrClosed) {
			t.Errorf("Close() error = %v", err)
		}
	})

	paths, err := filepath.Glob(filepath.Join(logDir, "events.out.tfevents.*"))
	if err != nil {
		t.Fatalf("Glob() error = %v", err)
	}
	if len(paths) != 1 {
		t.Fatalf("event files = %v, want exactly one", paths)
	}
	if filepath.Base(paths[0]) == "events.out.tfevents." {
		t.Fatalf("event filename %q is not unique", paths[0])
	}

	beforeClose := readRecords(t, paths[0])
	if len(beforeClose) != 1 {
		t.Fatalf("records before scalar write = %d, want 1", len(beforeClose))
	}
	fileVersion := parseEvent(t, beforeClose[0])
	if fileVersion.wallTime <= 0 {
		t.Fatalf("file version wall time = %v, want positive", fileVersion.wallTime)
	}
	if version := fileVersion.fileVersion; version != "brain.Event:2" {
		t.Fatalf("file version = %q, want brain.Event:2", version)
	}

	want := []tensorboard.Scalar{
		{Tag: "training/loss", Value: 0.125},
		{Tag: "validation/accuracy", Value: 0.875},
	}
	if err := w.WriteScalars(17, want...); err != nil {
		t.Fatalf("WriteScalars() error = %v", err)
	}
	if err := w.Flush(); err != nil {
		t.Fatalf("Flush() error = %v", err)
	}

	records := readRecords(t, paths[0])
	if len(records) != 2 {
		t.Fatalf("records visible before Close() = %d, want 2", len(records))
	}
	event := parseEvent(t, records[1])
	if event.step != 17 {
		t.Fatalf("step = %d, want 17", event.step)
	}
	if event.wallTime <= 0 {
		t.Fatalf("wall time = %v, want positive", event.wallTime)
	}
	if len(event.scalars) != len(want) {
		t.Fatalf("scalars = %d, want %d", len(event.scalars), len(want))
	}
	for i, scalar := range want {
		if got := event.scalars[i]; got.tag != scalar.Tag || math.Float32bits(got.value) != math.Float32bits(scalar.Value) {
			t.Errorf("scalar %d = (%q, %v), want (%q, %v)", i, got.tag, got.value, scalar.Tag, scalar.Value)
		}
	}
}

func TestWriterRejectsOperationsAfterClose(t *testing.T) {
	w, err := tensorboard.New(t.TempDir())
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	if err := w.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
	for name, operation := range map[string]func() error{
		"WriteScalars": func() error { return w.WriteScalars(1, tensorboard.Scalar{Tag: "loss", Value: 1}) },
		"WriteHistograms": func() error {
			return w.WriteHistograms(1, tensorboard.Histogram{Tag: "logits", Values: []float64{1}})
		},
		"Flush": w.Flush,
		"Close": w.Close,
	} {
		if err := operation(); !errors.Is(err, tensorboard.ErrClosed) {
			t.Errorf("%s() error = %v, want ErrClosed", name, err)
		}
	}
}

func TestWriterWritesTensorBoardHistogramEvents(t *testing.T) {
	logDir := t.TempDir()
	w, err := tensorboard.New(logDir)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	t.Cleanup(func() {
		if err := w.Close(); err != nil && !errors.Is(err, tensorboard.ErrClosed) {
			t.Errorf("Close() error = %v", err)
		}
	})

	if err := w.WriteHistograms(23,
		tensorboard.Histogram{Tag: "model/logits", Values: []float64{-2, 0, 0, 3}},
		tensorboard.Histogram{Tag: "empty", Values: []float64{math.NaN(), math.Inf(1), math.Inf(-1)}},
	); err != nil {
		t.Fatalf("WriteHistograms() error = %v", err)
	}
	if err := w.Flush(); err != nil {
		t.Fatalf("Flush() error = %v", err)
	}

	paths, err := filepath.Glob(filepath.Join(logDir, "events.out.tfevents.*"))
	if err != nil || len(paths) != 1 {
		t.Fatalf("event files = %v, %v; want one file", paths, err)
	}
	records := readRecords(t, paths[0])
	if len(records) != 2 {
		t.Fatalf("records = %d, want 2", len(records))
	}
	event := parseEvent(t, records[1])
	if event.step != 23 {
		t.Fatalf("step = %d, want 23", event.step)
	}
	if len(event.histograms) != 2 {
		t.Fatalf("histograms = %d, want 2", len(event.histograms))
	}

	got := event.histograms[0]
	if got.tag != "model/logits" {
		t.Errorf("histogram tag = %q, want model/logits", got.tag)
	}
	if got.minimum != -2 || got.maximum != 3 || got.count != 4 || got.sum != 1 || got.sumSquares != 13 {
		t.Errorf("histogram fields = min=%v max=%v count=%v sum=%v sumSquares=%v, want -2, 3, 4, 1, 13", got.minimum, got.maximum, got.count, got.sum, got.sumSquares)
	}
	if want := []float64{-2, 0, 3}; !equalFloat64s(got.bucketLimits, want) {
		t.Errorf("bucket limits = %v, want %v", got.bucketLimits, want)
	}
	if want := []float64{1, 2, 1}; !equalFloat64s(got.buckets, want) {
		t.Errorf("buckets = %v, want %v", got.buckets, want)
	}

	empty := event.histograms[1]
	if empty.tag != "empty" || empty.minimum != 0 || empty.maximum != 0 || empty.count != 0 || empty.sum != 0 || empty.sumSquares != 0 {
		t.Errorf("empty histogram = %+v, want a zero-sized histogram", empty)
	}
	if len(empty.bucketLimits) != 0 || len(empty.buckets) != 0 {
		t.Errorf("empty buckets = limits %v counts %v, want none", empty.bucketLimits, empty.buckets)
	}
}

type parsedEvent struct {
	wallTime    float64
	step        int64
	fileVersion string
	scalars     []parsedScalar
	histograms  []parsedHistogram
}

type parsedScalar struct {
	tag   string
	value float32
}

type parsedHistogram struct {
	tag          string
	minimum      float64
	maximum      float64
	count        float64
	sum          float64
	sumSquares   float64
	bucketLimits []float64
	buckets      []float64
}

func readRecords(t *testing.T, path string) [][]byte {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile(%q) error = %v", path, err)
	}
	var records [][]byte
	for len(data) > 0 {
		if len(data) < 12 {
			t.Fatalf("truncated TFRecord header: %d bytes", len(data))
		}
		length := binary.LittleEndian.Uint64(data[:8])
		if got, want := binary.LittleEndian.Uint32(data[8:12]), maskedCRC(data[:8]); got != want {
			t.Fatalf("length CRC = %08x, want %08x", got, want)
		}
		if length > uint64(len(data)-16) {
			t.Fatalf("record length = %d, remaining bytes = %d", length, len(data)-16)
		}
		payloadEnd := 12 + int(length)
		payload := data[12:payloadEnd]
		if got, want := binary.LittleEndian.Uint32(data[payloadEnd:payloadEnd+4]), maskedCRC(payload); got != want {
			t.Fatalf("payload CRC = %08x, want %08x", got, want)
		}
		records = append(records, payload)
		data = data[payloadEnd+4:]
	}
	return records
}

func maskedCRC(data []byte) uint32 {
	checksum := crc32.Checksum(data, crc32.MakeTable(crc32.Castagnoli))
	return (checksum>>15 | checksum<<17) + 0xa282ead8
}

func parseEvent(t *testing.T, data []byte) parsedEvent {
	t.Helper()
	var event parsedEvent
	for len(data) > 0 {
		field, wireType, value, rest := parseField(t, data)
		switch field {
		case 1:
			if wireType != 1 || len(value) != 8 {
				t.Fatalf("event wall_time has wire type %d and length %d", wireType, len(value))
			}
			event.wallTime = math.Float64frombits(binary.LittleEndian.Uint64(value))
		case 2:
			if wireType != 0 {
				t.Fatalf("event step has wire type %d", wireType)
			}
			step, n := binary.Uvarint(value)
			if n <= 0 || n != len(value) {
				t.Fatalf("event step is not a valid varint")
			}
			event.step = int64(step)
		case 3:
			if wireType != 2 {
				t.Fatalf("event file_version has wire type %d", wireType)
			}
			event.fileVersion = string(value)
		case 5:
			if wireType != 2 {
				t.Fatalf("event summary has wire type %d", wireType)
			}
			summary := parseSummary(t, value)
			event.scalars = summary.scalars
			event.histograms = summary.histograms
		}
		data = rest
	}
	return event
}

type parsedSummary struct {
	scalars    []parsedScalar
	histograms []parsedHistogram
}

func parseSummary(t *testing.T, data []byte) parsedSummary {
	t.Helper()
	var summary parsedSummary
	for len(data) > 0 {
		field, wireType, value, rest := parseField(t, data)
		if field == 1 {
			if wireType != 2 {
				t.Fatalf("summary value has wire type %d", wireType)
			}
			scalar, histogram := parseSummaryValue(t, value)
			if scalar != nil {
				summary.scalars = append(summary.scalars, *scalar)
			}
			if histogram != nil {
				summary.histograms = append(summary.histograms, *histogram)
			}
		}
		data = rest
	}
	return summary
}

func parseSummaryValue(t *testing.T, data []byte) (*parsedScalar, *parsedHistogram) {
	t.Helper()
	var tag string
	var scalar *parsedScalar
	var histogram *parsedHistogram
	for len(data) > 0 {
		field, wireType, value, rest := parseField(t, data)
		switch field {
		case 1:
			if wireType != 2 {
				t.Fatalf("scalar tag has wire type %d", wireType)
			}
			tag = string(value)
		case 2:
			if wireType != 5 || len(value) != 4 {
				t.Fatalf("scalar simple_value has wire type %d and length %d", wireType, len(value))
			}
			scalar = &parsedScalar{tag: tag, value: math.Float32frombits(binary.LittleEndian.Uint32(value))}
		case 5: // Summary.Value.histo in TensorFlow's summary.proto.
			if wireType != 2 {
				t.Fatalf("histogram has wire type %d", wireType)
			}
			parsed := parseHistogram(t, value)
			parsed.tag = tag
			histogram = &parsed
		case 6:
			t.Fatal("histogram used Summary.Value field 6 (audio), want field 5 (histo)")
		}
		data = rest
	}
	if scalar != nil {
		scalar.tag = tag
	}
	if histogram != nil {
		histogram.tag = tag
	}
	return scalar, histogram
}

func parseHistogram(t *testing.T, data []byte) parsedHistogram {
	t.Helper()
	var histogram parsedHistogram
	for len(data) > 0 {
		field, wireType, value, rest := parseField(t, data)
		if wireType != 1 || len(value) != 8 {
			t.Fatalf("histogram field %d has wire type %d and length %d", field, wireType, len(value))
		}
		decoded := math.Float64frombits(binary.LittleEndian.Uint64(value))
		switch field {
		case 1:
			histogram.minimum = decoded
		case 2:
			histogram.maximum = decoded
		case 3:
			histogram.count = decoded
		case 4:
			histogram.sum = decoded
		case 5:
			histogram.sumSquares = decoded
		case 6:
			histogram.bucketLimits = append(histogram.bucketLimits, decoded)
		case 7:
			histogram.buckets = append(histogram.buckets, decoded)
		}
		data = rest
	}
	return histogram
}

func equalFloat64s(got, want []float64) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range got {
		if math.Float64bits(got[i]) != math.Float64bits(want[i]) {
			return false
		}
	}
	return true
}

func parseField(t *testing.T, data []byte) (int, int, []byte, []byte) {
	t.Helper()
	key, keyLength := binary.Uvarint(data)
	if keyLength <= 0 {
		t.Fatalf("invalid protobuf field key")
	}
	data = data[keyLength:]
	field, wireType := int(key>>3), int(key&7)
	switch wireType {
	case 0:
		_, length := binary.Uvarint(data)
		if length <= 0 {
			t.Fatalf("invalid protobuf varint")
		}
		return field, wireType, data[:length], data[length:]
	case 1:
		if len(data) < 8 {
			t.Fatalf("truncated protobuf fixed64")
		}
		return field, wireType, data[:8], data[8:]
	case 2:
		length, lengthSize := binary.Uvarint(data)
		if lengthSize <= 0 || length > uint64(len(data)-lengthSize) {
			t.Fatalf("invalid protobuf byte length")
		}
		start := lengthSize
		end := start + int(length)
		return field, wireType, data[start:end], data[end:]
	case 5:
		if len(data) < 4 {
			t.Fatalf("truncated protobuf fixed32")
		}
		return field, wireType, data[:4], data[4:]
	default:
		t.Fatalf("unsupported protobuf wire type %d", wireType)
		return 0, 0, nil, nil
	}
}

package tensorboard

import (
	"encoding/binary"
	"errors"
	"fmt"
	"hash/crc32"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// ErrNoEventFiles indicates that an event directory contains no TensorBoard
// event files produced by Writer.
var ErrNoEventFiles = errors.New("no TensorBoard event files found")

// ErrInvalidRecord indicates a corrupt or truncated TFRecord frame.
var ErrInvalidRecord = errors.New("invalid TensorBoard TFRecord")

// ErrMalformedEvent indicates a syntactically invalid Event, Summary, or
// HistogramProto payload.
var ErrMalformedEvent = errors.New("malformed TensorBoard event")

const maxEventRecordSize = 64 << 20

// ScalarRecord identifies one scalar summary by tag and global training step.
// SourceFile is the base event-file name, which makes diagnostics useful when
// a resumed logical run consists of more than one physical event file.
type ScalarRecord struct {
	Tag        string
	Step       int64
	Value      float32
	SourceFile string
}

// HistogramRecord identifies one histogram summary by tag and global training
// step. Count is the HistogramProto count field written by Writer.
type HistogramRecord struct {
	Tag        string
	Step       int64
	Count      float64
	SourceFile string
}

// Inspection is the scalar and histogram index recovered from every event file
// in one TensorBoard log directory. Records are ordered by sorted event-file
// name and then by their order within each file.
type Inspection struct {
	Scalars    []ScalarRecord
	Histograms []HistogramRecord
}

// InspectDir safely decodes this project's TensorBoard event files in logDir.
// It verifies both TFRecord CRCs, bounds allocations, and parses only the
// Event/Summary/HistogramProto subset written by Writer. Unknown protobuf
// fields are skipped, so future TensorBoard metadata does not break proof-run
// inspection.
func InspectDir(logDir string) (Inspection, error) {
	entries, err := os.ReadDir(logDir)
	if err != nil {
		return Inspection{}, fmt.Errorf("read TensorBoard event directory %q: %w", logDir, err)
	}
	paths := make([]string, 0, len(entries))
	for _, entry := range entries {
		if entry.Type().IsRegular() && strings.HasPrefix(entry.Name(), "events.out.tfevents.") {
			paths = append(paths, filepath.Join(logDir, entry.Name()))
		}
	}
	if len(paths) == 0 {
		return Inspection{}, fmt.Errorf("%w in %q", ErrNoEventFiles, logDir)
	}
	sort.Strings(paths)

	var inspection Inspection
	for _, path := range paths {
		fileInspection, err := InspectFile(path)
		if err != nil {
			return Inspection{}, err
		}
		inspection.Scalars = append(inspection.Scalars, fileInspection.Scalars...)
		inspection.Histograms = append(inspection.Histograms, fileInspection.Histograms...)
	}
	return inspection, nil
}

// InspectFile safely decodes one TensorBoard event file produced by Writer.
func InspectFile(path string) (Inspection, error) {
	file, err := os.Open(path)
	if err != nil {
		return Inspection{}, fmt.Errorf("open TensorBoard event file %q: %w", path, err)
	}
	defer func() { _ = file.Close() }()
	info, err := file.Stat()
	if err != nil {
		return Inspection{}, fmt.Errorf("stat TensorBoard event file %q: %w", path, err)
	}
	if !info.Mode().IsRegular() {
		return Inspection{}, fmt.Errorf("TensorBoard event path %q is not a regular file", path)
	}

	var inspection Inspection
	for recordNumber := 1; ; recordNumber++ {
		payload, err := readEventRecord(file)
		if errors.Is(err, io.EOF) {
			return inspection, nil
		}
		if err != nil {
			return Inspection{}, fmt.Errorf("read TensorBoard event file %q record %d: %w", path, recordNumber, err)
		}
		scalars, histograms, err := decodeEvent(payload, filepath.Base(path))
		if err != nil {
			return Inspection{}, fmt.Errorf("decode TensorBoard event file %q record %d: %w", path, recordNumber, err)
		}
		inspection.Scalars = append(inspection.Scalars, scalars...)
		inspection.Histograms = append(inspection.Histograms, histograms...)
	}
}

func readEventRecord(reader io.Reader) ([]byte, error) {
	var lengthBytes [8]byte
	n, err := io.ReadFull(reader, lengthBytes[:])
	if err != nil {
		if errors.Is(err, io.EOF) && n == 0 {
			return nil, io.EOF
		}
		return nil, fmt.Errorf("%w: truncated record length", ErrInvalidRecord)
	}
	var lengthCRCBytes [4]byte
	if _, err := io.ReadFull(reader, lengthCRCBytes[:]); err != nil {
		return nil, fmt.Errorf("%w: truncated record length CRC", ErrInvalidRecord)
	}
	if got, want := binary.LittleEndian.Uint32(lengthCRCBytes[:]), maskedCRC(lengthBytes[:]); got != want {
		return nil, fmt.Errorf("%w: length CRC %08x, want %08x", ErrInvalidRecord, got, want)
	}
	length := binary.LittleEndian.Uint64(lengthBytes[:])
	if length > maxEventRecordSize {
		return nil, fmt.Errorf("%w: record length %d exceeds %d-byte limit", ErrInvalidRecord, length, maxEventRecordSize)
	}
	payload := make([]byte, int(length))
	if _, err := io.ReadFull(reader, payload); err != nil {
		return nil, fmt.Errorf("%w: truncated %d-byte payload", ErrInvalidRecord, length)
	}
	var payloadCRCBytes [4]byte
	if _, err := io.ReadFull(reader, payloadCRCBytes[:]); err != nil {
		return nil, fmt.Errorf("%w: truncated payload CRC", ErrInvalidRecord)
	}
	if got, want := binary.LittleEndian.Uint32(payloadCRCBytes[:]), maskedCRC(payload); got != want {
		return nil, fmt.Errorf("%w: payload CRC %08x, want %08x", ErrInvalidRecord, got, want)
	}
	return payload, nil
}

func decodeEvent(data []byte, sourceFile string) ([]ScalarRecord, []HistogramRecord, error) {
	var step int64
	summaries := make([][]byte, 0, 1)
	for len(data) != 0 {
		field, wireType, value, rest, err := nextProtoField(data)
		if err != nil {
			return nil, nil, err
		}
		switch field {
		case 2: // Event.step
			if wireType != 0 {
				return nil, nil, malformedField("Event.step", wireType, 0)
			}
			if value.varint > math.MaxInt64 {
				return nil, nil, fmt.Errorf("%w: Event.step %d exceeds int64", ErrMalformedEvent, value.varint)
			}
			step = int64(value.varint)
		case 5: // Event.summary
			if wireType != 2 {
				return nil, nil, malformedField("Event.summary", wireType, 2)
			}
			summaries = append(summaries, value.bytes)
		}
		data = rest
	}
	var scalars []ScalarRecord
	var histograms []HistogramRecord
	for _, summary := range summaries {
		decodedScalars, decodedHistograms, err := decodeSummary(summary, step, sourceFile)
		if err != nil {
			return nil, nil, err
		}
		scalars = append(scalars, decodedScalars...)
		histograms = append(histograms, decodedHistograms...)
	}
	return scalars, histograms, nil
}

func decodeSummary(data []byte, step int64, sourceFile string) ([]ScalarRecord, []HistogramRecord, error) {
	var scalars []ScalarRecord
	var histograms []HistogramRecord
	for len(data) != 0 {
		field, wireType, value, rest, err := nextProtoField(data)
		if err != nil {
			return nil, nil, err
		}
		if field == 1 { // Summary.value
			if wireType != 2 {
				return nil, nil, malformedField("Summary.value", wireType, 2)
			}
			scalar, histogram, err := decodeSummaryValue(value.bytes, step, sourceFile)
			if err != nil {
				return nil, nil, err
			}
			if scalar != nil {
				scalars = append(scalars, *scalar)
			}
			if histogram != nil {
				histograms = append(histograms, *histogram)
			}
		}
		data = rest
	}
	return scalars, histograms, nil
}

func decodeSummaryValue(data []byte, step int64, sourceFile string) (*ScalarRecord, *HistogramRecord, error) {
	var tag string
	var scalar *ScalarRecord
	var histogram *HistogramRecord
	for len(data) != 0 {
		field, wireType, value, rest, err := nextProtoField(data)
		if err != nil {
			return nil, nil, err
		}
		switch field {
		case 1: // Summary.Value.tag
			if wireType != 2 {
				return nil, nil, malformedField("Summary.Value.tag", wireType, 2)
			}
			tag = string(value.bytes)
		case 2: // Summary.Value.simple_value
			if wireType != 5 {
				return nil, nil, malformedField("Summary.Value.simple_value", wireType, 5)
			}
			if scalar != nil || histogram != nil {
				return nil, nil, fmt.Errorf("%w: Summary.Value has multiple supported value kinds", ErrMalformedEvent)
			}
			scalar = &ScalarRecord{Step: step, Value: math.Float32frombits(binary.LittleEndian.Uint32(value.bytes)), SourceFile: sourceFile}
		case 5, 6: // Summary.Value.histo; field 6 reads retained pre-fix runs only.
			if wireType != 2 {
				return nil, nil, malformedField("Summary.Value.histo", wireType, 2)
			}
			if scalar != nil || histogram != nil {
				return nil, nil, fmt.Errorf("%w: Summary.Value has multiple supported value kinds", ErrMalformedEvent)
			}
			count, err := decodeHistogramCount(value.bytes)
			if err != nil {
				return nil, nil, err
			}
			histogram = &HistogramRecord{Step: step, Count: count, SourceFile: sourceFile}
		}
		data = rest
	}
	if scalar != nil {
		scalar.Tag = tag
	}
	if histogram != nil {
		histogram.Tag = tag
	}
	return scalar, histogram, nil
}

func decodeHistogramCount(data []byte) (float64, error) {
	var count float64
	for len(data) != 0 {
		field, wireType, value, rest, err := nextProtoField(data)
		if err != nil {
			return 0, err
		}
		if field >= 1 && field <= 7 { // HistogramProto's fixed64 fields.
			if wireType != 1 {
				return 0, malformedField("HistogramProto fixed64 field", wireType, 1)
			}
			if field == 3 { // HistogramProto.num
				count = math.Float64frombits(binary.LittleEndian.Uint64(value.bytes))
			}
		}
		data = rest
	}
	return count, nil
}

type protoValue struct {
	varint uint64
	bytes  []byte
}

func nextProtoField(data []byte) (field int, wireType int, value protoValue, rest []byte, err error) {
	key, n := binary.Uvarint(data)
	if n <= 0 {
		return 0, 0, protoValue{}, nil, fmt.Errorf("%w: invalid protobuf field key", ErrMalformedEvent)
	}
	field = int(key >> 3)
	wireType = int(key & 7)
	if field == 0 {
		return 0, 0, protoValue{}, nil, fmt.Errorf("%w: protobuf field number is zero", ErrMalformedEvent)
	}
	data = data[n:]
	switch wireType {
	case 0:
		encoded, consumed := binary.Uvarint(data)
		if consumed <= 0 {
			return 0, 0, protoValue{}, nil, fmt.Errorf("%w: invalid varint for field %d", ErrMalformedEvent, field)
		}
		return field, wireType, protoValue{varint: encoded}, data[consumed:], nil
	case 1:
		if len(data) < 8 {
			return 0, 0, protoValue{}, nil, fmt.Errorf("%w: truncated fixed64 field %d", ErrMalformedEvent, field)
		}
		return field, wireType, protoValue{bytes: data[:8]}, data[8:], nil
	case 2:
		length, consumed := binary.Uvarint(data)
		if consumed <= 0 || length > uint64(len(data)-consumed) {
			return 0, 0, protoValue{}, nil, fmt.Errorf("%w: invalid length-delimited field %d", ErrMalformedEvent, field)
		}
		start := consumed
		end := start + int(length)
		return field, wireType, protoValue{bytes: data[start:end]}, data[end:], nil
	case 5:
		if len(data) < 4 {
			return 0, 0, protoValue{}, nil, fmt.Errorf("%w: truncated fixed32 field %d", ErrMalformedEvent, field)
		}
		return field, wireType, protoValue{bytes: data[:4]}, data[4:], nil
	default:
		return 0, 0, protoValue{}, nil, fmt.Errorf("%w: unsupported protobuf wire type %d", ErrMalformedEvent, wireType)
	}
}

func malformedField(name string, got, want int) error {
	return fmt.Errorf("%w: %s uses wire type %d, want %d", ErrMalformedEvent, name, got, want)
}

func maskedCRC(data []byte) uint32 {
	checksum := crc32.Checksum(data, castagnoliTable)
	return (checksum>>15 | checksum<<17) + 0xa282ead8
}

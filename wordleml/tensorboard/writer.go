// Package tensorboard writes TensorBoard event files without protobuf dependencies.
package tensorboard

import (
	"encoding/binary"
	"errors"
	"hash/crc32"
	"math"
	"os"
	"sort"
	"sync"
	"time"
)

// ErrClosed is returned when an operation is attempted on a closed Writer.
var ErrClosed = errors.New("tensorboard writer is closed")

var castagnoliTable = crc32.MakeTable(crc32.Castagnoli)

// Scalar is one named floating-point value for a TensorBoard summary event.
type Scalar struct {
	Tag   string
	Value float32
}

// Histogram is one named distribution for a TensorBoard summary event. Non-finite
// values are ignored when the histogram is written, so telemetry remains a valid
// TensorFlow HistogramProto even when an upstream diagnostic has bad values.
type Histogram struct {
	Tag    string
	Values []float64
}

const maxHistogramBuckets = 30

// Writer appends summaries to one TensorBoard event file.
type Writer struct {
	mu     sync.Mutex
	file   *os.File
	closed bool
}

// New creates a unique TensorBoard event file in logDir and writes its file version.
func New(logDir string) (*Writer, error) {
	if err := os.MkdirAll(logDir, 0o755); err != nil {
		return nil, err
	}
	file, err := os.CreateTemp(logDir, "events.out.tfevents.")
	if err != nil {
		return nil, err
	}
	w := &Writer{file: file}
	if err := w.writeEvent(fileVersionEvent()); err != nil {
		_ = file.Close()
		return nil, err
	}
	return w, nil
}

// WriteScalars writes one summary event at step.
func (w *Writer) WriteScalars(step int64, scalars ...Scalar) error {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.closed {
		return ErrClosed
	}
	return w.writeEvent(scalarEvent(step, scalars))
}

// WriteHistograms writes one summary event at step. Empty inputs, including
// inputs containing only NaN or infinity, are represented as zero-sized
// histograms. Finite values are grouped deterministically into at most 30
// buckets.
func (w *Writer) WriteHistograms(step int64, histograms ...Histogram) error {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.closed {
		return ErrClosed
	}
	return w.writeEvent(histogramEvent(step, histograms))
}

// Flush commits all written events to stable storage.
func (w *Writer) Flush() error {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.closed {
		return ErrClosed
	}
	return w.file.Sync()
}

// Close flushes and closes the event file.
func (w *Writer) Close() error {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.closed {
		return ErrClosed
	}
	w.closed = true
	if err := w.file.Sync(); err != nil {
		_ = w.file.Close()
		return err
	}
	return w.file.Close()
}

func (w *Writer) writeEvent(event []byte) error {
	return writeRecord(w.file, event)
}

func fileVersionEvent() []byte {
	event := make([]byte, 0, 32)
	event = appendFixed64(event, 1, math.Float64bits(float64(time.Now().UnixNano())/1e9))
	event = appendString(event, 3, "brain.Event:2")
	return event
}

func scalarEvent(step int64, scalars []Scalar) []byte {
	summary := make([]byte, 0, len(scalars)*24)
	for _, scalar := range scalars {
		value := make([]byte, 0, len(scalar.Tag)+8)
		value = appendString(value, 1, scalar.Tag)
		value = appendFixed32(value, 2, math.Float32bits(scalar.Value))
		summary = appendBytes(summary, 1, value)
	}

	event := make([]byte, 0, len(summary)+32)
	event = appendFixed64(event, 1, math.Float64bits(float64(time.Now().UnixNano())/1e9))
	event = appendVarintField(event, 2, uint64(step))
	return appendBytes(event, 5, summary)
}

func histogramEvent(step int64, histograms []Histogram) []byte {
	summary := make([]byte, 0, len(histograms)*64)
	for _, histogram := range histograms {
		value := make([]byte, 0, len(histogram.Tag)+64)
		value = appendString(value, 1, histogram.Tag)
		value = appendBytes(value, 6, histogramProto(histogram.Values))
		summary = appendBytes(summary, 1, value)
	}

	event := make([]byte, 0, len(summary)+32)
	event = appendFixed64(event, 1, math.Float64bits(float64(time.Now().UnixNano())/1e9))
	event = appendVarintField(event, 2, uint64(step))
	return appendBytes(event, 5, summary)
}

func histogramProto(values []float64) []byte {
	finite := make([]float64, 0, len(values))
	for _, value := range values {
		if !math.IsNaN(value) && !math.IsInf(value, 0) {
			finite = append(finite, value)
		}
	}
	if len(finite) == 0 {
		return histogramFields(0, 0, 0, 0, 0, nil, nil)
	}

	sort.Float64s(finite)
	minimum, maximum := finite[0], finite[len(finite)-1]
	var sum, sumSquares float64
	for _, value := range finite {
		sum = saturatingAdd(sum, value)
		sumSquares = saturatingAdd(sumSquares, saturatingSquare(value))
	}
	limits, counts := histogramBuckets(finite)
	return histogramFields(minimum, maximum, float64(len(finite)), sum, sumSquares, limits, counts)
}

func histogramFields(minimum, maximum, count, sum, sumSquares float64, limits, buckets []float64) []byte {
	proto := make([]byte, 0, 48+len(limits)*18)
	proto = appendFixed64(proto, 1, math.Float64bits(minimum))
	proto = appendFixed64(proto, 2, math.Float64bits(maximum))
	proto = appendFixed64(proto, 3, math.Float64bits(count))
	proto = appendFixed64(proto, 4, math.Float64bits(sum))
	proto = appendFixed64(proto, 5, math.Float64bits(sumSquares))
	for _, limit := range limits {
		proto = appendFixed64(proto, 6, math.Float64bits(limit))
	}
	for _, bucket := range buckets {
		proto = appendFixed64(proto, 7, math.Float64bits(bucket))
	}
	return proto
}

func histogramBuckets(sorted []float64) (limits, counts []float64) {
	// First compact equal values. This keeps a constant-valued tensor useful
	// rather than emitting dozens of indistinguishable buckets.
	type group struct {
		limit float64
		count float64
	}
	groups := make([]group, 0, len(sorted))
	for _, value := range sorted {
		if len(groups) == 0 || value != groups[len(groups)-1].limit {
			groups = append(groups, group{limit: value})
		}
		groups[len(groups)-1].count++
	}
	if len(groups) <= maxHistogramBuckets {
		limits = make([]float64, len(groups))
		counts = make([]float64, len(groups))
		for i, group := range groups {
			limits[i], counts[i] = group.limit, group.count
		}
		return limits, counts
	}

	limits = make([]float64, 0, maxHistogramBuckets)
	counts = make([]float64, 0, maxHistogramBuckets)
	for start := 0; start < len(groups); {
		remainingGroups := len(groups) - start
		remainingBuckets := maxHistogramBuckets - len(limits)
		width := (remainingGroups + remainingBuckets - 1) / remainingBuckets
		end := start + width
		var bucketCount float64
		for _, group := range groups[start:end] {
			bucketCount += group.count
		}
		limits = append(limits, groups[end-1].limit)
		counts = append(counts, bucketCount)
		start = end
	}
	return limits, counts
}

func saturatingAdd(total, value float64) float64 {
	if value > 0 && total > math.MaxFloat64-value {
		return math.MaxFloat64
	}
	if value < 0 && total < -math.MaxFloat64-value {
		return -math.MaxFloat64
	}
	return total + value
}

func saturatingSquare(value float64) float64 {
	if math.Abs(value) > math.Sqrt(math.MaxFloat64) {
		return math.MaxFloat64
	}
	return value * value
}

func writeRecord(file *os.File, payload []byte) error {
	var length [8]byte
	binary.LittleEndian.PutUint64(length[:], uint64(len(payload)))
	if _, err := file.Write(length[:]); err != nil {
		return err
	}
	if err := writeCRC(file, length[:]); err != nil {
		return err
	}
	if _, err := file.Write(payload); err != nil {
		return err
	}
	return writeCRC(file, payload)
}

func writeCRC(file *os.File, data []byte) error {
	var checksum [4]byte
	binary.LittleEndian.PutUint32(checksum[:], maskCRC(crc32.Checksum(data, castagnoliTable)))
	_, err := file.Write(checksum[:])
	return err
}

func maskCRC(checksum uint32) uint32 {
	return (checksum>>15 | checksum<<17) + 0xa282ead8
}

func appendVarintField(dst []byte, field int, value uint64) []byte {
	dst = appendVarint(dst, uint64(field<<3))
	return appendVarint(dst, value)
}

func appendString(dst []byte, field int, value string) []byte {
	return appendBytes(dst, field, []byte(value))
}

func appendBytes(dst []byte, field int, value []byte) []byte {
	dst = appendVarint(dst, uint64(field<<3|2))
	dst = appendVarint(dst, uint64(len(value)))
	return append(dst, value...)
}

func appendFixed32(dst []byte, field int, value uint32) []byte {
	dst = appendVarint(dst, uint64(field<<3|5))
	var encoded [4]byte
	binary.LittleEndian.PutUint32(encoded[:], value)
	return append(dst, encoded[:]...)
}

func appendFixed64(dst []byte, field int, value uint64) []byte {
	dst = appendVarint(dst, uint64(field<<3|1))
	var encoded [8]byte
	binary.LittleEndian.PutUint64(encoded[:], value)
	return append(dst, encoded[:]...)
}

func appendVarint(dst []byte, value uint64) []byte {
	var encoded [10]byte
	n := binary.PutUvarint(encoded[:], value)
	return append(dst, encoded[:n]...)
}

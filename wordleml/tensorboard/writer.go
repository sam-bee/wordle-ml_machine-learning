// Package tensorboard writes TensorBoard scalar event files without protobuf dependencies.
package tensorboard

import (
	"encoding/binary"
	"errors"
	"hash/crc32"
	"math"
	"os"
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

// Writer appends scalar summaries to one TensorBoard event file.
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

// WriteScalars writes and flushes one summary event at step.
func (w *Writer) WriteScalars(step int64, scalars ...Scalar) error {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.closed {
		return ErrClosed
	}
	return w.writeEvent(scalarEvent(step, scalars))
}

// Flush makes all written events visible to readers.
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
	if err := writeRecord(w.file, event); err != nil {
		return err
	}
	return w.file.Sync()
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

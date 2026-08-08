package gameeval

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"reflect"
	"strings"
	"testing"
)

func TestWriteJSONLWritesOneDecodableGamePerLineInOrder(t *testing.T) {
	want := []GameResult{
		{Solution: "GEESE", Solved: true, Guesses: 2, Turns: []TurnResult{{Turn: 1, Guess: "EERIE", Feedback: "YG--G"}, {Turn: 2, Guess: "GEESE", Feedback: "GGGGG"}}},
		{Solution: "CIGAR", Solved: false, Guesses: 6, Failure: "unsolved_after_six_guesses", InvalidSelections: 1, SuppressedRawTopSelections: 2, RepeatedSelections: 1, Turns: []TurnResult{}},
	}
	var output bytes.Buffer
	if err := WriteJSONL(&output, want); err != nil {
		t.Fatal(err)
	}
	if !strings.HasSuffix(output.String(), "\n") {
		t.Fatalf("JSONL output lacks final newline: %q", output.String())
	}
	lines := strings.Split(strings.TrimSuffix(output.String(), "\n"), "\n")
	if got, wantLines := len(lines), len(want); got != wantLines {
		t.Fatalf("line count = %d, want %d", got, wantLines)
	}
	got := make([]GameResult, len(lines))
	for index, line := range lines {
		if err := json.Unmarshal([]byte(line), &got[index]); err != nil {
			t.Fatalf("decode line %d: %v", index+1, err)
		}
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("decoded games = %#v, want %#v", got, want)
	}
}

func TestWriteJSONLHandlesShortWritesAndFailures(t *testing.T) {
	writer := &shortWriter{limit: 3}
	if err := WriteJSONL(writer, []GameResult{{Solution: "CIGAR"}}); err != nil {
		t.Fatalf("short writer: %v", err)
	}
	if !strings.HasSuffix(writer.String(), "\n") {
		t.Fatalf("short writer output lacks newline: %q", writer.String())
	}

	errWriter := errorWriter{err: errors.New("disk full")}
	if err := WriteJSONL(errWriter, []GameResult{{Solution: "CIGAR"}}); !errors.Is(err, errWriter.err) {
		t.Fatalf("write error = %v, want disk full", err)
	}
	if err := WriteJSONL(nil, nil); err == nil {
		t.Fatal("nil writer succeeded")
	}
}

type shortWriter struct {
	bytes.Buffer
	limit int
}

func (w *shortWriter) Write(contents []byte) (int, error) {
	if len(contents) > w.limit {
		contents = contents[:w.limit]
	}
	return w.Buffer.Write(contents)
}

type errorWriter struct{ err error }

func (w errorWriter) Write([]byte) (int, error) { return 0, w.err }

var _ io.Writer = (*shortWriter)(nil)

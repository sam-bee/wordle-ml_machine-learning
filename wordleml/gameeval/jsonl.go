package gameeval

import (
	"encoding/json"
	"fmt"
	"io"
)

// WriteJSONL writes exactly one complete GameResult JSON object and trailing
// newline for every supplied result, preserving its stable caller-provided
// solution order. It intentionally writes no aggregate summary: callers keep
// that separately in final metrics.
func WriteJSONL(writer io.Writer, games []GameResult) error {
	if writer == nil {
		return fmt.Errorf("JSONL writer must not be nil")
	}
	for index, game := range games {
		encoded, err := json.Marshal(game)
		if err != nil {
			return fmt.Errorf("encode game %d solution %q: %w", index, game.Solution, err)
		}
		encoded = append(encoded, '\n')
		if err := writeAll(writer, encoded); err != nil {
			return fmt.Errorf("write game %d solution %q: %w", index, game.Solution, err)
		}
	}
	return nil
}

func writeAll(writer io.Writer, contents []byte) error {
	for len(contents) > 0 {
		written, err := writer.Write(contents)
		if written > 0 {
			contents = contents[written:]
		}
		if err != nil {
			return err
		}
		if written == 0 {
			return io.ErrShortWrite
		}
	}
	return nil
}

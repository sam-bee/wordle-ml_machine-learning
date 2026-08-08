// Command inspect prints one frozen imitation-learning record in a form useful
// for checking the dataset-to-model boundary.
package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"github.com/gomlx/gomlx/core/tensors"
	"github.com/sam-bee/wordle-ml_machine-learning/imitationdata"
	"github.com/sam-bee/wordle-ml_machine-learning/modelstate"
	"github.com/sam-bee/wordle-ml_machine-learning/vocabulary"
)

func main() {
	dataDir := os.Getenv("WORDLEML_DATA_DIR")
	if dataDir == "" {
		dataDir = "../data"
	}
	splitName := flag.String("split", "mini", "dataset split: mini, train, or validation")
	recordIndex := flag.Int("record", 0, "record index")
	candidateLimit := flag.Int("candidates", 12, "number of candidate words to print")
	vocabularyDir := flag.String("data-dir", dataDir, "directory containing frozen word lists")
	imitationDir := flag.String("imitation-dir", "", "directory containing WDIT files (default: <data-dir>/imitation)")
	flag.Parse()

	split := imitationdata.Split(*splitName)
	if split == imitationdata.Test {
		fatalf("inspect deliberately refuses the final-test split")
	}
	if split != imitationdata.Mini && split != imitationdata.Train && split != imitationdata.Validation {
		fatalf("-split must be mini, train, or validation")
	}
	if *candidateLimit < 0 {
		fatalf("-candidates must not be negative")
	}
	if *imitationDir == "" {
		*imitationDir = filepath.Join(*vocabularyDir, "imitation")
	}
	v, err := vocabulary.Load(*vocabularyDir)
	if err != nil {
		fatalf("load vocabulary: %v", err)
	}
	data, err := imitationdata.Load(v, *imitationDir, split)
	if err != nil {
		fatalf("load %s: %v", split, err)
	}
	example, err := data.Example(*recordIndex)
	if err != nil {
		fatalf("record %d: %v", *recordIndex, err)
	}

	header := data.Header()
	fmt.Printf("split=%s records=%d header={version:%d record_size:%d top_k:%d actions:%d solutions:%d}\n",
		data.Split(), data.Len(), header.Version, header.RecordSize, header.TopK, header.GlobalActionCount, header.GlobalSolutionCount)
	fmt.Printf("record=%d turn=%d candidates=%d opening=%t\n", *recordIndex, example.Record.TurnDepth, example.Record.CandidateCount, example.Record.SolutionID == 0xffff)
	for i := 0; i < int(example.Record.TurnDepth); i++ {
		turn := example.Record.History[i]
		word, _ := v.ActionWord(int(turn.GuessID))
		fmt.Printf("history[%d]=%d %s %s\n", i, turn.GuessID, word, feedback(turn.FeedbackCode))
	}
	printed := 0
	fmt.Print("candidate_words:")
	for solutionID := 0; solutionID < vocabulary.NumSolutions && printed < *candidateLimit; solutionID++ {
		if example.Record.CandidateBits[solutionID/8]&(1<<uint(solutionID%8)) == 0 {
			continue
		}
		word, _ := v.SolutionWord(solutionID)
		fmt.Printf(" %s", word)
		printed++
	}
	fmt.Println()
	for i, actionID := range example.Record.TopKActionIDs {
		word, _ := v.ActionWord(int(actionID))
		fmt.Printf("teacher[%d]=%d %s reduction=%.6f worst_case=%d\n", i+1, actionID, word, example.Record.TopKReductionRatios[i], example.Record.TopKWorstCaseSizes[i])
	}
	printTensor("candidate_mask", tensors.FromFlatDataAndDimensions(example.CandidateMask, vocabulary.NumSolutions))
	printTensor("candidate_stats", tensors.FromFlatDataAndDimensions(example.CandidateStats, modelstate.CandidateStatsSize))
	printTensor("turn", tensors.FromScalar(example.Turn))
	printTensor("remaining_action_mask", tensors.FromFlatDataAndDimensions(example.RemainingActionMask, vocabulary.NumActions))
	printTensor("available_action_mask", tensors.FromFlatDataAndDimensions(example.AvailableActionMask, vocabulary.NumActions))
	printTensor("label", tensors.FromFlatDataAndDimensions([]int32{example.TeacherTopAction}, 1))
}

func feedback(code uint8) string {
	result := [5]byte{}
	for i := range result {
		switch code % 3 {
		case 0:
			result[i] = '-'
		case 1:
			result[i] = 'Y'
		case 2:
			result[i] = 'G'
		}
		code /= 3
	}
	return string(result[:])
}

func printTensor(name string, tensor *tensors.Tensor) {
	defer tensor.MustFinalizeAll()
	fmt.Printf("tensor %s shape=%s dtype=%s\n", name, tensor.Shape(), tensor.DType())
}

func fatalf(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "inspect: "+format+"\n", args...)
	os.Exit(1)
}

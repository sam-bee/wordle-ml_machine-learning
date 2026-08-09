package checkpoints

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestCreateMakesIsolatedPPOLayout(t *testing.T) {
	root := filepath.Join(t.TempDir(), "checkpoints")
	layout, err := Create(root, "ppo-pilot-001")
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	if layout.Root != root {
		t.Fatalf("root = %q, want %q", layout.Root, root)
	}
	for _, path := range []string{
		layout.Root,
		layout.SupervisedBaselineDir,
		layout.PPODir,
		layout.Dir,
		layout.EventsDir,
		layout.BestDir,
	} {
		info, err := os.Stat(path)
		if err != nil {
			t.Errorf("required directory %q: %v", path, err)
			continue
		}
		if !info.IsDir() {
			t.Errorf("required path %q is not a directory", path)
		}
	}
	if _, err := os.Stat(layout.SupervisedBaselineMetadataPath); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("supervised metadata exists before identity is recorded: %v", err)
	}
	if err := layout.WriteSupervisedBaselineMetadata(map[string]string{
		"checkpoint": "/read-only/supervised/checkpoint",
		"sha256":     checksum('a'),
	}); err != nil {
		t.Fatalf("WriteSupervisedBaselineMetadata: %v", err)
	}
	if err := layout.WriteSupervisedBaselineMetadata(map[string]string{
		"checkpoint": "/read-only/supervised/checkpoint",
		"sha256":     checksum('a'),
	}); err != nil {
		t.Fatalf("idempotent baseline identity: %v", err)
	}
	if err := layout.WriteSupervisedBaselineMetadata(map[string]string{"checkpoint": "different"}); err == nil {
		t.Fatal("different supervised identity overwrote immutable baseline metadata")
	}
	if err := layout.WriteConfig(map[string]any{"algorithm": "ppo", "seed": 71}); err != nil {
		t.Fatalf("WriteConfig: %v", err)
	}
	if err := layout.WriteMetadata(map[string]any{"final_test_sealed_unopened": true}); err != nil {
		t.Fatalf("WriteMetadata: %v", err)
	}
	for _, path := range []string{layout.ConfigPath, layout.MetadataPath, layout.SupervisedBaselineMetadataPath} {
		contents, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("read %q: %v", path, err)
		}
		if !strings.HasSuffix(string(contents), "\n") {
			t.Errorf("metadata %q has no trailing newline", path)
		}
	}
	if _, err := Create(root, "ppo-pilot-001"); err == nil {
		t.Fatal("Create reused existing PPO run ID")
	}
	opened, err := Open(root, "ppo-pilot-001")
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	if !reflect.DeepEqual(opened, layout) {
		t.Fatalf("Open = %#v, want %#v", opened, layout)
	}
}

func TestIterationPathsAndStateRoundTrip(t *testing.T) {
	layout, err := Create(t.TempDir(), "paths")
	if err != nil {
		t.Fatal(err)
	}
	iteration, err := layout.CreateIteration(7)
	if err != nil {
		t.Fatalf("CreateIteration: %v", err)
	}
	if got, want := filepath.Base(iteration.Dir), "iter-007"; got != want {
		t.Fatalf("iteration directory = %q, want %q", got, want)
	}
	for _, path := range []string{iteration.ActorCriticDir, iteration.ActorDir, iteration.CriticDir, iteration.ActorOnlyDir} {
		info, err := os.Stat(path)
		if err != nil || !info.IsDir() {
			t.Errorf("checkpoint destination %q = %v, %v", path, info, err)
		}
	}
	state := IterationState{
		SchemaVersion:  SchemaVersion,
		Iteration:      7,
		RolloutSeed:    -71,
		ActorSteps:     4,
		CriticSteps:    33,
		ActorChecksum:  checksum('a'),
		CriticChecksum: checksum('b'),
	}
	if err := iteration.WriteState(state); err != nil {
		t.Fatalf("WriteState: %v", err)
	}
	got, err := iteration.LoadState()
	if err != nil {
		t.Fatalf("LoadState: %v", err)
	}
	if !reflect.DeepEqual(got, state) {
		t.Fatalf("iteration state = %#v, want %#v", got, state)
	}
	if err := iteration.WriteEvaluation(map[string]any{"solved": 190, "numerical_failure": false}); err != nil {
		t.Fatalf("WriteEvaluation: %v", err)
	}
	var evaluation map[string]any
	contents, err := os.ReadFile(iteration.EvaluationPath)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(contents, &evaluation); err != nil || evaluation["solved"] != float64(190) {
		t.Fatalf("evaluation = %s, %v", contents, err)
	}
	for _, path := range []string{iteration.StatePath, iteration.EvaluationPath} {
		leftovers, err := filepath.Glob(filepath.Join(filepath.Dir(path), "."+filepath.Base(path)+".tmp-*"))
		if err != nil {
			t.Fatal(err)
		}
		if len(leftovers) != 0 {
			t.Errorf("temporary artifact left beside %q: %v", path, leftovers)
		}
	}
}

func TestPromotionCopiesOnlySmallMetadataAndRejectedCandidateRetainsAccepted(t *testing.T) {
	layout, err := Create(t.TempDir(), "promotion")
	if err != nil {
		t.Fatal(err)
	}
	first := writeReadyIteration(t, layout, 0, 'a')
	accepted, err := layout.Promote(0)
	if err != nil {
		t.Fatalf("Promote first candidate: %v", err)
	}
	want, err := layout.NewAcceptedState(0)
	if err != nil {
		t.Fatal(err)
	}
	if accepted != want {
		t.Fatalf("accepted state = %#v, want %#v", accepted, want)
	}
	loaded, err := layout.LoadAccepted()
	if err != nil {
		t.Fatalf("LoadAccepted: %v", err)
	}
	if loaded != want {
		t.Fatalf("loaded accepted state = %#v, want %#v", loaded, want)
	}
	for source, destination := range map[string]string{
		first.StatePath:      layout.BestIterationStatePath,
		first.EvaluationPath: layout.BestEvaluationPath,
	} {
		sourceContents, err := os.ReadFile(source)
		if err != nil {
			t.Fatal(err)
		}
		destinationContents, err := os.ReadFile(destination)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(destinationContents, sourceContents) {
			t.Errorf("best metadata %q differs from candidate metadata %q", destination, source)
		}
	}
	// The runner, not Promote, is responsible for copying large payloads into
	// best/. In particular, promotion does not turn actor directories into a
	// second checkpoint copy behind the runner's back.
	if _, err := os.Stat(filepath.Join(layout.BestDir, actorCriticDirectory)); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("Promote unexpectedly copied actor-critic payload: %v", err)
	}

	second := writeReadyIteration(t, layout, 1, 'c')
	// Deliberately reject iteration 1 by not calling Promote. Its diagnostic
	// artifacts remain inspectable, while accepted.json and best/ stay on 0.
	if _, err := os.Stat(second.EvaluationPath); err != nil {
		t.Fatalf("rejected candidate evaluation unavailable: %v", err)
	}
	stillAccepted, err := layout.LoadAccepted()
	if err != nil {
		t.Fatal(err)
	}
	if stillAccepted != want {
		t.Fatalf("rejected candidate changed accepted state: got %#v, want %#v", stillAccepted, want)
	}
	bestState, err := os.ReadFile(layout.BestIterationStatePath)
	if err != nil {
		t.Fatal(err)
	}
	if bytesContain(bestState, []byte("\"iteration\": 1")) {
		t.Fatalf("rejected candidate changed best state: %s", bestState)
	}
}

func TestPromotionRequiresCompleteSmallMetadata(t *testing.T) {
	layout, err := Create(t.TempDir(), "incomplete")
	if err != nil {
		t.Fatal(err)
	}
	iteration, err := layout.CreateIteration(0)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := layout.Promote(0); err == nil {
		t.Fatal("promotion without state and evaluation succeeded")
	}
	if err := iteration.WriteState(validState(0, 'a')); err != nil {
		t.Fatal(err)
	}
	if _, err := layout.Promote(0); err == nil {
		t.Fatal("promotion without evaluation succeeded")
	}
	if _, err := layout.LoadAccepted(); !errors.Is(err, ErrAcceptedNotFound) {
		t.Fatalf("LoadAccepted after failed promotion = %v, want ErrAcceptedNotFound", err)
	}
}

func TestTraversalAndValidationAreRejected(t *testing.T) {
	for _, runID := range []string{"", ".", "..", ".hidden", "../escape", "a/b", "a\\b", "spaces bad", strings.Repeat("a", maximumRunIDLength+1)} {
		if err := ValidateRunID(runID); err == nil {
			t.Errorf("ValidateRunID(%q) succeeded", runID)
		}
	}
	for _, runID := range []string{"ppo-001", "seed_2", "run.3"} {
		if err := ValidateRunID(runID); err != nil {
			t.Errorf("ValidateRunID(%q): %v", runID, err)
		}
	}
	for _, iteration := range []int{-1, maximumIteration + 1} {
		if err := ValidateIteration(iteration); err == nil {
			t.Errorf("ValidateIteration(%d) succeeded", iteration)
		}
	}
	layout, err := New(t.TempDir(), "safe")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := layout.Iteration(-1); err == nil {
		t.Fatal("Iteration(-1) succeeded")
	}
	for _, path := range []string{"", ".", "..", "../other", "/absolute", "iter-000/../actor", "iter-000\\actor"} {
		if safeRelativePath(path) {
			t.Errorf("safeRelativePath(%q) accepted traversal or noncanonical path", path)
		}
	}
	for _, path := range []string{"iter-000/actor-critic", "iter-000/actor-only", "iter-000/evaluation.json"} {
		if !safeRelativePath(path) {
			t.Errorf("safeRelativePath(%q) rejected valid checkpoint path", path)
		}
	}
}

func TestAcceptedStateRoundTripAndPathBinding(t *testing.T) {
	layout, err := Create(t.TempDir(), "accepted")
	if err != nil {
		t.Fatal(err)
	}
	state, err := layout.NewAcceptedState(12)
	if err != nil {
		t.Fatal(err)
	}
	contents, err := json.Marshal(state)
	if err != nil {
		t.Fatal(err)
	}
	var decoded AcceptedState
	if err := json.Unmarshal(contents, &decoded); err != nil {
		t.Fatal(err)
	}
	if decoded != state || state.Validate() != nil {
		t.Fatalf("accepted-state JSON round trip = %#v, want %#v", decoded, state)
	}
	decoded.ActorOnlyCheckpoint = "iter-011/actor-only"
	if err := layout.validateAcceptedState(decoded); err == nil {
		t.Fatal("accepted state was allowed to point to a different iteration")
	}
}

func TestIterationStateValidation(t *testing.T) {
	for name, state := range map[string]IterationState{
		"wrong schema":        {SchemaVersion: 2, Iteration: 0, ActorChecksum: checksum('a'), CriticChecksum: checksum('b')},
		"negative actor step": {SchemaVersion: SchemaVersion, Iteration: 0, ActorSteps: -1, ActorChecksum: checksum('a'), CriticChecksum: checksum('b')},
		"short checksum":      {SchemaVersion: SchemaVersion, Iteration: 0, ActorChecksum: "bad", CriticChecksum: checksum('b')},
		"upper checksum":      {SchemaVersion: SchemaVersion, Iteration: 0, ActorChecksum: strings.ToUpper(checksum('a')), CriticChecksum: checksum('b')},
	} {
		t.Run(name, func(t *testing.T) {
			if err := state.Validate(); err == nil {
				t.Fatalf("Validate(%#v) succeeded", state)
			}
		})
	}
}

func writeReadyIteration(t *testing.T, layout Layout, iteration int, sum rune) IterationLayout {
	t.Helper()
	entry, err := layout.CreateIteration(iteration)
	if err != nil {
		t.Fatal(err)
	}
	if err := entry.WriteState(validState(iteration, sum)); err != nil {
		t.Fatal(err)
	}
	if err := entry.WriteEvaluation(map[string]any{"iteration": iteration, "status": "rejected-or-accepted"}); err != nil {
		t.Fatal(err)
	}
	return entry
}

func validState(iteration int, sum rune) IterationState {
	return IterationState{
		SchemaVersion:  SchemaVersion,
		Iteration:      iteration,
		RolloutSeed:    731,
		ActorSteps:     4,
		CriticSteps:    8,
		ActorChecksum:  checksum(sum),
		CriticChecksum: checksum(sum + 1),
	}
}

func checksum(character rune) string {
	return strings.Repeat(string(character), 64)
}

func bytesContain(contents, needle []byte) bool {
	return strings.Contains(string(contents), string(needle))
}

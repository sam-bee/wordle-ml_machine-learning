package ppo

import (
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/sam-bee/wordle-ml_machine-learning/modelstate"
	"github.com/sam-bee/wordle-ml_machine-learning/policy"
	rlenv "github.com/sam-bee/wordle-ml_machine-learning/rl"
	"github.com/sam-bee/wordle-ml_machine-learning/vocabulary"
)

// TestCollectorRealVocabularyCheckpointSeedReproducibility is deliberately a
// small end-to-end integration test. It uses the full frozen vocabulary, the
// live game-engine environment (through CollectRollout), real ActorSession and
// CriticSession checkpoints, and just two complete games. It does not open the
// sealed final-test word list.
func TestCollectorRealVocabularyCheckpointSeedReproducibility(t *testing.T) {
	vocab := collectorIntegrationVocabulary(t)
	policyConfig := policy.Config{NumSolutions: vocabulary.NumSolutions, NumActions: vocabulary.NumActions}
	seedInput := collectorIntegrationOpeningInput(t, vocab)

	actorCheckpoint, criticCheckpoint := collectorIntegrationCheckpoints(t, policyConfig, seedInput)
	actorA := collectorIntegrationActor(t, policyConfig)
	defer actorA.Finalize()
	criticA := collectorIntegrationCritic(t, policyConfig)
	defer criticA.Finalize()
	if err := LoadActorCheckpoint(actorA, actorCheckpoint); err != nil {
		t.Fatal(err)
	}
	if err := LoadCriticCheckpoint(criticA, criticCheckpoint); err != nil {
		t.Fatal(err)
	}

	actorB := collectorIntegrationActor(t, policyConfig)
	defer actorB.Finalize()
	criticB := collectorIntegrationCritic(t, policyConfig)
	defer criticB.Finalize()
	if err := LoadActorCheckpoint(actorB, actorCheckpoint); err != nil {
		t.Fatal(err)
	}
	if err := LoadCriticCheckpoint(criticB, criticCheckpoint); err != nil {
		t.Fatal(err)
	}
	// Checkpoint handlers attach tensors when their graph first materializes.
	// Force that materialization before comparing the two loaded checkpoints.
	for _, session := range []struct {
		actor  *ActorSession
		critic *CriticSession
	}{
		{actor: actorA, critic: criticA},
		{actor: actorB, critic: criticB},
	} {
		if _, err := session.actor.Predict([]modelstate.Inputs{seedInput}); err != nil {
			t.Fatal(err)
		}
		if _, err := session.critic.Predict([]modelstate.Inputs{seedInput}); err != nil {
			t.Fatal(err)
		}
	}
	checksumA, err := VariableChecksum(actorA.Store, actorVariablePrefix)
	if err != nil {
		t.Fatal(err)
	}
	checksumB, err := VariableChecksum(actorB.Store, actorVariablePrefix)
	if err != nil {
		t.Fatal(err)
	}
	if checksumA != checksumB {
		t.Fatalf("actors loaded from the same checkpoint differ: %s != %s", checksumA, checksumB)
	}

	config := CollectorConfig{
		AnswerPool:    vocab.Training()[:2],
		Games:         2,
		Balanced:      true,
		ParallelGames: 2,
		Seed:          9_847,
		Gamma:         1,
		GAELambda:     0.95,
	}
	first, err := CollectRollout(vocab, actorA, criticA, actorA, config)
	if err != nil {
		t.Fatal(err)
	}
	second, err := CollectRollout(vocab, actorB, criticB, actorB, config)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(first, second) {
		t.Fatal("same actor/critic checkpoint and rollout seed produced different trajectories")
	}

	assertCompleteEngineBackedRollout(t, vocab, first)
	diagnostics, err := DiagnosePolicy(actorA, actorA, first.Transitions, 2, 0.1)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(diagnostics.MeanRatio-1) > 1e-6 || diagnostics.MaximumAbsoluteRatioFromOne > 1e-6 || diagnostics.ApproxOldPolicyKL > 1e-10 {
		t.Fatalf("pre-update stored-policy ratio diagnostics = %+v, want ratio approximately one and KL approximately zero", diagnostics)
	}
}

func collectorIntegrationCheckpoints(t *testing.T, config policy.Config, input modelstate.Inputs) (actorPath, criticPath string) {
	t.Helper()
	seedActor := collectorIntegrationActor(t, config)
	defer seedActor.Finalize()
	seedCritic := collectorIntegrationCritic(t, config)
	defer seedCritic.Finalize()
	if _, err := seedActor.Predict([]modelstate.Inputs{input}); err != nil {
		t.Fatal(err)
	}
	if _, err := seedCritic.Predict([]modelstate.Inputs{input}); err != nil {
		t.Fatal(err)
	}
	actorPath = filepath.Join(t.TempDir(), "actor")
	criticPath = filepath.Join(filepath.Dir(actorPath), "critic")
	if err := SaveStoreCheckpoint(seedActor.Store, actorPath); err != nil {
		t.Fatal(err)
	}
	if err := SaveStoreCheckpoint(seedCritic.Store, criticPath); err != nil {
		t.Fatal(err)
	}
	return actorPath, criticPath
}

func collectorIntegrationActor(t *testing.T, config policy.Config) *ActorSession {
	t.Helper()
	actor, err := NewActorSession(ActorConfig{
		Policy:                     config,
		LearningRate:               1e-5,
		ClipRange:                  0.1,
		EntropyCoefficient:         0.001,
		SupervisedReferenceKLCoeff: 0.05,
		MaximumGradientNorm:        1,
		Seed:                       7_331,
	}, sessionTestBackend)
	if err != nil {
		t.Fatal(err)
	}
	return actor
}

func collectorIntegrationCritic(t *testing.T, config policy.Config) *CriticSession {
	t.Helper()
	critic, err := NewCriticSession(CriticConfig{
		Policy:               config,
		LearningRate:         1e-4,
		ValueLossCoefficient: 0.5,
		MaximumGradientNorm:  1,
		Seed:                 7_332,
	}, sessionTestBackend)
	if err != nil {
		t.Fatal(err)
	}
	return critic
}

func collectorIntegrationVocabulary(t *testing.T) *vocabulary.Vocabulary {
	t.Helper()
	dataDir := os.Getenv("WORDLEML_DATA_DIR")
	if dataDir == "" {
		dataDir = filepath.Join("..", "..", "..", "data")
	}
	vocab, err := vocabulary.LoadWithoutFinalTest(dataDir)
	if err != nil {
		t.Fatal(err)
	}
	if len(vocab.Test()) != 0 {
		t.Fatal("collector integration test must not load final-test solutions")
	}
	return vocab
}

func collectorIntegrationOpeningInput(t *testing.T, vocab *vocabulary.Vocabulary) modelstate.Inputs {
	t.Helper()
	environment, err := rlenv.NewEnvironment(vocab, vocab.Training()[0], 0)
	if err != nil {
		t.Fatal(err)
	}
	observation, err := environment.Observation()
	if err != nil {
		t.Fatal(err)
	}
	return observation.Inputs
}

func assertCompleteEngineBackedRollout(t *testing.T, vocab *vocabulary.Vocabulary, rollout Rollout) {
	t.Helper()
	if rollout.Metrics.RolloutGames != 2 || len(rollout.Episodes) != 2 || len(rollout.Transitions) == 0 {
		t.Fatalf("bounded rollout shape = %+v, episodes=%d transitions=%d", rollout.Metrics, len(rollout.Episodes), len(rollout.Transitions))
	}
	episodes := make(map[int64]RolloutEpisode, len(rollout.Episodes))
	for _, episode := range rollout.Episodes {
		if episode.Guesses < 1 || episode.Guesses > 6 || episode.OpeningAction < 0 || episode.OpeningAction >= vocabulary.NumActions {
			t.Fatalf("invalid completed episode: %+v", episode)
		}
		episodes[episode.EpisodeID] = episode
	}
	byEpisode := make(map[int64][]TrajectoryTransition, len(episodes))
	for index, transition := range rollout.Transitions {
		if len(transition.ModelInputs.CandidateMask) != vocabulary.NumSolutions || len(transition.ModelInputs.CandidateStats) != modelstate.CandidateStatsSize || len(transition.ModelInputs.RemainingActionMask) != vocabulary.NumActions || len(transition.Availability) != vocabulary.NumActions {
			t.Fatalf("transition %d did not retain complete actor/critic inputs and exact mask", index)
		}
		if transition.ModelInputs.Turn != int32(transition.Turn) || transition.Turn < 0 || transition.Turn >= 6 {
			t.Fatalf("transition %d retained inconsistent turn input/metadata: input=%d turn=%d", index, transition.ModelInputs.Turn, transition.Turn)
		}
		if transition.ActionID < 0 || int(transition.ActionID) >= vocabulary.NumActions || transition.Availability[transition.ActionID] != 1 {
			t.Fatalf("transition %d selected unavailable action %d", index, transition.ActionID)
		}
		if !finiteIntegrationValues(transition.OldLogProb, transition.OldValue, transition.Reward, transition.Return, transition.Advantage) {
			t.Fatalf("transition %d retained non-finite PPO value: %+v", index, transition)
		}
		if _, found := episodes[transition.EpisodeID]; !found {
			t.Fatalf("transition %d has unknown episode metadata %d", index, transition.EpisodeID)
		}
		byEpisode[transition.EpisodeID] = append(byEpisode[transition.EpisodeID], transition)
	}

	var openingInputs modelstate.Inputs
	var openingMask []float32
	for episodeID, episode := range episodes {
		transitions := byEpisode[episodeID]
		if len(transitions) != episode.Guesses {
			t.Fatalf("episode %d retained %d transitions, want %d", episodeID, len(transitions), episode.Guesses)
		}
		seenActions := make(map[int32]struct{}, len(transitions))
		for turn, transition := range transitions {
			if transition.Turn != turn || transition.SolutionID != episode.SolutionID || transition.Solved != episode.Solved {
				t.Fatalf("episode %d transition %d metadata = %+v, episode = %+v", episodeID, turn, transition, episode)
			}
			for prior := range seenActions {
				if transition.Availability[prior] != 0 {
					t.Fatalf("episode %d turn %d left prior action %d available", episodeID, turn, prior)
				}
			}
			if _, duplicate := seenActions[transition.ActionID]; duplicate {
				t.Fatalf("episode %d repeated accepted action %d", episodeID, transition.ActionID)
			}
			seenActions[transition.ActionID] = struct{}{}
			wantTerminal := turn == len(transitions)-1
			if transition.Terminal != wantTerminal {
				t.Fatalf("episode %d turn %d terminal=%t, want %t", episodeID, turn, transition.Terminal, wantTerminal)
			}
			if math.Abs(transition.Reward-RewardForTransition(episode.Solved && wantTerminal, wantTerminal)) > 1e-7 {
				t.Fatalf("episode %d turn %d reward=%v does not match completed-game objective", episodeID, turn, transition.Reward)
			}
			if turn == 0 {
				if openingMask == nil {
					openingInputs = transition.ModelInputs
					openingMask = transition.Availability
				} else if !reflect.DeepEqual(openingInputs, transition.ModelInputs) || !reflect.DeepEqual(openingMask, transition.Availability) {
					t.Fatal("opening actor state varied with the environment-side hidden solution")
				}
			}
		}
	}
}

func finiteIntegrationValues(values ...float64) bool {
	for _, value := range values {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return false
		}
	}
	return true
}

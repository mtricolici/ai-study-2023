package ai

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/mtricolici/ai-study-2023/golibs/feed-forward-neural-network/backpropagation"
	"github.com/mtricolici/ai-study-2023/golibs/feed-forward-neural-network/neural_net"
	"github.com/mtricolici/ai-study-2023/golibs/snake"
)

var (
	rnd = rand.New(rand.NewSource(time.Now().UnixNano()))
)

func NewVanillaDeepQLearning(game *snake.SnakeGame, network *neural_net.FeedForwardNeuralNetwork) *VanillaDeepQLearning {
	game.Reward_apple = 1.0
	game.Reward_die = -1.0
	game.Reward_move_to_apple = 0.1
	game.Reward_move_from_apple = -0.1

	return &VanillaDeepQLearning{
		network:               network,
		game:                  game,
		LearningRate:          0.1,
		DiscountFactor:        0.9,
		InitialEpsilon:        1.0,
		FinalEpsilon:          0.01,
		max_moves_without_eat: game.Size * 4,
		// In practice, it is common to use batch sizes in the range of 32 to 256 for Deep Q-learning
		TrainBatchSize:            256,
		BackpropagationIterations: 1,
	}
}

func (ql *VanillaDeepQLearning) PredictAndMakeNextMove() {
	_, action := ql.predictNextAction()
	ql.gameChangeDirection(action)
	ql.game.NextTick()
}

func (ql *VanillaDeepQLearning) Train(numEpisodes int) {
	report := NewProgressReport()

	epsilonDecrement := (ql.InitialEpsilon - ql.FinalEpsilon) / float64(numEpisodes-1)
	epsilon := ql.InitialEpsilon

	replayMemory := NewReplayMemory()

	for i := 0; i < numEpisodes; i++ {
		// Start a new game!
		ql.game.Reset()

		score, apples := ql.playRandomGame(replayMemory, epsilon)

		report.CollectStatistics(ql.game, score, apples)

		// decrease epsilon over time
		epsilon -= epsilonDecrement

		// Print progress every 5%
		report.PrintProgress(i, numEpisodes, 5.0, epsilon)
	}

	fmt.Println("Deep-Q-Learning finished.")
}

func (ql *VanillaDeepQLearning) playRandomGame(replayMemory *ReplayMemory, epsilon float64) (float64, int) {
	sum_rewards := 0.0

	for !ql.game.GameOver {
		state, action := ql.chooseAction(epsilon)
		reward, nextState := ql.takeAction(action)

		if ql.game.Moves_since_apple >= ql.max_moves_without_eat {
			ql.game.GameOver = true
			reward = ql.game.Reward_die
		}

		replayMemory.Add(state, action, reward, nextState, ql.game.GameOver)
		ql.trainNeuralNetwork(replayMemory)

		sum_rewards += reward
	}

	return sum_rewards, ql.game.ConsumedApples
}

func (ql *VanillaDeepQLearning) chooseAction(epsilon float64) ([]float64, Action) {
	if rnd.Float64() < epsilon {
		return ql.game.GetStateForNeuralNetwork(), Action(rnd.Intn(4))
	}
	return ql.predictNextAction()
}

func (ql *VanillaDeepQLearning) takeAction(action Action) (float64, []float64) {
	ql.gameChangeDirection(action)
	ql.game.NextTick()
	reward := ql.game.Reward
	nextState := ql.game.GetStateForNeuralNetwork()

	return reward, nextState
}

func (ql *VanillaDeepQLearning) trainNeuralNetwork(replayMemory *ReplayMemory) {
	batch := replayMemory.Sample(ql.TrainBatchSize)
	if batch == nil {
		return
	}

	inputs := make([][]float64, ql.TrainBatchSize)
	targets := make([][]float64, ql.TrainBatchSize)

	// Compute the target Q-values for the batch using the Bellman equation
	for i, sample := range batch {
		inputs[i] = sample.state

		targets[i] = ql.network.Predict(sample.state)

		if sample.done {
			targets[i][int(sample.action)] = sample.reward
		} else {
			nextStateMax := args_max(ql.network.Predict(sample.newState))
			targets[i][int(sample.action)] = sample.reward + ql.DiscountFactor*nextStateMax
		}
	}

	// Train the network via back propagation
	training := backpropagation.NewBackpropagationTraining(ql.network)
	training.Verbose = false
	training.LearningRate = ql.LearningRate
	training.StopTrainingMaxAvgError = 0.00001
	training.Train(inputs, targets, ql.BackpropagationIterations)
}

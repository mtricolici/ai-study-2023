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
		FinalEpsilon:          0.1,
		EpsilonDecayRate:      0.01,
		max_moves_without_eat: 100,
		// In practice, it is common to use batch sizes in the range of 32 to 256 for Deep Q-learning
		TrainBatchSize:            32,
		BackpropagationIterations: 50,
	}
}

func (ql *VanillaDeepQLearning) PredictAndMakeNextMove() {
	_, action := ql.predictNextAction()
	ql.gameChangeDirection(action, true)
	ql.game.NextTick()
}

func (ql *VanillaDeepQLearning) Train(numEpisodes int) {
	sum_rewards := 0.0
	max_reward := 0.0
	max_apples_eaten := 0
	sum_apples_eaten := 0.0

	sum_moves_per_game := 0.0

	epsilonDecrement := (ql.InitialEpsilon - ql.FinalEpsilon) / float64(numEpisodes-1)
	epsilon := ql.InitialEpsilon

	replayMemory := NewReplayMemory()

	for i := 0; i < numEpisodes; i++ {
		// Start a new game!
		ql.game.Reset()

		score, apples := ql.playRandomGame(replayMemory, epsilon)

		sum_moves_per_game += float64(ql.game.Moves_made)

		sum_rewards += score
		sum_apples_eaten += float64(apples)

		if max_reward < score {
			max_reward = score
		}

		if max_apples_eaten < apples {
			max_apples_eaten = apples
		}

		// decrease epsilon over time
		epsilon -= epsilonDecrement

		if i > 0 && i%(numEpisodes/5) == 0 {
			avgReward := sum_rewards / float64(i) // Avg Reward for played games
			avgApples := sum_apples_eaten / float64(i)
			avgMoves := sum_moves_per_game / float64(i)
			progress := float64(i) / float64(numEpisodes) * 100.0
			fmt.Printf("%.2f%% avgScore: %f, maxScore: %f, max-apples: %d, avg-apples: %f, Avg-Moves: %f, Epsilon: %f\n",
				progress, avgReward, max_reward, max_apples_eaten, avgApples, avgMoves, epsilon)
		}
	}

	fmt.Println("Deep-Q-Learning finished.")
}

func (ql *VanillaDeepQLearning) playRandomGame(replayMemory *ReplayMemory, epsilon float64) (float64, int) {
	max_reward := 0.0

	for !ql.game.GameOver && ql.game.Moves_since_apple < ql.max_moves_without_eat {
		state, action := ql.chooseAction(epsilon)
		reward, nextState := ql.takeAction(action)

		replayMemory.Add(state, action, reward, nextState, ql.game.GameOver)
		ql.trainNeuralNetwork(replayMemory)

		if reward > max_reward {
			max_reward = reward
		}
	}

	return max_reward, ql.game.ConsumedApples
}

func (ql *VanillaDeepQLearning) chooseAction(epsilon float64) ([]float64, Action) {
	if rnd.Float64() < epsilon {
		return ql.game.GetStateForNeuralNetwork(), Action(rnd.Intn(4))
	}
	return ql.predictNextAction()
}

func (ql *VanillaDeepQLearning) takeAction(action Action) (float64, []float64) {
	ql.gameChangeDirection(action, false)
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

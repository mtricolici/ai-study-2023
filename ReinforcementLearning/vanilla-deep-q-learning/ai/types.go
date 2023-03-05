package ai

import (
	"github.com/mtricolici/ai-study-2023/golibs/feed-forward-neural-network/neural_net"
	"github.com/mtricolici/ai-study-2023/golibs/snake"
)

type Action int

const (
	TurnLeft Action = iota
	TurnRight
	TurnUp
	TurnDown
)

type VanillaDeepQLearning struct {
	game                  *snake.SnakeGame
	max_moves_without_eat int

	network *neural_net.FeedForwardNeuralNetwork

	// A higher value - agent learns faster but less stable
	LearningRate float64 // alpha

	// how much the agent values future rewards compared to immediate rewards
	DiscountFactor float64 // gama

	InitialEpsilon   float64
	FinalEpsilon     float64
	EpsilonDecayRate float64

	TrainBatchSize int
}

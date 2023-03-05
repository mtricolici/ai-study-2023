package ai

import "github.com/mtricolici/ai-study-2023/golibs/snake"

type Action int

const (
	TurnLeft Action = iota
	TurnRight
	TurnUp
	TurnDown
)

type StateActionReward struct {
	State  string
	Action Action
	Reward float64
}

type QTable map[string]map[Action]float64

type MonteCarlo struct {
	qtable QTable
	game   *snake.SnakeGame

	// A higher value - agent learns faster but less stable
	LearningRate float64 // alpha

	// how much the agent values future rewards compared to immediate rewards
	DiscountFactor float64 // gama

	InitialEpsilon   float64
	FinalEpsilon     float64
	EpsilonDecayRate float64

	max_moves_without_eat int
}

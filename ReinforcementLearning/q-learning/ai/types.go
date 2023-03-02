package ai

import "qlsample/snake"

type Action int

const (
	TurnLeft Action = iota
	TurnRight
	TurnUp
	TurnDown
)

type QTable map[string]map[Action]float64

type QLearning struct {
	qtable             QTable
	game               *snake.SnakeGame
	learningRate       float64
	max_moves_per_game int
}

package ai

import "qlsample/snake"

type Action int

const (
	ContinueTheSame Action = iota
	TurnLeft
	TurnRight
)

type QTable map[string]map[Action]float64

type QLearning struct {
	qtable             QTable
	game               *snake.SnakeGame
	alpha, gamma       float64
	max_moves_per_game int
}

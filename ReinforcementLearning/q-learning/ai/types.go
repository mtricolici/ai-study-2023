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
	qtable                QTable
	game                  *snake.SnakeGame
	alpha, gamma, epsilon float64
}

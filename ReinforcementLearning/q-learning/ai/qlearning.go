package ai

import (
	"math"
	"math/rand"
	"qlsample/snake"
)

type Action int

const (
	Up Action = iota
	Down
	Left
	Right
)

type QTable map[string]map[Action]float64

func TrainQTable(game *snake.SnakeGame, alpha float64, gamma float64, epsilon float64, iterations int) QTable {
	qTable := make(QTable)

	for i := 0; i < iterations; i++ {
		state := getState(game)

		for !game.GameOver {
			var action Action

			if rand.Float64() < epsilon {
				action = Action(rand.Intn(4))
			} else {
				maxQ := math.Inf(-1)

				for a := Up; a <= Right; a++ {
					if qTable[state][a] > maxQ {
						maxQ = qTable[state][a]
						action = a
					}
				}
			}

			game.ChangeDirection(getDirection(action))
			game.NextTick()
			reward := game.Score

			if reward >= 0 {
				newState := getState(game)

				if qTable[newState] == nil {
					qTable[newState] = make(map[Action]float64)
				}

				maxQ := math.Inf(-1)

				for a := Up; a <= Right; a++ {
					if qTable[newState][a] > maxQ {
						maxQ = qTable[newState][a]
					}
				}

				qTable[state][action] = qTable[state][action] + alpha*(reward+gamma*maxQ-qTable[state][action])
				state = newState
			}
		}

		game.Reset()

		// Reduce epsilon over time
		epsilon *= 0.99
	}

	return qTable
}

func PlayQTableNextMove(game *snake.SnakeGame, qTable QTable) {
	state := getState(game)

	maxQ := math.Inf(-1)
	var action Action

	for a := Up; a <= Right; a++ {
		if qTable[state][a] > maxQ {
			maxQ = qTable[state][a]
			action = a
		}
	}

	game.ChangeDirection(getDirection(action))
}

package ai

import (
	"fmt"
	"math"
	"qlsample/snake"
)

func NewQLearning(game *snake.SnakeGame) *QLearning {
	q := QLearning{
		qtable: make(QTable),
		game:   game,
	}

	return &q
}

func (ql *QLearning) PredictNextTurn() {
	state := ql.game.GetState()
	action, _ := ql.getMaxQValue(state)
	ql.gameChangeDirection(action)
}

func (ql *QLearning) gameChangeDirection(action Action) {
	switch action {
	case TurnLeft:
		ql.game.TurnLeft()
	case TurnRight:
		ql.game.TurnRight()
	case ContinueTheSame: // nothing to be done
	}
}

func (ql *QLearning) Train(alpha, gamma, epsilon float64, iterations int) {
	sum_rewards := 0.0

	for i := 0; i < iterations; i++ {
		// Start a new game!
		ql.game.Reset()

		max_reward := ql.playRandomGame(alpha, gamma, epsilon)
		sum_rewards += max_reward

		// Reduce epsilon over time
		epsilon *= 0.99

		if i > 0 && i%(iterations/10) == 0 {
			progress := float64(i) / float64(iterations) * 100.0
			avgReward := sum_rewards / float64(i) // Avg Reward for played games
			fmt.Printf("Training %.2f%% avgScore: %f\n", progress, avgReward)
		}
	}
}

func (ql *QLearning) playRandomGame(alpha, gamma, epsilon float64) float64 {
	state := ql.game.GetState()

	max_reward := 0.0

	for !ql.game.GameOver {
		var action Action

		if rnd.Float64() < epsilon {
			action = Action(rnd.Intn(3))
		} else {
			action, _ = ql.getMaxQValue(state)
		}

		ql.gameChangeDirection(action)
		ql.game.NextTick()
		reward := ql.game.Score

		if reward > max_reward {
			max_reward = reward
		}

		if reward >= 0 {
			newState := ql.game.GetState()
			_, maxQ := ql.getMaxQValue(newState)

			ql.qtable[state][action] = ql.qtable[state][action] + alpha*(reward+gamma*maxQ-ql.qtable[state][action])
			state = newState
		}
	}

	return max_reward
}

func (ql *QLearning) getMaxQValue(state string) (Action, float64) {
	maxQ := math.Inf(-1)
	bestAction := ContinueTheSame // Do not change direction - default one

	for action, reward := range ql.qtable[state] {
		if reward > maxQ {
			maxQ = reward
			bestAction = action
		}
	}

	return bestAction, maxQ
}

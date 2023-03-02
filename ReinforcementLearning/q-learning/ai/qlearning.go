package ai

import (
	"fmt"
	"math"
	"math/rand"
	"qlsample/snake"
	"time"
)

var (
	rnd = rand.New(rand.NewSource(time.Now().UnixNano()))
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
	max_reward := 0.0

	for i := 0; i < iterations; i++ {
		// Start a new game!
		ql.game.Reset()

		reward := ql.playRandomGame(alpha, gamma, epsilon)
		sum_rewards += reward
		if max_reward < reward {
			max_reward = reward
		}

		// Reduce epsilon over time
		epsilon *= 0.99

		if i > 0 && i%(iterations/10) == 0 {
			progress := float64(i) / float64(iterations) * 100.0
			avgReward := sum_rewards / float64(i) // Avg Reward for played games
			fmt.Printf("Training %.2f%% avgScore: %f, maxScore: %f\n", progress, avgReward, max_reward)
		}
	}
}

func (ql *QLearning) playRandomGame(alpha, gamma, epsilon float64) float64 {
	state := ql.game.GetState()

	max_reward := 0.0

	for !ql.game.GameOver {
		var action Action

		ql.checkStatePresence(state)

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

		newState := ql.game.GetState()
		_, maxQ := ql.getMaxQValue(newState)
		ql.checkStatePresence(newState)

		ql.qtable[state][action] = ql.qtable[state][action] + alpha*(reward+gamma*maxQ-ql.qtable[state][action])
		state = newState
	}

	return max_reward
}

func (ql *QLearning) getMaxQValue(state string) (Action, float64) {
	max_q := math.Inf(-1)
	bestAction := ContinueTheSame // Do not change direction - default one

	for action, q := range ql.qtable[state] {
		if q > max_q {
			max_q = q
			bestAction = action
		}
	}

	return bestAction, max_q
}

func (ql *QLearning) checkStatePresence(state string) {
	if _, ok := ql.qtable[state]; !ok {
		ql.qtable[state] = make(map[Action]float64)
		// initialize with 0.0
		ql.qtable[state][ContinueTheSame] = 0.0
		ql.qtable[state][TurnLeft] = 0.0
		ql.qtable[state][TurnRight] = 0.0
	}
}

package ai

import (
	"fmt"
	"math/rand"
	"qlsample/snake"
	"time"
)

var (
	rnd = rand.New(rand.NewSource(time.Now().UnixNano()))
)

func NewQLearning(game *snake.SnakeGame, alpha, gamma, epsilon float64) *QLearning {
	q := QLearning{
		qtable:  make(QTable),
		game:    game,
		alpha:   alpha,
		gamma:   gamma,
		epsilon: epsilon,
	}

	return &q
}

func (ql *QLearning) PredictNextTurn() {
	state := ql.game.GetState()
	action, _ := ql.getMaxQValue(state)
	ql.gameChangeDirection(action)
	switch action {
	case TurnLeft:
		fmt.Println("-- predict: turn LEFT")
	case TurnRight:
		fmt.Println("-- predict: turn RIGHT")
	case ContinueTheSame:
		fmt.Println("-- predict: continue")
	default:
		panic("predict: UNKNOWN action detected")
	}
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

func (ql *QLearning) Train(iterations int) {
	sum_rewards := 0.0
	max_reward := 0.0

	for i := 0; i < iterations; i++ {
		// Start a new game!
		ql.game.Reset()

		score := ql.playRandomGame()
		sum_rewards += score
		if max_reward < score {
			max_reward = score
		}

		// Reduce epsilon over time
		ql.epsilon *= 0.99

		if i > 0 && i%(iterations/5) == 0 {
			progress := float64(i) / float64(iterations) * 100.0
			avgReward := sum_rewards / float64(i) // Avg Reward for played games
			fmt.Printf("Training %.2f%% avgScore: %f, maxScore: %f\n", progress, avgReward, max_reward)
		}
	}
}

func (ql *QLearning) playRandomGame() float64 {
	state := ql.game.GetState()

	max_reward := 0.0

	for !ql.game.GameOver {
		var action Action

		ql.checkStatePresence(state)

		if rnd.Float64() < ql.epsilon {
			action = Action(rnd.Intn(3))
		} else {
			action, _ = ql.getMaxQValue(state)
		}

		ql.gameChangeDirection(action)
		ql.game.NextTick()
		reward := ql.game.Score

		newState := ql.game.GetState()
		ql.checkStatePresence(newState)

		_, maxQ := ql.getMaxQValue(newState)

		ql.updateQValue(state, action, reward, maxQ)

		state = newState
		if reward > max_reward {
			max_reward = reward
		}
	}

	return max_reward
}

func (ql *QLearning) updateQValue(state string, action Action, reward, maxq float64) {
	v := ql.qtable[state][action]
	v += ql.alpha * (reward + ql.gamma*maxq - v)

	ql.qtable[state][action] = v
}

func (ql *QLearning) getMaxQValue(state string) (Action, float64) {
	max_q := -1.0                 //math.Inf(-1)
	bestAction := ContinueTheSame // Do not change direction - default one

	//qtable[state] can be nil. i.e. a new total state!
	ql.checkStatePresence(state)

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

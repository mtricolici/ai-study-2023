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

func NewQLearning(game *snake.SnakeGame, alpha, gamma float64) *QLearning {
	q := QLearning{
		qtable:             make(QTable),
		game:               game,
		alpha:              alpha,
		gamma:              gamma,
		max_moves_per_game: 100,
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
	max_apples_eaten := 0
	sum_apples_eaten := 0.0

	for i := 0; i < iterations; i++ {
		// Start a new game!
		ql.game.Reset()

		progress := float64(i) / float64(iterations) * 100.0
		epsilon := ql.GetRandomRate(progress)

		score, apples := ql.playRandomGame(epsilon)
		sum_rewards += score
		sum_apples_eaten += float64(apples)

		if max_reward < score {
			max_reward = score
		}

		if max_apples_eaten < apples {
			max_apples_eaten = apples
		}

		if i > 0 && i%(iterations/5) == 0 {
			avgReward := sum_rewards / float64(i) // Avg Reward for played games
			avgApples := sum_apples_eaten / float64(i)
			fmt.Printf("%.2f%% avgScore: %f, maxScore: %f, max-apples: %d, avg-apples: %f. States:%d\n",
				progress, avgReward, max_reward, max_apples_eaten, avgApples, len(ql.qtable))
		}
	}

	fmt.Printf("Q-Learning finished. States: %d\n", len(ql.qtable))
}

func (ql *QLearning) playRandomGame(epsilon float64) (float64, int) {
	state := ql.game.GetState()

	max_reward := 0.0

	for !ql.game.GameOver && ql.game.Moves_made < ql.max_moves_per_game {
		var action Action

		ql.checkStatePresence(state)

		if rnd.Float64() < epsilon {
			action = Action(rnd.Intn(3))
		} else {
			action, _ = ql.getMaxQValue(state)
		}

		ql.gameChangeDirection(action)
		ql.game.NextTick()
		reward := ql.game.Reward

		newState := ql.game.GetState()
		ql.checkStatePresence(newState)

		_, maxQ := ql.getMaxQValue(newState)

		ql.updateQValue(state, action, reward, maxQ)

		state = newState
		if reward > max_reward {
			max_reward = reward
		}
	}

	return max_reward, ql.game.ConsumedApples
}

func (ql *QLearning) updateQValue(state string, action Action, reward, maxq float64) {
	// v := ql.qtable[state][action]
	// v += ql.alpha * (reward + ql.gamma*maxq - v)

	ql.qtable[state][action] = reward + ql.gamma*maxq
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

func (ql *QLearning) GetRandomRate(progress float64) float64 {
	if progress < 5 {
		return 1.0 // exploration 100%
	}

	if progress < 20 {
		return 0.5 // 50% exploration
	}

	if progress < 80 {
		return 0.3
	}

	return 0.1
}

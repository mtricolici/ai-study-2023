package ai

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/mtricolici/ai-study-2023/golibs/snake"
)

var (
	rnd = rand.New(rand.NewSource(time.Now().UnixNano()))
)

func NewMonteCarlo(game *snake.SnakeGame) *MonteCarlo {
	q := MonteCarlo{
		qtable:             make(QTable),
		game:               game,
		LearningRate:       0.1,
		DiscountFactor:     0.9,
		InitialEpsilon:     1.0,
		FinalEpsilon:       0.1,
		EpsilonDecayRate:   0.01,
		max_moves_per_game: 200,
	}

	return &q
}

func (mc *MonteCarlo) PredictAndMakeNextMove() {
	state := mc.game.GetState()
	action, _ := mc.getMaxQValue(state)
	mc.gameChangeDirection(action, true)
	mc.game.NextTick()
}

func (mc *MonteCarlo) Train(numEpisodes int) {
	sum_rewards := 0.0
	max_reward := 0.0
	max_apples_eaten := 0
	sum_apples_eaten := 0.0

	sum_moves_per_game := 0.0

	epsilon := mc.InitialEpsilon

	for episode := 0; episode < numEpisodes; episode++ {
		// Start a new game!
		mc.game.Reset()

		score, apples := mc.playRandomGame(epsilon)

		sum_moves_per_game += float64(mc.game.Moves_made)

		sum_rewards += score
		sum_apples_eaten += float64(apples)

		if max_reward < score {
			max_reward = score
		}

		if max_apples_eaten < apples {
			max_apples_eaten = apples
		}

		// decrease epsilon over time
		epsilon = mc.calculateEpsilon(episode)

		if episode > 0 && episode%(numEpisodes/5) == 0 {
			avgReward := sum_rewards / float64(episode) // Avg Reward for played games
			avgApples := sum_apples_eaten / float64(episode)
			avgMoves := sum_moves_per_game / float64(episode)
			progress := float64(episode) / float64(numEpisodes) * 100.0
			fmt.Printf("%.2f%% avgScore: %f, maxScore: %f, max-apples: %d, avg-apples: %f. States:%d, Avg-Moves: %f\n",
				progress, avgReward, max_reward, max_apples_eaten, avgApples, len(mc.qtable), avgMoves)
		}
	}

	fmt.Printf("MonteCarlo finished. States: %d\n", len(mc.qtable))
}

func (mc *MonteCarlo) playRandomGame(epsilon float64) (float64, int) {
	max_reward := 0.0

	state := mc.game.GetState()

	history := make([]StateActionReward, 0)

	// Play the game until over or max moves
	for !mc.game.GameOver && mc.game.Moves_made < mc.max_moves_per_game {
		mc.checkStatePresence(state)
		action := mc.chooseAction(state, epsilon)
		reward, nextState := mc.takeAction(action)

		history = append(history, StateActionReward{State: state, Action: action, Reward: reward})

		state = nextState
		if reward > max_reward {
			max_reward = reward
		}
	}

	// Update the qTable based on history !
	for i, h := range history {
		g := 0.0
		for j := i; j < len(history); j++ {
			g = mc.DiscountFactor*g + history[j].Reward
		}

		mc.updateQValue(h.State, h.Action, g)
	}

	return max_reward, mc.game.ConsumedApples
}

func (mc *MonteCarlo) updateQValue(state string, action Action, g float64) {
	mc.qtable[state][action] += mc.LearningRate * (g - mc.qtable[state][action])
}

func (mc *MonteCarlo) chooseAction(state string, epsilon float64) Action {
	if rnd.Float64() < epsilon {
		return Action(rnd.Intn(4))
	}
	bestAction, _ := mc.getMaxQValue(state)
	return bestAction
}

func (mc *MonteCarlo) takeAction(action Action) (float64, string) {
	mc.gameChangeDirection(action, false)
	mc.game.NextTick()
	reward := mc.game.Reward
	nextState := mc.game.GetState()

	return reward, nextState
}

func (ql *MonteCarlo) getMaxQValue(state string) (Action, float64) {
	max_q := math.Inf(-1)
	var bestAction Action

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

package ai

import (
	"encoding/gob"
	"fmt"
	"math"
	"os"

	"github.com/mtricolici/ai-study-2023/golibs/snake"
)

func (mc *MonteCarlo) calculateEpsilon(episode int) float64 {
	return mc.FinalEpsilon + (mc.InitialEpsilon-mc.FinalEpsilon)*math.Exp(-mc.EpsilonDecayRate*float64(episode))
}

func (mc *MonteCarlo) SaveToFile(fileName string) {
	fmt.Printf("MonteCarlo qtable save to '%s' ...\n", fileName)

	file, err := os.Create(fileName)
	if err != nil {
		panic("Failed to save QTable to file :" + err.Error())
	}
	defer file.Close()

	encoder := gob.NewEncoder(file)

	// Encode and save QTable to a file
	if err := encoder.Encode(mc.qtable); err != nil {
		panic("Failed to encode brain:" + err.Error())
	}
}

func (mc *MonteCarlo) LoadFromFile(fileName string) {
	fmt.Printf("MonteCarlo qtable load from '%s' ...\n", fileName)

	file, err := os.Open(fileName)
	if err != nil {
		panic("Failed to load QTable from file :" + err.Error())
	}
	defer file.Close()

	decoder := gob.NewDecoder(file)

	// Decode QTable from file
	if err := decoder.Decode(&mc.qtable); err != nil {
		panic("Failed to decode brain from file:" + err.Error())
	}
}

func (mc *MonteCarlo) checkStatePresence(state string) {
	if _, ok := mc.qtable[state]; !ok {
		mc.qtable[state] = make(map[Action]float64)
		// initialize with 0.0
		mc.qtable[state][TurnLeft] = 0.0
		mc.qtable[state][TurnRight] = 0.0
		mc.qtable[state][TurnUp] = 0.0
		mc.qtable[state][TurnDown] = 0.0
	}
}

func (mc *MonteCarlo) gameChangeDirection(action Action, debug bool) {
	switch action {
	case TurnLeft:
		mc.game.ChangeDirection(snake.Left)
		if debug {
			fmt.Println("-- predict: turn LEFT")
		}

	case TurnRight:
		mc.game.ChangeDirection(snake.Right)
		if debug {
			fmt.Println("-- predict: turn RIGHT")
		}

	case TurnDown:
		mc.game.ChangeDirection(snake.Down)
		if debug {
			fmt.Println("-- predict: turn Down")
		}

	case TurnUp:
		mc.game.ChangeDirection(snake.Up)
		if debug {
			fmt.Println("-- predict: turn Down")
		}

	default:
		panic("predict: UNKNOWN action detected")
	}
}

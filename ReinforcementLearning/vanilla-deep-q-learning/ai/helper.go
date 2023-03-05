package ai

import (
	"encoding/gob"
	"fmt"
	"os"

	"github.com/mtricolici/ai-study-2023/golibs/snake"
)

func (ql *VanillaDeepQLearning) SaveToFile(fileName string) {
	fmt.Printf("DeepQLearning save network to '%s' ...\n", fileName)

	file, err := os.Create(fileName)
	if err != nil {
		panic("Failed to save QTable to file :" + err.Error())
	}
	defer file.Close()

	encoder := gob.NewEncoder(file)

	// Encode and save QTable to a file
	if err := encoder.Encode(ql.network); err != nil {
		panic("Failed to encode brain:" + err.Error())
	}
}

func (ql *VanillaDeepQLearning) LoadFromFile(fileName string) {
	fmt.Printf("DeepQLearning load network from '%s' ...\n", fileName)

	file, err := os.Open(fileName)
	if err != nil {
		panic("Failed to load QTable from file :" + err.Error())
	}
	defer file.Close()

	decoder := gob.NewDecoder(file)

	// Decode QTable from file
	if err := decoder.Decode(&ql.network); err != nil {
		panic("Failed to decode brain from file:" + err.Error())
	}
}

func (ql *VanillaDeepQLearning) predictNextAction() ([]float64, Action) {
	state := ql.game.GetStateForNeuralNetwork()
	actions := ql.network.Predict(state)
	idx := args_max_index(actions)
	return state, Action(idx)
}

func (ql *VanillaDeepQLearning) gameChangeDirection(action Action, debug bool) {
	switch action {
	case TurnLeft:
		ql.game.ChangeDirection(snake.Left)
		if debug {
			fmt.Println("-- predict: turn LEFT")
		}

	case TurnRight:
		ql.game.ChangeDirection(snake.Right)
		if debug {
			fmt.Println("-- predict: turn RIGHT")
		}

	case TurnDown:
		ql.game.ChangeDirection(snake.Down)
		if debug {
			fmt.Println("-- predict: turn Down")
		}

	case TurnUp:
		ql.game.ChangeDirection(snake.Up)
		if debug {
			fmt.Println("-- predict: turn Down")
		}

	default:
		panic("predict: UNKNOWN action detected")
	}
}

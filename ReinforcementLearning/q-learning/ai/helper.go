package ai

import (
	"encoding/gob"
	"fmt"
	"os"
)

func (ql *QLearning) SaveToFile(fileName string) {
	fmt.Printf("QLearning save to '%s' ...\n", fileName)

	file, err := os.Create(fileName)
	if err != nil {
		panic("Failed to save QTable to file :" + err.Error())
	}
	defer file.Close()

	encoder := gob.NewEncoder(file)

	// Encode and save QTable to a file
	if err := encoder.Encode(ql.qtable); err != nil {
		panic("Failed to encode brain:" + err.Error())
	}
}

func (ql *QLearning) LoadFromFile(fileName string) {
	fmt.Printf("QLearning load from '%s' ...\n", fileName)

	file, err := os.Open(fileName)
	if err != nil {
		panic("Failed to load QTable from file :" + err.Error())
	}
	defer file.Close()

	decoder := gob.NewDecoder(file)

	// Decode QTable from file
	if err := decoder.Decode(&ql.qtable); err != nil {
		panic("Failed to decode brain from file:" + err.Error())
	}
}

package ai

import "fmt"

func (ql *QLearning) SaveToFile(fileName string) {
	fmt.Printf("QLearning save to '%s' ...\n", fileName)
}

func (ql *QLearning) LoadFromFile(fileName string) {
	fmt.Printf("QLearning load from '%s' ...\n", fileName)
}

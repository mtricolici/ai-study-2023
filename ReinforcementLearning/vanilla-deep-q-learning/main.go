package main

import (
	"fmt"
	"os"
	"time"
	"vanilla_deep_qlearn/ai"
	"vanilla_deep_qlearn/cimport"

	"github.com/mtricolici/ai-study-2023/golibs/feed-forward-neural-network/neural_net"
	"github.com/mtricolici/ai-study-2023/golibs/snake"
)

func createGame() *snake.SnakeGame {
	g := snake.NewSnakeGame(8, true)
	g.Small_State_for_Neural = false // See FULL board
	return g
}

func createNetwork(game *snake.SnakeGame) *neural_net.FeedForwardNeuralNetwork {
	inputSize := len(game.GetStateForNeuralNetwork())
	hiddenNeurons := 100
	outputSize := 4 // left, right, up, down
	network := neural_net.NewFeedForwardNeuralNetwork([]int{inputSize, hiddenNeurons, outputSize})
	return network
}

func invoke_DeepQLearning(numberOfGamesToPlay int, saveFileName string) {
	g := createGame()
	network := createNetwork(g)
	bot := ai.NewVanillaDeepQLearning(g, network)

	bot.Train(numberOfGamesToPlay)

	fmt.Println("Training finished!")
	bot.SaveToFile(saveFileName)
	fmt.Println("bye bye")
}

func play_DemoGame(brainFileLocation string) {
	g := createGame()
	network := createNetwork(g)
	bot := ai.NewVanillaDeepQLearning(g, network)
	bot.LoadFromFile(brainFileLocation)

	cimport.Create_game(g.Size)
	cimport.X_create_window()

	cimport.UpdateGameData(g)
	time.Sleep(1 * time.Second)
	cimport.X_draw_objects()
	fmt.Println("Game starts in 10 seconds... prepare video recorder! ;)")
	// time.Sleep(10 * time.Second)
	fmt.Println("Game starts in 3 seconds... prepare video recorder! ;)")
	time.Sleep(3 * time.Second)

	for !g.GameOver {
		bot.PredictAndMakeNextMove()
		cimport.UpdateGameData(g)
		cimport.X_draw_objects()
		fmt.Printf("Direction: %s Reward: %f\n", g.GetDirectionAsString(), g.Reward)
		time.Sleep(100 * time.Millisecond)
	}

	fmt.Println("Game over")
	time.Sleep(1 * time.Second)
	cimport.Destroy_game()
}

func show_usage() {
	fmt.Println("Bad argument. Please use 'train' or 'demo' argument")
}

func main() {
	args := os.Args[1:] // Skip the first argument

	if len(args) != 1 {
		show_usage()
		return
	}

	switch args[0] {
	case "train":
		invoke_DeepQLearning(5_000, "/home/boris/temp/deepbrain.zzz")
	case "demo":
		play_DemoGame("/home/boris/temp/deepbrain.zzz")
	default:
		show_usage()
	}
}

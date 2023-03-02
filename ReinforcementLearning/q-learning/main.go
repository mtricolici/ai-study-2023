package main

import (
	"fmt"
	"os"
	"qlsample/ai"
	"qlsample/cimport"
	"qlsample/snake"
	"time"
)

func invoke_QLearning(numberOfGamesToPlay int, saveFileName string) {
	// train on very small tables 6x6
	g := snake.NewSnakeGame(6, false)
	bot := ai.NewQLearning(g, 0.1)

	fmt.Printf("Training AI for %d games ...\n", numberOfGamesToPlay)
	bot.Train(numberOfGamesToPlay)

	fmt.Println("Training finished!")
	bot.SaveToFile(saveFileName)
	fmt.Println("bye bye")
}

func play_DemoGame(brainFileLocation string) {
	// demo on a biggger table ;)
	g := snake.NewSnakeGame(20, false)

	cimport.Create_game(g.Size)
	cimport.X_create_window()

	bot := ai.NewQLearning(g, 0.0) // Learning rate is not needed here
	bot.LoadFromFile(brainFileLocation)

	g.Reset()
	cimport.UpdateGameData(g)

	for !g.GameOver {
		bot.PredictNextTurn()
		g.NextTick()

		cimport.UpdateGameData(g)
		cimport.X_draw_objects()
		fmt.Printf("state: '%s'\n", g.GetState())
		fmt.Printf("Direction: %s Reward: %f\n", g.GetDirectionAsString(), g.Reward)
		time.Sleep(100 * time.Millisecond)
	}

	fmt.Println("Game over")
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
		invoke_QLearning(50_000_000, "/tmp/brain.zzz")
	case "demo":
		play_DemoGame("/tmp/brain.zzz")
	default:
		show_usage()
	}
}

package main

import (
	"fmt"
	"montecarlo/ai"
	"montecarlo/cimport"
	"os"
	"time"

	"github.com/mtricolici/ai-study-2023/golibs/snake"
)

func invoke_MonteCarlo(numberOfGamesToPlay int, saveFileName string) {
	// train on very small tables 8x8
	g := snake.NewSnakeGame(10, false)
	bot := ai.NewMonteCarlo(g)

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

	bot := ai.NewMonteCarlo(g)
	bot.LoadFromFile(brainFileLocation)

	g.Reset()
	cimport.UpdateGameData(g)
	time.Sleep(1 * time.Second)
	cimport.X_draw_objects()
	fmt.Println("Game starts in 10 seconds... prepare video recorder! ;)")
	time.Sleep(10 * time.Second)

	for !g.GameOver {
		bot.PredictAndMakeNextMove()

		cimport.UpdateGameData(g)
		cimport.X_draw_objects()
		fmt.Printf("state: '%s'\n", g.GetState())
		fmt.Printf("Direction: %s Reward: %f\n", g.GetDirectionAsString(), g.Reward)
		time.Sleep(100 * time.Millisecond)
	}

	fmt.Println("Game over")
	time.Sleep(10 * time.Second)
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
		invoke_MonteCarlo(5_000_000, "/tmp/brain.zzz")
	case "demo":
		play_DemoGame("/tmp/brain.zzz")
	default:
		show_usage()
	}
}

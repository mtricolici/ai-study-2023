package main

import (
	"fmt"
	"qlsample/ai"
	"qlsample/cimport"
	"qlsample/snake"
	"time"
)

var (
	alpha      = 0.1 // bigger value means QTable is updated faster
	gamma      = 0.9
	iterations = 50_000_000 // Number of games to train
)

func invoke_QLearning() *ai.QLearning {
	// train on very small tables 6x6
	g := snake.NewSnakeGame(6, false)
	bot := ai.NewQLearning(g, alpha, gamma)

	fmt.Printf("Training AI for %d games ...\n", iterations)
	bot.Train(iterations)

	fmt.Println("Training finished! Let's play a game ;)")

	return bot
}

func play_DemoGame(bot *ai.QLearning) {
	// demo on a biggger table ;)
	g := snake.NewSnakeGame(20, false)

	cimport.Create_game(g.Size)
	cimport.X_create_window()

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

func main() {
	bot := invoke_QLearning()
	play_DemoGame(bot)
}

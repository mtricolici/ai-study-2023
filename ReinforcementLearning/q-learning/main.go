package main

import (
	"fmt"
	"qlsample/ai"
	"qlsample/snake"
	"time"
)

var (
	alpha      = 0.2 // bigger value means QTable is updated faster
	gamma      = 0.9
	epsilon    = 0.1       // how often a RANDOM action is invoked. 1 means ALWAYS
	iterations = 5_000_000 // Number of games to train
)

func main() {
	fmt.Println("Hello world")

	snake.Create_game(10)

	g := snake.NewSnakeGame(10, false)
	ai := ai.NewQLearning(g)

	fmt.Printf("Training AI for %d games ...\n", iterations)
	ai.Train(alpha, gamma, epsilon, iterations)

	fmt.Println("Training finished! Let's play a game ;)")

	snake.X_create_window()

	g.Reset()
	snake.UpdateGameData(g)

	for !g.GameOver {
		// AI decides to turn Left or Right or keep the same direction
		ai.PredictNextTurn()
		g.NextTick()
		snake.UpdateGameData(g)
		snake.X_draw_objects()
		time.Sleep(500 * time.Millisecond)
	}

	fmt.Println("bye bye!")
	snake.Destroy_game()
}

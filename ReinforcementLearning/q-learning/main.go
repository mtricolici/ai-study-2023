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
	epsilon    = 0.1
	iterations = 5_000_000 // Number of games to train
)

func main() {
	fmt.Println("Hello world")

	snake.Create_game(10)

	g := snake.NewSnakeGame(10)

	fmt.Printf("Training AI for %d games ...\n", iterations)
	qtable := ai.TrainQTable(g, alpha, gamma, epsilon, iterations)

	fmt.Println("Training finished! Let's play a game ;)")

	snake.UpdateGameData(g)

	snake.X_create_window()

	g.Reset()
	for !g.GameOver {
		ai.PlayQTableNextMove(g, qtable)
		g.NextTick()
		snake.UpdateGameData(g)
		snake.X_draw_objects()
		time.Sleep(1 * time.Second)
	}

	fmt.Println("End of program")

	time.Sleep(500 * time.Millisecond)

	snake.Destroy_game()
}

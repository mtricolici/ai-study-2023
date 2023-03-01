package main

import (
	"fmt"
	"qlsample/ai"
	"qlsample/snake"
	"time"
)

var (
	alpha      = 0.1 // bigger value means QTable is updated faster
	gamma      = 0.9
	epsilon    = 0.1
	iterations = 5000 // Number of games to train
)

func main() {
	fmt.Println("Hello world")

	snake.Create_game(20)

	g := snake.NewSnakeGame(20)

	qtable := ai.TrainQTable(g, alpha, gamma, epsilon, iterations)

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

	time.Sleep(2 * time.Second)

	snake.Destroy_game()
}

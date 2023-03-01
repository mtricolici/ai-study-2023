package main

import (
	"fmt"
	"qlsample/snake"
	"time"
)

func main() {
	fmt.Println("Hello world")

	snake.Create_game(20)

	g := snake.NewSnakeGame(20)

	snake.UpdateGameData(g)

	snake.X_create_window()

	time.Sleep(1 * time.Second)

	for !g.GameOver {
		g.NextTick()
		snake.UpdateGameData(g)
		snake.X_draw_objects()
		time.Sleep(1 * time.Second)
	}

	fmt.Println("End of program")

	time.Sleep(2 * time.Second)

	snake.Destroy_game()
}

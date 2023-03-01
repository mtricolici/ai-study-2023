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
	time.Sleep(3 * time.Second)

	//snake.Set_game_data(14, 15, 3) // new apple
	//snake.X_draw_objects()
	//time.Sleep(10 * time.Second)

	snake.Destroy_game()
}

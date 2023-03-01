package main

import (
	"fmt"
	"qlsample/snake"
	"time"
)

func main() {
	fmt.Println("Hello world")

	snake.Create_game(20)
	snake.Set_game_data(10, 5, 1)  // head
	snake.Set_game_data(9, 5, 2)   //body
	snake.Set_game_data(8, 5, 2)   //body
	snake.Set_game_data(13, 13, 3) // apple

	snake.X_create_window()
	time.Sleep(2 * time.Second)

	snake.Set_game_data(14, 15, 3) // new apple
	snake.X_draw_objects()
	time.Sleep(10 * time.Second)

	snake.Destroy_game()
}

package ai

import (
	"fmt"
	"qlsample/snake"
)

func getState(game *snake.SnakeGame) string {
	return fmt.Sprintf("%v|%v|%v", game.Body, game.Apple, game.Direction)
}

func getDirection(action Action) snake.Direction {
	switch action {
	case Up:
		return snake.Up
	case Down:
		return snake.Down
	case Left:
		return snake.Left
	}
	return snake.Right
}

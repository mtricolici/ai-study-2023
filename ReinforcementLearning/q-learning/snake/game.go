package snake

import (
	"fmt"
	"math/rand"
	"time"
)

type Position struct {
	X int
	Y int
}

type Direction int

const (
	Up Direction = iota
	Down
	Left
	Right
)

var (
	rnd = rand.New(rand.NewSource(time.Now().UnixNano()))
)

type SnakeGame struct {
	Body      []Position
	Apple     Position
	Size      int
	Direction Direction
}

func NewSnakeGame(size int) *SnakeGame {
	game := SnakeGame{
		Size: size,
	}

	game.generateRandomSnakeBody()
	game.generateRandomApple()

	return &game
}

func (sn *SnakeGame) generateRandomSnakeBody() {
	mlt := 1
	var startX int

	if rnd.Intn(2) == 0 {
		sn.Direction = Left
		startX = 3 + rnd.Intn(sn.Size-4)
		fmt.Println("new direction: LEFT")
	} else {
		sn.Direction = Right
		startX = rnd.Intn(sn.Size - 4)
		fmt.Println("new direction: RIGHT")
		mlt = -1
	}

	startY := rnd.Intn(sn.Size)

	sn.Body = []Position{
		{X: startX, Y: startY},
		{X: startX + mlt*1, Y: startY},
		{X: startX + mlt*2, Y: startY},
	}
}

func (sn *SnakeGame) generateRandomApple() {

	for {
		x := rnd.Intn(sn.Size)
		y := rnd.Intn(sn.Size)
		position_ocupied := false
		for _, p := range sn.Body {
			if p.X == x && p.Y == y {
				position_ocupied = true
				break
			}
		}
		if !position_ocupied {
			sn.Apple = Position{X: x, Y: y}
			return
		}
	}
}

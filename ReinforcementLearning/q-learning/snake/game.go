package snake

import (
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

type Object int

const (
	Nothing Object = iota
	Border
	Apple
	Body
)

var (
	rnd = rand.New(rand.NewSource(time.Now().UnixNano()))
)

type SnakeGame struct {
	Body                    []Position
	Apple                   Position
	Size                    int
	Direction               Direction
	GameOver                bool
	Score                   float64
	max_moves_without_apple int
	moves_left              int
}

func NewSnakeGame(size int) *SnakeGame {
	game := SnakeGame{
		Size:                    size,
		GameOver:                false,
		max_moves_without_apple: size * 3,
	}
	game.Reset()
	return &game
}

func (sn *SnakeGame) Reset() {
	sn.moves_left = sn.max_moves_without_apple
	sn.GameOver = false
	sn.Score = 0.0
	sn.generateRandomSnakeBody()
	sn.generateRandomApple()
}

func (sn *SnakeGame) ChangeDirection(dir Direction) {
	sn.Direction = dir
}

func (sn *SnakeGame) generateRandomSnakeBody() {
	mlt := 1

	startX := 3 + rnd.Intn(sn.Size-6)

	if rnd.Intn(2) == 0 {
		sn.Direction = Left
	} else {
		sn.Direction = Right
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

func (sn *SnakeGame) NextTick() {
	if !sn.GameOver {
		// Do not allow it to move in circle without eating an apple
		sn.moves_left -= 1
		if sn.moves_left < 0 {
			sn.GameOver = true
			sn.Score = -1.0
			return
		}

		nextObj, next_x, next_y := sn.getObjectInFront()
		if nextObj == Body || nextObj == Border {
			sn.GameOver = true
			sn.Score = -1.0
		} else if nextObj == Apple {
			apple := Position{X: next_x, Y: next_y}
			sn.Body = append([]Position{apple}, sn.Body...)
			sn.Score += 1.0
			// Give the snake More moves ! Reward
			sn.moves_left = sn.max_moves_without_apple
		} else {
			for i := len(sn.Body) - 1; i > 0; i-- {
				sn.Body[i].X = sn.Body[i-1].X
				sn.Body[i].Y = sn.Body[i-1].Y
			}
			sn.Body[0].X = next_x
			sn.Body[0].Y = next_y
			sn.Score += 0.001
		}
	}
}

func (sn *SnakeGame) getObjectInFront() (Object, int, int) {
	bx := sn.Body[0].X
	by := sn.Body[0].Y

	switch sn.Direction {
	case Up:
		return get_object_at(bx, by-1), bx, by - 1
	case Down:
		return get_object_at(bx, by+1), bx, by + 1
	case Left:
		return get_object_at(bx-1, by), bx - 1, by
	case Right:
		return get_object_at(bx+1, by), bx + 1, by
	}
	panic("getObjectInFront: Unkown direction detected!")
}

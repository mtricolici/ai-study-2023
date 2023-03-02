package snake

import (
	"math/rand"
	"strings"
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
	ConsumedApples          int
	random_initial_position bool
	max_moves_without_apple int
	moves_left              int
	moves_made              int
}

func NewSnakeGame(size int, randomPosition bool) *SnakeGame {
	game := SnakeGame{
		Size:                    size,
		max_moves_without_apple: size * 2,
		random_initial_position: randomPosition,
	}
	game.Reset()
	return &game
}

func (sn *SnakeGame) Reset() {
	sn.moves_left = sn.max_moves_without_apple
	sn.GameOver = false
	sn.ConsumedApples = 0
	sn.moves_made = 0

	if sn.random_initial_position {
		sn.generateRandomSnakeBody()
		sn.generateRandomApple()
	} else {
		// best for QLearning to learn. the same position for all Games
		sn.generateStaticPosition()
	}
}

func (sn *SnakeGame) GetScore() float64 {
	score := float64(sn.ConsumedApples)
	score += float64(sn.moves_made) * 0.01
	return score
}

func (sn *SnakeGame) GetDirectionAsString() string {
	switch sn.Direction {
	case Up:
		return "up"
	case Down:
		return "down"
	case Left:
		return "left"
	case Right:
		return "right"
	}
	return "UNKNOWN"
}

func (sn *SnakeGame) TurnLeft() {
	switch sn.Direction {
	case Up:
		sn.Direction = Left
	case Down:
		sn.Direction = Right
	case Left:
		sn.Direction = Down
	case Right:
		sn.Direction = Up
	}
}

func (sn *SnakeGame) TurnRight() {
	switch sn.Direction {
	case Up:
		sn.Direction = Right
	case Down:
		sn.Direction = Left
	case Left:
		sn.Direction = Up
	case Right:
		sn.Direction = Down
	}
}

func (sn *SnakeGame) generateStaticPosition() {
	sn.Direction = Left

	x := sn.Size/2 - 2
	y := 2

	sn.Body = []Position{
		{X: x, Y: y},
		{X: x + 1, Y: y},
		{X: x + 2, Y: y},
	}

	sn.Apple = Position{X: x + 1, Y: y + 1}
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
			return
		}

		sn.moves_made += 1

		nextObj, next_x, next_y := sn.getObjectInFront()
		if nextObj == Body || nextObj == Border {
			sn.GameOver = true
		} else if nextObj == Apple {
			apple := Position{X: next_x, Y: next_y}
			sn.Body = append([]Position{apple}, sn.Body...)
			sn.ConsumedApples += 1

			// Give the snake More moves ! Reward
			sn.moves_left = sn.max_moves_without_apple
		} else {
			for i := len(sn.Body) - 1; i > 0; i-- {
				sn.Body[i].X = sn.Body[i-1].X
				sn.Body[i].Y = sn.Body[i-1].Y
			}
			sn.Body[0].X = next_x
			sn.Body[0].Y = next_y
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

func (sn *SnakeGame) GetState() string {
	x := sn.Body[0].X
	y := sn.Body[0].Y

	var sb strings.Builder

	// is LEFT move illegal
	sb.WriteString(bool_to_str(Can_move_to(x-1, y)))
	// is RIGHT move illegal
	sb.WriteString(bool_to_str(Can_move_to(x+1, y)))
	// is UP move illegal
	sb.WriteString(bool_to_str(Can_move_to(x, y-1)))
	// is Down move illegal
	sb.WriteString(bool_to_str(Can_move_to(x, y+1)))

	// is FOOD on the left
	sb.WriteString(bool_to_str(x > sn.Apple.X))
	// is FOOD on the right
	sb.WriteString(bool_to_str(x < sn.Apple.X))
	// is FOOD up
	sb.WriteString(bool_to_str(y > sn.Apple.Y))
	// is FOOD down
	sb.WriteString(bool_to_str(y < sn.Apple.Y))

	return sb.String()
}

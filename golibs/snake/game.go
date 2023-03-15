package snake

import (
	"math"
	"math/rand"
	"strings"
	"time"
)

var (
	rnd = rand.New(rand.NewSource(time.Now().UnixNano()))
)

func NewSnakeGame(size int, randomPosition bool) *SnakeGame {
	game := SnakeGame{
		Size:                    size,
		Random_initial_position: randomPosition,
		Reward_apple:            10.0,
		Reward_die:              -10.0,
		Reward_move_to_apple:    0.5,
		Reward_move_from_apple:  -0.5,
		Small_State_for_Neural:  true,
	}
	game.Reset()
	return &game
}

func (sn *SnakeGame) Reset() {
	sn.GameOver = false
	sn.ConsumedApples = 0
	sn.Moves_made = 0
	sn.Moves_since_apple = 0

	if sn.Random_initial_position {
		sn.generateRandomSnakeBody()
		sn.generateRandomApple()
	} else {
		// best for QLearning to learn. the same position for all Games
		sn.generateStaticPosition()
	}
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

func (sn *SnakeGame) ChangeDirection(newDirection Direction) {
	sn.Direction = newDirection
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
		sn.Moves_made += 1
		sn.Moves_since_apple += 1

		nextObj, next_x, next_y := sn.getObjectInFront()
		if nextObj == Body || nextObj == Border {
			sn.Reward = sn.Reward_die
			sn.GameOver = true
		} else if nextObj == Apple {
			apple := Position{X: next_x, Y: next_y}
			sn.Body = append([]Position{apple}, sn.Body...)
			sn.ConsumedApples += 1
			sn.Reward = sn.Reward_apple
			sn.Moves_since_apple = 0
			sn.generateRandomApple() // Generate a new apple!
		} else {
			sn.Reward = sn.CalculateReward(next_x, next_y)
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
		return sn.get_object_at(bx, by-1), bx, by - 1
	case Down:
		return sn.get_object_at(bx, by+1), bx, by + 1
	case Left:
		return sn.get_object_at(bx-1, by), bx - 1, by
	case Right:
		return sn.get_object_at(bx+1, by), bx + 1, by
	}
	panic("getObjectInFront: Unkown direction detected!")
}

func (sn *SnakeGame) CalculateReward(next_x, next_y int) float64 {
	x := sn.Body[0].X
	y := sn.Body[0].Y

	ax := sn.Apple.X
	ay := sn.Apple.Y

	current_distance := math.Hypot(float64(x-ax), float64(y-ay))
	next_distance := math.Hypot(float64(next_x-ax), float64(next_y-ay))

	// If snake is moving TO apple then a small reward!
	if next_distance < current_distance {
		return sn.Reward_move_to_apple
	}

	return sn.Reward_move_from_apple
}

func (sn *SnakeGame) GetState() string {
	x := sn.Body[0].X
	y := sn.Body[0].Y

	var sb strings.Builder

	// Save current direction
	//sb.WriteString(strconv.Itoa(int(sn.Direction)))

	// is LEFT move allowed
	sb.WriteString(bool_to_str(sn.can_move_to(x-1, y)))
	// is RIGHT move allowed
	sb.WriteString(bool_to_str(sn.can_move_to(x+1, y)))
	// is UP move allowed
	sb.WriteString(bool_to_str(sn.can_move_to(x, y-1)))
	// is Down move allowed
	sb.WriteString(bool_to_str(sn.can_move_to(x, y+1)))

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

func (sn *SnakeGame) GetStateForNeuralNetwork() []float64 {
	if sn.Small_State_for_Neural {
		return sn.getLimittedViewState()
	}

	return sn.getFullBoardState()
}

func (sn *SnakeGame) getLimittedViewState() []float64 {
	x := sn.Body[0].X
	y := sn.Body[0].Y

	state := make([]float64, 12)

	state[0] = bool_to_float(sn.Direction == Left)
	state[1] = bool_to_float(sn.Direction == Right)
	state[2] = bool_to_float(sn.Direction == Up)
	state[3] = bool_to_float(sn.Direction == Down)

	// is LEFT move allowed
	state[4] = bool_to_float(sn.can_move_to(x-1, y))
	// is RIGHT move allowed
	state[5] = bool_to_float(sn.can_move_to(x+1, y))
	// is UP move allowed
	state[6] = bool_to_float(sn.can_move_to(x, y-1))
	// is Down move allowed
	state[7] = bool_to_float(sn.can_move_to(x, y+1))

	// is FOOD on the left
	state[8] = bool_to_float(x > sn.Apple.X)
	// is FOOD on the right
	state[9] = bool_to_float(x < sn.Apple.X)
	// is FOOD up
	state[10] = bool_to_float(y > sn.Apple.Y)
	// is FOOD down
	state[11] = bool_to_float(y < sn.Apple.Y)

	return state
}

func (sn *SnakeGame) getFullBoardState() []float64 {
	// input neurons:
	// Size*Size for each square in board
	// + 4 inputs for each direction
	// + 1 input that shows how many moves since last apple
	stateSize := sn.Size*sn.Size + 5
	state := make([]float64, stateSize)

	for _, b := range sn.Body {
		idx := b.X*sn.Size + b.Y
		state[idx] = 0.5 // This is body!
	}

	appleIndex := sn.Apple.X*sn.Size + sn.Apple.Y
	state[appleIndex] = 1.0 // this is apple!

	// Direction as 4 inputs
	state[stateSize-5] = bool_to_float(sn.Direction == Left)
	state[stateSize-4] = bool_to_float(sn.Direction == Right)
	state[stateSize-3] = bool_to_float(sn.Direction == Up)
	state[stateSize-2] = bool_to_float(sn.Direction == Down)

	state[stateSize-1] = float64(sn.Moves_since_apple) / float64(sn.Size*10)

	return state
}

func (sn *SnakeGame) TurnLeft() {
	switch sn.Direction {
	case Up:
		sn.Direction = Left
	case Down:
		sn.Direction = Right
	case Right:
		sn.Direction = Up
	case Left:
		sn.Direction = Down
	default:
		panic("Snake.TurnLeft - unknown current direction detected!")
	}
}

func (sn *SnakeGame) TurnRight() {
	switch sn.Direction {
	case Up:
		sn.Direction = Right
	case Down:
		sn.Direction = Left
	case Right:
		sn.Direction = Down
	case Left:
		sn.Direction = Up
	default:
		panic("Snake.TurnRight - unknown current direction detected!")
	}
}

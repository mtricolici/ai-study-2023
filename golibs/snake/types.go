package snake

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

type SnakeGame struct {
	Body           []Position
	Apple          Position
	Size           int
	Direction      Direction
	GameOver       bool
	ConsumedApples int
	Moves_made     int
	Reward         float64

	random_initial_position bool
}

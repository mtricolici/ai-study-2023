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
	Body              []Position
	Apple             Position
	Size              int
	Direction         Direction
	GameOver          bool
	ConsumedApples    int
	Moves_made        int
	Moves_since_apple int
	Reward            float64

	random_initial_position bool

	Reward_apple           float64
	Reward_die             float64
	Reward_move_to_apple   float64
	Reward_move_from_apple float64
}

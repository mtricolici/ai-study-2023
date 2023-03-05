package snake

func (sn *SnakeGame) get_object_at(x, y int) Object {
	if x < 0 || x >= sn.Size || y < 0 || y >= sn.Size {
		return Border
	}

	if sn.Apple.X == x && sn.Apple.Y == y {
		return Apple
	}

	for _, p := range sn.Body {
		if p.X == x && p.Y == y {
			return Body
		}
	}

	return Nothing
}

func (sn *SnakeGame) can_move_to(x, y int) bool {
	obj := sn.get_object_at(x, y)
	return obj == Nothing || obj == Apple
}

func bool_to_str(b bool) string {
	if b {
		return "1"
	}

	return "0"
}

func bool_to_float(b bool) float64 {
	if b {
		return 1.0
	}

	return 0.0
}

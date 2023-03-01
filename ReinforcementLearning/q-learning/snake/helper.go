package snake

func get_object_at(x, y int) Object {
	size := Get_game_size()
	if x < 0 || x >= size || y < 0 || y >= size {
		return Border
	}

	data := Get_game_data(x, y)
	switch data {
	case 3:
		return Apple
	case 2:
		return Body // head actually
	case 1:
		return Body
	}

	return Nothing
}

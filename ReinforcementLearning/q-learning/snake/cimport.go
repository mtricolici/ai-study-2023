package snake

// #cgo LDFLAGS: -L/tmp -lsnake
// void create_game(int size);
// int get_game_size();
// void set_game_data(int x, int y, int value);
// int get_game_data(int x, int y);
// void destroy_game();
//int create_window();
//void draw_objects();
//void close_window();
import "C"

func Create_game(size int) {
	C.create_game(C.int(size))
}

func Get_game_size() int {
	return int(C.get_game_size())
}

func Get_game_data(x, y int) int {
	return int(C.get_game_data(C.int(x), C.int(y)))
}

func Can_move_to(x, y int) bool {
	s := Get_game_size()

	if x < 0 || x >= s || y < 0 || y >= s {
		return false
	}
	data := Get_game_data(x, y)

	// should not be Head or Body
	return data != 1 && data != 2
}

func Set_game_data(x, y, value int) {
	C.set_game_data(C.int(x), C.int(y), C.int(value))
}

func UpdateGameData(g *SnakeGame) {
	for i := 0; i < g.Size; i++ {
		for j := 0; j < g.Size; j++ {
			Set_game_data(i, j, 0) // put 'nothing'
		}
	}

	Set_game_data(g.Apple.X, g.Apple.Y, 3) // Apple
	for i, p := range g.Body {
		if i == 0 {
			Set_game_data(p.X, p.Y, 1) // Head
		} else {
			Set_game_data(p.X, p.Y, 2) // Body
		}
	}
}

func Destroy_game() {
	C.destroy_game()
}

func X_create_window() {
	C.create_window()
}

func X_draw_objects() {
	C.draw_objects()
}

func X_close_window() {
	C.close_window()
}

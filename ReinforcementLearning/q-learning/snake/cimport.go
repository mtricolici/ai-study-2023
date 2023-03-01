package snake

// #cgo LDFLAGS: -L/tmp -lsnake
// void create_game(int size);
// void set_game_data(int x, int y, int value);
// void destroy_game();
//int create_window();
//void draw_objects();
//void close_window();
import "C"

func Create_game(size int) {
	C.create_game(C.int(size))
}

func Set_game_data(x, y, value int) {
	C.set_game_data(C.int(x), C.int(y), C.int(value))
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

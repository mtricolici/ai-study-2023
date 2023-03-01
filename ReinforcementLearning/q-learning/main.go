package main

import "fmt"

// #cgo LDFLAGS: -L${SRCDIR}/../../c-ui-app -lsnake
// void create_game(int size);
// void set_game_data(int x, int y, int value);
// void destroy_game();
import "C"

func main() {
	fmt.Println("Hello world")
	C.create_game(20)
	C.set_game_data(3, 3, 2)
	C.destroy_game()
}

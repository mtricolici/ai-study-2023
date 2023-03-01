//#include <stdio.h>
//#include <unistd.h>
#include "game.h"
#include "xwrapper.h"

/*
Example of usage:
int main() {
    create_game(20);
    set_game_data(10,5,1); // head at 10x5
    set_game_data(9,5,2); // body at 9x5
    set_game_data(8,5,2); // body at 8x5
    set_game_data(7,5,2); // body at 7x5
    set_game_data(13,13,3); // apple at 13x13

    if (create_window()) {

       printf("window created!\n");
       sleep(3);

       set_game_data(14,14,3); // apple at 14x14
       draw_objects();
       sleep(3);

       printf("stopping...\n");
       close_window();
    } else {
       printf("window creation failure\n");
    }

    destroy_game();
    return 0;
}
*/

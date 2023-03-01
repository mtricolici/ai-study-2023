#ifndef __GAME_H__
#define __GAME_H__

void create_game(int size);
void set_game_data(int x, int y, int value);
void draw_objects();
void destroy_game();

int create_window();
void close_window();

#endif

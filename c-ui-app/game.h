#ifndef __GAME_H__
#define __GAME_H__

void create_game(int size);
int get_game_size();

int get_game_data(int x, int y);
void set_game_data(int x, int y, int value);

void destroy_game();

#endif

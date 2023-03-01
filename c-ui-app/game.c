#include <stdio.h>
#include <stdlib.h>

#include "game.h"
#define win_width 400
#define win_height 400


#pragma GCC visibility push(hidden)
// game variables
int g_size;
int **game_data = NULL; // matrix. 0-nothing, 1-> head, 2->body, 3->apple
#pragma GCC visibility pop

int get_game_size() {
  return g_size;
}

// creates a matrix of ints (snake game)
void create_game(int size) {
  g_size = size;
  game_data = (int**)malloc(size * sizeof(int*));

  // alocate memory for each array
  for (int i=0; i<size; i++) {
    game_data[i] = malloc(size * sizeof(int));
    for (int j=0; j<size; j++) {
      game_data[i][j] = 0; // nothing
    }
  }
}

int get_game_data(int x, int y) {
  return game_data[x][y];
}

void set_game_data(int x, int y, int value) {
  game_data[x][y] = value;
}

void destroy_game() {
  if (game_data != NULL) {
    for (int i=0; i<g_size; i++) {
      free(game_data[i]);
    }
    free(game_data);
  }
  game_data = NULL;
}


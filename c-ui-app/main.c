#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <stdbool.h>
#define win_width 400
#define win_height 400


// display variables used in bellow functions
Display *display = NULL;
Window window;
int screen = 0;
GC gc;
pthread_t loop_thr;
bool thread_created = false;
bool thread_should_stop = false;

// game variables
int g_size;
int **game_data = NULL; // matrix. 0-nothing, 1-> head, 2->body, 3->apple

// color constants
unsigned long color_nothing = 0xFFFFFF; // white
unsigned long color_head = 0x0000EE;
unsigned long color_body = 0x000055;
unsigned long color_apple = 0xAA0000;

unsigned long get_color(int x, int y) {
  switch (game_data[x][y]) {
    case 1: return color_head;
    case 2: return color_body;
    case 3: return color_apple;
  }

  return color_nothing;
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

void* window_loop(); // implemented later


// Creates a simple X window
int create_window() {
    display = XOpenDisplay(NULL);
    if (display == NULL) {
        fprintf(stderr, "Cannot open display\n");
        return 0;
    }

    screen = DefaultScreen(display);

    unsigned long bg_color = 0xFFAAAA;
    window = XCreateSimpleWindow(display, RootWindow(display, screen), 0, 0, win_width, win_height, 0, BlackPixel(display, screen), bg_color);
    XSizeHints *hints = XAllocSizeHints();
    if (hints != NULL) {
        hints->flags |= PMinSize | PMaxSize | PResizeInc;
        hints->min_width = hints->max_width = hints->base_width = win_width;
        hints->min_height = hints->max_height = hints->base_height = win_height;
        hints->width_inc = hints->height_inc = 0;
        XSetWMNormalHints(display, window, hints);
        XFree(hints);
    }

    XSelectInput(display, window, ExposureMask | KeyPressMask);
    XMapWindow(display, window);
    gc = XCreateGC(display, window, 0, NULL);

    int ret = pthread_create(&loop_thr, NULL, window_loop, NULL);
    if (ret != 0) {
      fprintf(stderr, "Error creating window loop");
      return 2;
    } else {
      thread_created = true;
    }
    return 1;
}

void draw_objects(){
    if (game_data == NULL) {
      return; // nothing to draw
    }

    int p_w = win_width / g_size;
    int p_h = win_height / g_size;


    for (int i=0; i<g_size; i++) {
      for (int j=0; j<g_size; j++) {
        XSetForeground(display, gc, get_color(i, j));
        int x=i*p_w;
        int y=j*p_h;
        XFillRectangle(display, window, gc, x+1, y+1, p_w-2, p_h-2);
      }
    }

    //XDrawLine(display, window, gc, 50, 50, 350, 350); // adjusted coordinates
    //XDrawLine(display, window, gc, 200, 200, 100, 10); // adjusted coordinates
    XFlush(display);
}

// Closes previous created window
void close_window() {
    if (thread_created) {
      thread_should_stop = true;
      pthread_join(loop_thr, NULL);
    }
    XFreeGC(display, gc);
    XDestroyWindow(display, window);
    XCloseDisplay(display);
}

void* window_loop() {
    XEvent event;

    while (!thread_should_stop) {
        while(XPending(display)) {
            XNextEvent(display, &event);
            switch(event.type){
            case Expose:
              draw_objects();
              break;
            }
        }
        usleep(10000); // sleep 10 ms = 10,000 microseconds
    }

    return NULL;
}

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


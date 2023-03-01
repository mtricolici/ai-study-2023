#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <stdbool.h>

#include "xwrapper.h"
#include "game.h"
#define win_width 400
#define win_height 400


#pragma GCC visibility push(hidden)

// display variables used in bellow functions
Display *display = NULL;
Window window;
int screen = 0;
GC gc;

// thread variables
pthread_t loop_thr;
bool thread_created = false;
bool thread_should_stop = false;

// color constants
unsigned long color_nothing = 0xFFFFFF; // white
unsigned long color_head = 0x0000EE;
unsigned long color_body = 0x000055;
unsigned long color_apple = 0xAA0000;


unsigned long get_color(int x, int y) {
  switch (get_game_data(x, y)) {
    case 1: return color_head;
    case 2: return color_body;
    case 3: return color_apple;
  }

  return color_nothing;
}

void* window_loop(); // implemented later

#pragma GCC visibility pop

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
    int w = win_width / get_game_size();
    int h = win_height / get_game_size();


    for (int i=0; i<get_game_size(); i++) {
      for (int j=0; j<get_game_size(); j++) {
        XSetForeground(display, gc, get_color(i, j));
        XFillRectangle(display, window, gc, i*w + 1, j*h + 1, w - 2, h - 2);
      }
    }

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

#pragma GCC visibility push(hidden)
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
#pragma GCC visibility pop



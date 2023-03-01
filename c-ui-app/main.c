#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <stdio.h>
#include <stdlib.h>
#define win_width 400
#define win_height 400


// display variables used in bellow functions
Display *display = NULL;
Window window;
int screen = 0;
GC gc;

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
    return 1;
}

void draw_objects(){
    XSetForeground(display, gc, BlackPixel(display, screen));
    XDrawLine(display, window, gc, 50, 50, 350, 350); // adjusted coordinates
    XDrawLine(display, window, gc, 200, 200, 100, 10); // adjusted coordinates
    XFlush(display);
}

// Closes previous created window
void close_window() {
    XFreeGC(display, gc);
    XDestroyWindow(display, window);
    XCloseDisplay(display);
}

void window_loop() {
    XEvent event;
    while (1) {
        XNextEvent(display, &event);
        switch(event.type){
        case Expose:
          draw_objects();
          break;
        case KeyPress:
          return;
        }
    }
}

int main() {
    if (create_window()) {
       printf("window created!\n");
       window_loop();
       close_window();
    } else {
       printf("window creation failure\n");
    }
    return 0;
}


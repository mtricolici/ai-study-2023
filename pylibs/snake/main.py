#!/usr/bin/env python

from snake import SnakeGame

def main():
    g = SnakeGame(10, True)
    print(vars(g))

    while not g.game_over:
        g.next_tick()
        print(vars(g))


if __name__ == "__main__":
    main()

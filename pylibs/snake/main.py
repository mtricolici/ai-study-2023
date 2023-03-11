#!/usr/bin/env python

from snake import SnakeGame
from snake import Direction
from snake_ui import SnakeWindow
import pygame

if __name__ == "__main__":
    game = SnakeGame(10, True)
    win = SnakeWindow(game, "Mega Snake Game")
    win.draw()

    pygame.time.set_timer(pygame.USEREVENT+1, 200)
    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.USEREVENT+1:
                game.next_tick()
                win.draw()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    game.direction = Direction.UP
                elif event.key == pygame.K_DOWN:
                    game.direction = Direction.DOWN
                elif event.key == pygame.K_LEFT:
                    game.direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT:
                    game.direction = Direction.RIGHT
                elif event.key == pygame.K_RETURN:
                    game.reset()

        pygame.display.flip()
        clock.tick(60) # 60 fps
    
    pygame.quit()

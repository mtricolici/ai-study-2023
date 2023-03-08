#!/usr/bin/env python

from snake import SnakeGame
from snake import Direction
import pygame
from pygame.surface import Surface

GAME_SIZE=400
WINDOW_WIDTH = GAME_SIZE + 300
WINDOW_HEIGHT = GAME_SIZE + 20

BG_COLOR = (255, 255, 255)
LINE_COLOR = (111, 111, 111)
APPLE_COLOR = (200, 0, 0)
HEAD_COLOR = (0,0,255)
BODY_COLOR = (0,0,100)

def draw_text(screen: Surface, text:str, color, coords):
    font = pygame.font.Font(None, 36)
    text = font.render(text, True, color)
    screen.blit(text, coords)

def draw(screen: Surface, game: SnakeGame):
    screen.fill(BG_COLOR)

    offset_x = 10
    offset_y = (WINDOW_HEIGHT - GAME_SIZE) // 2

    dx = GAME_SIZE // game.size
    dy = GAME_SIZE // game.size

    # draw upper horizontal line
    pygame.draw.line(screen, LINE_COLOR, (offset_x, offset_y), (game.size*dx+offset_x, offset_y), 1)
    # draw left vertical line
    pygame.draw.line(screen, LINE_COLOR, (offset_x, offset_y), (offset_x, game.size*dy + offset_y), 1)

    for x in range(game.size):
        for y in range(game.size):
            # draw bottom horizontal line
            pygame.draw.line(screen, LINE_COLOR, (x*dx+offset_x, (y+1)*dy+offset_y), ((x+1)*dx+offset_x, (y+1)*dy+offset_y), 1)
            # draw right vertical line
            pygame.draw.line(screen, LINE_COLOR, ((x+1)*dx+offset_x, y*dy+offset_y), ((x+1)*dx+offset_x, (y+1)*dy+offset_y), 1)

    # draw body
    for i, b in enumerate(game.body):
        color = BODY_COLOR
        if i==0:
            color = HEAD_COLOR
        pygame.draw.rect(screen, color, pygame.Rect(
            b[0]*dx+offset_x+2, b[1]*dy+offset_y+2, dx-3, dy-3))

    # draw apple
    pygame.draw.rect(screen, APPLE_COLOR, pygame.Rect(
        game.apple[0]*dx+offset_x+2, game.apple[1]*dy+offset_y+2, dx-3, dy-3))
    
    draw_text(screen, f"reward:{game.reward}", (200, 200, 0), (GAME_SIZE+offset_x + 5, offset_y))
    draw_text(screen, f"moves:{game.moves_made}", (200, 200, 0), (GAME_SIZE+offset_x + 5, offset_y+25))
    draw_text(screen, f"apples:{game.consumed_apples}", (200, 200, 0), (GAME_SIZE+offset_x + 5, offset_y+50))
    draw_text(screen, f"game-over:{game.game_over}", (200, 200, 0), (GAME_SIZE+offset_x + 5, offset_y+75))
    draw_text(screen, f"without-apple:{game.moves_since_apple}", (200, 200, 0), (GAME_SIZE+offset_x + 5, offset_y+100))
    draw_text(screen, f"state:{game.get_state()}", (200, 200, 0), (GAME_SIZE+offset_x + 5, offset_y+125))

def handle_key(key, game: SnakeGame):
    if key == pygame.K_UP:
        game.direction = Direction.UP
    elif key == pygame.K_DOWN:
        game.direction = Direction.DOWN
    elif key == pygame.K_LEFT:
        game.direction = Direction.LEFT
    elif key == pygame.K_RIGHT:
        game.direction = Direction.RIGHT
    elif key == pygame.K_RETURN:
        game.reset()

if __name__ == "__main__":
    game = SnakeGame(10, True)

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Mega Snake Game")
    draw(screen, game)
    pygame.time.set_timer(pygame.USEREVENT+1, 200)

    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.USEREVENT+1:
                game.next_tick()
                draw(screen, game)
            elif event.type == pygame.KEYDOWN:
                handle_key(event.key, game)

        pygame.display.flip()
        clock.tick(60) # 60 fps
    
    pygame.quit()

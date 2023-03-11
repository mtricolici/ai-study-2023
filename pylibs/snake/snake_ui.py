
import pygame
from pygame.surface import Surface
from snake import SnakeGame
from snake import Direction

GAME_SIZE=400
WINDOW_WIDTH = GAME_SIZE + 300
WINDOW_HEIGHT = GAME_SIZE + 20

BG_COLOR = (255, 255, 255)
LINE_COLOR = (111, 111, 111)
APPLE_COLOR = (200, 0, 0)
HEAD_COLOR = (0,0,255)
BODY_COLOR = (0,0,100)

class SnakeWindow:
    def __init__(self, game:SnakeGame, caption:str):
        self.game = game
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption(caption)

    def _draw_text(self, text:str, color, coords):
        font = pygame.font.Font(None, 36)
        text = font.render(text, True, color)
        self.screen.blit(text, coords)

    def draw(self):
        self.screen.fill(BG_COLOR)

        offset_x = 10
        offset_y = (WINDOW_HEIGHT - GAME_SIZE) // 2

        dx = GAME_SIZE // self.game.size
        dy = GAME_SIZE // self.game.size

        # draw upper horizontal line
        pygame.draw.line(self.screen, LINE_COLOR, (offset_x, offset_y), (self.game.size*dx+offset_x, offset_y), 1)
        # draw left vertical line
        pygame.draw.line(self.screen, LINE_COLOR, (offset_x, offset_y), (offset_x, self.game.size*dy + offset_y), 1)

        for x in range(self.game.size):
            for y in range(self.game.size):
                # draw bottom horizontal line
                pygame.draw.line(self.screen, LINE_COLOR, (x*dx+offset_x, (y+1)*dy+offset_y), ((x+1)*dx+offset_x, (y+1)*dy+offset_y), 1)
                # draw right vertical line
                pygame.draw.line(self.screen, LINE_COLOR, ((x+1)*dx+offset_x, y*dy+offset_y), ((x+1)*dx+offset_x, (y+1)*dy+offset_y), 1)

        # draw body
        for i, b in enumerate(self.game.body):
            color = BODY_COLOR
            if i==0:
                color = HEAD_COLOR
            pygame.draw.rect(self.screen, color, pygame.Rect(
                b[0]*dx+offset_x+2, b[1]*dy+offset_y+2, dx-3, dy-3))

        # draw apple
        pygame.draw.rect(self.screen, APPLE_COLOR, pygame.Rect(
            self.game.apple[0]*dx+offset_x+2, self.game.apple[1]*dy+offset_y+2, dx-3, dy-3))
        
        self._draw_text(f"reward:{self.game.reward}", (200, 200, 0), (GAME_SIZE+offset_x + 5, offset_y))
        self._draw_text(f"moves:{self.game.moves_made}", (200, 200, 0), (GAME_SIZE+offset_x + 5, offset_y+25))
        self._draw_text(f"apples:{self.game.consumed_apples}", (200, 200, 0), (GAME_SIZE+offset_x + 5, offset_y+50))
        self._draw_text(f"game-over:{self.game.game_over}", (200, 200, 0), (GAME_SIZE+offset_x + 5, offset_y+75))
        self._draw_text(f"without-apple:{self.game.moves_since_apple}", (200, 200, 0), (GAME_SIZE+offset_x + 5, offset_y+100))
        self._draw_text(f"state:{self.game.get_state()}", (200, 200, 0), (GAME_SIZE+offset_x + 5, offset_y+125))

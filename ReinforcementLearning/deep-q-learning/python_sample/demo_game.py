import os
import sys
from ai import deep_qlearn
current = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(current, "../../../../pylibs/snake")))
from snake import SnakeGame
from snake_ui import SnakeWindow
import pygame

def demo(size:int, saveFileName:str):
    print("Running demo...")
    display = os.environ["DISPLAY"]
    print(f"display is '{display}'")
    game = SnakeGame(size, True)
    ai = deep_qlearn.DeepQLearning(game)
    ai.load(saveFileName)

    win = SnakeWindow(game, "Demo AI game")
    win.draw()

    pygame.time.set_timer(pygame.USEREVENT+1, 200)
    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.USEREVENT+1:
                ai.demo_predict_next_direction()
                game.next_tick()
                win.draw()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    game.reset()

        pygame.display.flip()
        clock.tick(60) # 60 fps
    
    pygame.quit()
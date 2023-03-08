from enum import Enum
import math
import random

class Direction(Enum):
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"

class Object(Enum):
    NOTHING = 0
    APPLE = 1
    BORDER = 2
    BODY = 3

class SnakeGame:
    def __init__(self, size: int, randomPosition: bool):
        self.size = size
        self.random_position = randomPosition
        self.reward_apple = 1.0
        self.reward_die = -1.0
        self.reward_move_to_apple = 0.1
        self.reward_move_from_apple = -0.2
        self.small_state_for_neural = False
        self.reward = 0.0
        self.reset()
    
    # reset - create new game
    def reset(self):
        self.game_over = False
        self.consumed_apples = 0
        self.moves_made = 0
        self.moves_since_apple = 0
        if self.random_position:
            self._generate_random_snake_body()
            self._generate_random_apple()
        else:
            self._generate_static_position()
    
    def next_tick(self):
        if self.game_over:
            return
        self.moves_made += 1
        self.moves_since_apple += 1

        next_obj, next_x, next_y = self._get_object_in_front()

        if next_obj == Object.BODY or next_obj == Object.BORDER:
            self.reward = self.reward_die
            self.game_over = True
        elif next_obj == Object.APPLE:
            self.body.insert(0, (next_x, next_y))
            self.consumed_apples += 1
            self.reward = self.reward_apple
            self.moves_since_apple = 0
            self._generate_random_apple()
        else:
            self.reward = self._calculate_reward(next_x, next_y)
            for i in range(len(self.body) - 1, 0, -1):
                self.body[i] = self.body[i-1]
            self.body[0] = (next_x, next_y)
    
    def _generate_random_snake_body(self):
        mlt = 1

        startX = 3 + random.randint(0, self.size - 6)

        if random.randint(0, 1) == 0:
            self.direction = Direction.LEFT
        else:
            self.direction = Direction.RIGHT
            mlt = -1

        startY = random.randint(0, self.size - 1)

        self.body = [
            (startX, startY),
            (startX + mlt * 1, startY),
            (startX + mlt * 2, startY)
        ]
    
    def _generate_random_apple(self):
        while True:
            x = random.randint(0, self.size - 1)
            y = random.randint(0, self.size - 1)
            position_occupied = False
            for p in self.body:
                if p[0] == x and p[1] == y:
                    position_occupied = True
                    break
            if not position_occupied:
                self.apple = (x, y)
                return

    def _generate_static_position(self):
        self.direction = Direction.LEFT
        x = self.size // 2 - 2
        self.body = [(x, 2), (x + 1, 2), (x + 2, 2)]
        self.apple = (x, 3)

    def _get_object_at(self, x:int, y:int):
        if x < 0 or x >= self.size or y < 0 or y >= self.size:
            return Object.BORDER

        if (x, y) == self.apple:
            return Object.APPLE

        for p in self.body:
            if (x, y) == p:
                return Object.BODY

        return Object.NOTHING

    def _get_object_in_front(self):
        bx = self.body[0][0]
        by = self.body[0][1]

        if self.direction == Direction.UP:
            return self._get_object_at(bx, by - 1), bx, by - 1
        elif self.direction == Direction.DOWN:
            return self._get_object_at(bx, by + 1), bx, by + 1
        elif self.direction == Direction.LEFT:
            return self._get_object_at(bx - 1, by), bx - 1, by
        elif self.direction == Direction.RIGHT:
            return self._get_object_at(bx + 1, by), bx + 1, by

        raise ValueError("SnakeGame._get_object_in_front: Unknown direction detected!")

    def _can_move_to(self, x:int, y:int):
        obj = self._get_object_at(x, y)
        return obj == Object.NOTHING or obj == Object.APPLE

    def _calculate_reward(self, next_x:int, next_y:int):
        x = self.body[0][0]
        y = self.body[0][1]

        ax = self.apple[0]
        ay = self.apple[1]

        current_distance = math.hypot(x - ax, y - ay)
        next_distance = math.hypot(next_x - ax, next_y - ay)

        # If snake is moving TO apple then a small reward!
        if next_distance < current_distance:
            return self.reward_move_to_apple

        return self.reward_move_from_apple

    def _bool_to_float(self, b:bool):
        if b:
            return 1.0
        else:
            return 0.0

    def _bool_to_str(self, b:bool):
        if b:
            return "1"
        else:
            return "0"

    
    def get_state(self):
        x = self.body[0][0]
        y = self.body[0][1]
        # is LEFT move allowed
        state = self._bool_to_str(self._can_move_to(x-1, y))
        # is RIGHT move allowed
        state += self._bool_to_str(self._can_move_to(x+1, y))
        # is UP move allowed
        state += self._bool_to_str(self._can_move_to(x, y-1))
        # is DOWN move allowed
        state += self._bool_to_str(self._can_move_to(x, y+1))
        # is food on the LEFT
        state += self._bool_to_str(x > self.apple[0])
        # is food on the RIGHT
        state += self._bool_to_str(x < self.apple[0])
        # is food UP
        state += self._bool_to_str(y > self.apple[1])
        # is food DOWN
        state += self._bool_to_str(y < self.apple[1])
        return state

    def get_state_for_nn(self):
        if self.small_state_for_neural:
            return self._get_limited_view_state()
        return self._get_full_board_view_state()

    def _get_limited_view_state(self):
        raise NotImplementedError()
    
    def _get_full_board_view_state(self):
        raise NotImplementedError()
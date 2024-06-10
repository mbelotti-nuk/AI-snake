# importing libraries
import pygame
import random
from collections import namedtuple, deque
import numpy as np

point = namedtuple("point", ["x", "y"])
# Initialising pygame
pygame.init()

class SnakeGame:
    snake_speed = 40
    # Window size
    window_size = point(x=640, y=480)
    # block
    BLOCK_SIZE = 20
    # defining colors
    black = pygame.Color(0, 0, 0)
    white = pygame.Color(255, 255, 255)
    red =   pygame.Color(255, 0, 0)
    green = pygame.Color(0, 255, 0)
    blue =  pygame.Color(0, 0, 255)
    directions_indices = {"UP":0, "LEFT":1, "DOWN":2, "RIGHT":3}

    def __init__(self) -> None:
        # Initialise game window
        self.game_window = pygame.display.set_mode((self.window_size.x, self.window_size.y))
        pygame.display.set_caption('Snake')
        self.fps = pygame.time.Clock()
        self.reset()

    @property
    def frame(self):
        return np.array(pygame.surfarray.array3d( pygame.transform.rotate(pygame.transform.flip(self.game_window, True, False), 90)))

    def reset(self)->None:
        # snake default position
        self.snake_head = point(x=100, y=40)

        # snake body as made by 4 pixels
        self.snake_body = [point(x=100, y=40),
                           point(x=80,  y=40),
                           point(x=60,  y=40),
                           point(x=40,  y=40)]
        # fruit position
        self.fruit_position = point(x=0, y=0)
        self.new_fruit()
        self.fruit_spawn = True

        # setting default snake direction towards
        # right
        self.direction = 'RIGHT'
        self.current_action = 0
        self.last_action = 0
        self.last_direction = self.direction
        # initial score
        self.score = 0


    def new_fruit(self):
        self.fruit_position = point( x = random.randrange(1, (self.window_size.x//self.BLOCK_SIZE)) * self.BLOCK_SIZE, 
                                     y = random.randrange(1, (self.window_size.y//self.BLOCK_SIZE)) * self.BLOCK_SIZE)

    def render(self, font=pygame.font.SysFont('arial', 25)):
        self.game_window.fill(self.black)
        for pos in self.snake_body:
            pygame.draw.rect(self.game_window, self.green, pygame.Rect(pos.x, pos.y, self.BLOCK_SIZE, self.BLOCK_SIZE))
        pygame.draw.rect(self.game_window, self.red, pygame.Rect(self.fruit_position.x, self.fruit_position.y, self.BLOCK_SIZE, self.BLOCK_SIZE))
        text = font.render("Score: " + str(self.score), True, self.white)
        self.game_window.blit(text, [0, 0])
        pygame.display.flip()
        pygame.display.flip()

    def move_to(self, direction:str)->None:
        """Move the self.snake_position 

        Args:
            direction (str): motion direction (UP, DOWN, LEFT or RIGHT)
        """
        if direction == 'UP':
            self.snake_head = point(self.snake_head.x, self.snake_head.y + self.BLOCK_SIZE)
        if direction == 'DOWN':
            self.snake_head = point(self.snake_head.x, self.snake_head.y - self.BLOCK_SIZE)
        if direction == 'LEFT':
            self.snake_head = point(self.snake_head.x - + self.BLOCK_SIZE, self.snake_head.y)
        if direction == 'RIGHT':
            self.snake_head = point(self.snake_head.x + self.BLOCK_SIZE, self.snake_head.y)

    def set_direction(self, action)->None:
        if action == 0:
            self.direction = 'UP'
        if action == 1:
            self.direction = 'LEFT'
        if action == 2:
            self.direction = 'DOWN'
        if action == 3:
            self.direction = 'RIGHT'
        
    def is_collision(self, p:point):
        # boundary collision
        if p.x < 0 or p.x > self.window_size.x-self.BLOCK_SIZE:
            return True
        if p.y < 0 or p.y > self.window_size.y-self.BLOCK_SIZE:
            return True
        # self collision
        if p in self.snake_body[1:]:
            return True
        return False

    @property
    def state(self)->np.array:

        direction_left  = self.direction == 'LEFT'
        direction_right = self.direction == 'RIGHT'
        direction_up    = self.direction == 'UP'
        direction_down  = self.direction == 'DOWN'

        up_point =      point(self.snake_head.x, self.snake_head.y + self.BLOCK_SIZE)
        down_point =    point(self.snake_head.x, self.snake_head.y - self.BLOCK_SIZE)
        left_point =    point(self.snake_head.x - self.BLOCK_SIZE, self.snake_head.y)
        right_point =   point(self.snake_head.x + self.BLOCK_SIZE, self.snake_head.y)

        state = [
            # Collision straight
            (direction_right    and     self.is_collision(right_point)) or 
            (direction_left     and     self.is_collision(left_point)) or 
            (direction_up       and     self.is_collision(up_point)) or 
            (direction_down     and     self.is_collision(down_point)),

            # Collision right
            (direction_up       and     self.is_collision(right_point)) or 
            (direction_down     and     self.is_collision(left_point)) or 
            (direction_left     and     self.is_collision(up_point)) or 
            (direction_right    and     self.is_collision(down_point)),

            # Collision left
            (direction_down     and     self.is_collision(right_point)) or 
            (direction_up       and     self.is_collision(left_point)) or 
            (direction_right    and     self.is_collision(up_point)) or 
            (direction_left     and     self.is_collision(down_point)),
            
            # Move direction
            direction_left,
            direction_right,
            direction_up,
            direction_down,
            
            # Food location 
            self.fruit_position.x < self.snake_head.x,  # food left
            self.fruit_position.x > self.snake_head.x,  # food right
            self.fruit_position.y < self.snake_head.y,  # food up
            self.fruit_position.y > self.snake_head.y  # food down
            ]


        return np.array(state, dtype=int)


    def step(self, action)->tuple[np.array, int, bool, int, int]:
        # initialise
        game_over = False
        reward = 0
        
        # handling action
        self.last_direction = self.direction
        self.set_direction(action)

        # *****
        # invalid move: can't move the snake other way back : set the next direction equal to the last valid one
        if (self.directions_indices[self.last_direction] % 2 == self.directions_indices[self.direction] % 2) and (self.last_direction != self.direction):
            self.move_to(self.last_direction)
            self.direction = self.last_direction
            self.current_action = self.last_action
        else:
            self.move_to(self.direction)
            self.current_action = action
        self.last_action = self.current_action
        # *****


        # if there is a collision, end
        if self.is_collision(self.snake_head):
            game_over = True
            reward = -1
            return self.state, reward, game_over, self.score, self.current_action

        # Snake body growing mechanism
        self.snake_body.insert(0, self.snake_head)

        if self.snake_head == self.fruit_position:
            reward = 1
            self.score += reward*10
            self.fruit_spawn = False
        else:
            self.snake_body.pop()
                
        if not self.fruit_spawn:
            self.new_fruit()         
        self.fruit_spawn = True

        # render game UI
        self.render()        
        # Refresh game screen
        pygame.display.update()

        # frame per second
        self.fps.tick(self.snake_speed)
        
        return self.state, reward, game_over, self.score, self.current_action

import math
import random
import pygame
import neat

pygame.font.init()

GEN = 0

class Cube:
    rows = 20
    width = 500
    def __init__(self, start, x_dir=1, y_dir=0, color=(255, 0, 0)):
        self.pos = start
        self.x_dir = x_dir
        self.y_dir = y_dir
        self.color = color
        
    def move(self, x_dir, y_dir):
        self.x_dir = x_dir
        self.y_dir = y_dir
        self.pos = (self.pos[0] + self.x_dir, self.pos[1] + self.y_dir)

    def draw(self, win, eyes=False):
        dis = self.width // self.rows
        i = self.pos[0]
        j = self.pos[1]

        # making sure that we're drawing inside the grid, not overlapping the lines
        pygame.draw.rect(win, self.color, (i*dis+1, j*dis+1, dis-2, dis-2))
        if eyes:
            center = dis // 2
            rad = 3
            circleMid_1 = (i*dis+center-rad, j*dis+8)
            circleMid_2 = (i*dis+dis-rad*2, j*dis+8)
            pygame.draw.circle(win, (0, 0, 0), circleMid_1, rad)
            pygame.draw.circle(win, (0, 0, 0), circleMid_2, rad)


class Snake:
    # storing each cube into the body
    body = []
    # storing the pos when turning
    turns = {}
    def __init__(self, color, pos):
        self.color = color
        self.head = Cube(pos)
        self.body.append(self.head)
        
        # initializing the movement and direction
        self.x_dir = 0
        self.y_dir = 1
        
    def move(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        
            keys = pygame.key.get_pressed()
            
            for key in keys:
                if keys[pygame.K_LEFT]:
                    self.x_dir = -1
                    self.y_dir = 0
                    # adding the new position into the turns list
                    self.turns[self.head.pos[:]] = [self.x_dir, self.y_dir]
                elif keys[pygame.K_RIGHT]:
                    self.x_dir = 1
                    self.y_dir = 0
                    self.turns[self.head.pos[:]] = [self.x_dir, self.y_dir]
                elif keys[pygame.K_UP]:
                    self.x_dir = 0
                    self.y_dir = -1
                    self.turns[self.head.pos[:]] = [self.x_dir, self.y_dir]
                elif keys[pygame.K_DOWN]:
                    self.x_dir = 0
                    self.y_dir = 1
                    self.turns[self.head.pos[:]] = [self.x_dir, self.y_dir]

        for i, cube in enumerate(self.body):
            # the : means making a copy instead of modifying it directly
            p = cube.pos[:]
            # check if the current cube is at the turning position
            if p in self.turns:
                # choosing the turn
                turn = self.turns[p]
                # turn the list according to the position and the direction
                cube.move(turn[0], turn[1])
                # if it is the last cube in the body
                # then remove the turn (position and direction)
                if i == len(self.body) - 1:
                    self.turns.pop(p)
            else:
                # if moving left and we're moving out of the screen
                if cube.x_dir == -1 and cube.pos[0] <= 0:
                    # change the position to right of the screen
                    # change the x-coord to the last coord on the right
                    # while preserving the y-coord
                    cube.pos = (cube.rows-1, cube.pos[1])
                elif cube.x_dir == 1 and cube.pos[0] >= cube.rows - 1:
                    cube.pos = (0, cube.pos[1])
                elif cube.y_dir == 1 and cube.pos[1] >= cube.rows - 1:
                    cube.pos = (cube.pos[0], 0)
                elif cube.y_dir == -1 and cube.pos[1] <= 0:
                    cube.pos = (cube.pos[0], cube.rows - 1)
                else:
                    cube.move(cube.x_dir, cube.y_dir)
                        

    def reset(self, pos):
        self.head = Cube(pos)
        self.body = []
        self.body.append(self.head)
        self.turns = {}
        self.x_dir = 0
        self.y_dir = 1
        
    def addCube(self):
        tail = self.body[-1]
        x_dir = tail.x_dir
        y_dir = tail.y_dir
        
        # if the snake is moving right
        if x_dir == 1 and y_dir == 0:
            # add the new cube to the end on the left
            self.body.append(Cube((tail.pos[0]-1, tail.pos[1])))
        elif x_dir == -1 and y_dir == 0:
            # add the new cube to the end on the right
            self.body.append(Cube((tail.pos[0]+1, tail.pos[1])))
        elif x_dir == 0 and y_dir == 1:
            # add the new cube to the end at the bottom
            self.body.append(Cube((tail.pos[0], tail.pos[1]-1)))
        elif x_dir == 0 and y_dir == -1:
            # add the new cube to the end on the top
            self.body.append(Cube((tail.pos[0], tail.pos[1]+1)))

        self.body[-1].x_dir = x_dir
        self.body[-1].y_dir = y_dir
        

    def draw(self, win):
        for i, cube in enumerate(self.body):
            if i == 0:
                cube.draw(win, True)
            else:
                cube.draw(win)




def drawGrid(width, rows, win):
    space_between = width // rows
    
    x = 0
    y = 0

    for i in range(rows):
        x += space_between
        y += space_between
        
        # draw horizontal lines
        pygame.draw.line(win, (255, 255, 255),  (x, 0), (x, width))
        # draw vertical lines
        pygame.draw.line(win, (255, 255, 255),  (0, y), (width, y))

def drawWindow(win):
    global ROWS, width, window, snake, snack
    win.fill((0, 0, 0))
    drawGrid(width, ROWS, win)
    snake.draw(win)
    snack.draw(win)

    pygame.display.update()

def randomSnack(rows, s):
    global ROWS
    positions = s.body
    while True:
        x = random.randrange(rows)
        y = random.randrange(rows)
        # check if the newly generated is on top of the snake or not
        if len(list(filter(lambda z: z.pos == (x, y), positions))) > 0:
            continue
        else:
            break
    return (x, y)
def message_box():
    pass

def main():
    global width, ROWS, window, snake, snack
    width = 500
    ROWS = 20
    window = pygame.display.set_mode((width, width))
    snake = Snake((255, 0, 0), (10, 10))

    snack = Cube(randomSnack(ROWS, snake), color=(0, 255, 0))
    clock = pygame.time.Clock()
    
    run = True
    while run:
        # delay if the game is going to fast
        pygame.time.delay(50)
        
        # frame is limited to 10fps
        clock.tick(10)
        snake.move()

        if snake.body[0].pos == snack.pos:
            snake.addCube()
            snack = Cube(randomSnack(ROWS, snake), color=(0, 255, 0))

        for x in range(len(snake.body)):
            if snake.body[x].pos in list(map(lambda z: z.pos, snake.body[x+1:])):
                print('Score: ', len(snake.body))
                snake.reset((10, 10))
                break

        drawWindow(window)
        
main()
import pygame
import sys
import time

BLACK = (0, 0, 0)
WHITE = (200, 200, 200)
WINDOW_HEIGHT = 800
WINDOW_WIDTH = 800
from game_rules import *


def main_2():
    global SCREEN, CLOCK
    pygame.init()
    SCREEN = pygame.display.set_mode((WINDOW_HEIGHT, WINDOW_WIDTH))
    CLOCK = pygame.time.Clock()
    SCREEN.fill(BLACK)
    drawGrid()
    side = 1

    while True:
        # board.print_board(board.board)
        # r = int(input("row: "))
        # c = int(input("column: "))
        # board.add_position(r, c, 1)
        #
        # board.is_there_five_in_row(1, (r, c))
        # l = [board.board]
        #
        # l = tf.expand_dims(l, axis=-1)
        # value = model.predict(l)
        # l = np.asarray(l) * -1
        # policy_prediction = (policy_model.predict(l))
        # policy_prediction = np.ndarray.tolist(policy_prediction[0])
        # policy_prediction = search.normalise_policy(policy_prediction)
        #
        # print("MATRIX", l)
        # print(value)
        #
        # g.update_predictions(policy_prediction)
        # g.update_board(board.board)
        # drawGrid()
        # g.draw_board(max(policy_prediction))
        #
        #
        #
        # pygame.display.update()
        #
        #
        # board.print_board(board.board)
        # r = int(input("row: "))
        # c = int(input("column: "))
        # board.add_position(r, c, -1)
        # board.is_there_five_in_row(-1, (r, c))
        #
        # l = [board.board]
        # l = np.asarray(l)
        # l = tf.expand_dims(l, axis=-1)
        # policy_prediction = (policy_model.predict(l))
        # policy_prediction = np.ndarray.tolist(policy_prediction[0])
        # policy_prediction = search.normalise_policy(policy_prediction)
        # value = model.predict(l)
        # print(value)
        # g.update_predictions(policy_prediction)
        # g.update_board(board.board)
        # drawGrid()
        # g.draw_board(max(policy_prediction))

        if side == 1:
            search.MCTS(side)
            g.update_board(board.board)
            g.draw_board()
            side *= -1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONUP:
                pos = pygame.mouse.get_pos()
                row = int((800 - pos[1])//53.3333)
                column = int((pos[0])//53.3333)
                if board.board[row][column] == 0:
                    board.add_position(row, column, side)
                    side *= -1

                g.update_board(board.board)
                g.draw_board()

        # g.update_board(board.board)

        pygame.display.update()


def get_move_from_pos(pos):
    return (10, 10)

def drawGrid():
    blockSize = WINDOW_WIDTH/15 #Set the size of the grid block
    for x in range(WINDOW_WIDTH):
        for y in range(WINDOW_HEIGHT):
            rect = pygame.Rect(x*blockSize, y*blockSize,
                               blockSize, blockSize)
            pygame.draw.rect(SCREEN, WHITE, rect, 1)


class Square():
    def __init__(self, coords):
        self.stone = None
        self.policy = None
        self.position = coords


class Grid():
    def __init__(self):
        self.squares = []
        self.first_square_pos = [0, 800]

    def init_squares(self):
        for i in range(15):
            for x in range(15):
                square = Square((self.first_square_pos[0]+x*(800/15), self.first_square_pos[1]-i*(800/15)))
                self.squares.append(square)

    def draw_board(self, max_policy=None):
        for s in self.squares:
            if s.policy != None:
                pygame.draw.rect(SCREEN, self.number_to_colour(s.policy, max_policy), [s.position[0], s.position[1]- 800/15, 800/15 - 4, 800/15 - 4])
            if s.stone == 1:
                pygame.draw.circle(SCREEN, (195, 0, 0), (s.position[0]+800/30, s.position[1]-800/30), 16)
            elif s.stone == -1:
                pygame.draw.circle(SCREEN, (0, 50, 183), (s.position[0]+800/30, s.position[1]-800/30), 16)

    def update_board(self, matrix_repr):
        index = 0
        for row in matrix_repr:
            for column in row:
                if int(column) == 1:
                    self.squares[index].stone = 1
                elif int(column) == -1:
                    self.squares[index].stone = -1
                index += 1

    def update_predictions(self, predictions):
        for i, p in enumerate(predictions):
            self.squares[i].policy = p

    def number_to_colour(self, number, max_policy):
        variable = 255/max_policy
        x = int(number*variable)
        return (x,x,x)



g = Grid()
g.init_squares()
main_2()
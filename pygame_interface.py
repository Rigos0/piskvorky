""""" Primitivni graficke interface pro hru v Pygame """""
""""" Momentalne bude "zamrzat" vzdy pri propoctu variant! """""


import pygame
import sys
# importujeme vse z searche a pravidel
from game_rules import *


BLACK = (0, 0, 0)
WHITE = (200, 200, 200)
WINDOW_HEIGHT = 800
WINDOW_WIDTH = 800


def main():
    global SCREEN, CLOCK
    pygame.init()
    SCREEN = pygame.display.set_mode((WINDOW_HEIGHT, WINDOW_WIDTH))
    CLOCK = pygame.time.Clock()
    SCREEN.fill(BLACK)
    pygame.display.set_caption("PyŠkvor Beta")
    side = 1
    # namaluj sit
    draw_grid()

    # hlavni Pygame loop
    while True:
        # pokud je na tahu hrac 1, proved search
        if side == 1:
            best_move = search.MCTS(side)
            board.add_position(best_move[0], best_move[1], side)

            g.update_board(board.board)
            g.draw_board()
            side *= -1

        for event in pygame.event.get():
            # kliknuti na krizek vypne program
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            # kliknuti mysi
            if event.type == pygame.MOUSEBUTTONUP:
                # najdi pozici kliku
                pos = pygame.mouse.get_pos()
                # najdi na jake policko jsme klikli
                row = int((800 - pos[1])//53.3333)
                column = int((pos[0])//53.3333)
                # pokud je kliknuti mozny tah, tak ho zahraj
                if board.board[row][column] == 0:
                    board.add_position(row, column, side)
                    side *= -1

            g.update_board(board.board)
            g.draw_board()

        pygame.display.update()


# malovani site
def draw_grid():
    block_size = WINDOW_WIDTH/15
    for x in range(WINDOW_WIDTH):
        for y in range(WINDOW_HEIGHT):
            rect = pygame.Rect(x*block_size, y*block_size,block_size, block_size)
            pygame.draw.rect(SCREEN, WHITE, rect, 1)

# kazdy ctverecek bude jeden objekt
class Square():
    def __init__(self, coords):
        self.stone = None
        self.policy = None
        self.position = coords


class Grid():
    def __init__(self):
        self.squares = []
        self.first_square_pos = [0, 800]

    # inicializuj vsechny ctverecky
    def init_squares(self):
        for i in range(15):
            for x in range(15):
                square = Square((self.first_square_pos[0]+x*(800/15), self.first_square_pos[1]-i*(800/15)))
                self.squares.append(square)

    # namaluj vsechny ctverecky
    def draw_board(self, max_policy=None):
        for s in self.squares:
            # vizualizace outputu policy funkce
            if s.policy != None:
                pygame.draw.rect(SCREEN,
self.number_to_colour(s.policy, max_policy), [s.position[0], s.position[1] - 800/15, 800/15 - 4, 800/15 - 4])
            # namaluj cerveny kolecka
            if s.stone == 1:
                pygame.draw.circle(SCREEN, (195, 0, 0), (s.position[0]+800/30, s.position[1]-800/30), 16)
            # namaluj modry kolecka
            elif s.stone == -1:
                pygame.draw.circle(SCREEN, (0, 50, 183), (s.position[0]+800/30, s.position[1]-800/30), 16)

    # update pozici v grafickem prostredi po provedeni tahu
    def update_board(self, matrix_repr):
        index = 0
        for row in matrix_repr:
            for column in row:
                if int(column) == 1:
                    self.squares[index].stone = 1
                elif int(column) == -1:
                    self.squares[index].stone = -1
                index += 1

    # update vizualizaci policy funkce
    def update_predictions(self, predictions):
        for i, p in enumerate(predictions):
            self.squares[i].policy = p

    # jak vyrazne policko zabarvime
    def number_to_colour(self, number, max_policy):
        variable = 255/max_policy
        x = int(number*variable)
        return (x,x,x)


g = Grid()
g.init_squares()

if __name__ == "__main__":
    main()

import numpy as np
import math


class Board:
    def __init__(self):
        self.board = np.zeros((15, 15))

    def add_position(self, row, column, side_to_move):
        self.board[row][column] = side_to_move

    def new_game(self):
        self.board = np.zeros((15, 15))

    @staticmethod
    def remove_directions(to_remove, directions):
        to_be_deleted = []
        for i in directions:
            if i[0] == to_remove[0] or i[1] == to_remove[1]:
                to_be_deleted.append(i)

        for d in to_be_deleted:
            directions.remove(d)

        return directions

    def get_directons(self, square):
        directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]

        if square[1] == 0:
            directions = self.remove_directions((None, -1), directions)

        elif square[1] == 14:
            directions = self.remove_directions((None, 1), directions)

        if square[0] == 0:
            directions = self.remove_directions((-1, None), directions)

        elif square[0] == 14:
            directions = self.remove_directions((1, None), directions)

        return directions

    def is_there_five_in_row(self, side_to_move, added_move):
        directions = self.get_directons(added_move)
        checked_directions = []

        for d in directions:
            row_length = 1

            if d in checked_directions:
                continue
            row_length = self.how_many_in_row(added_move, d, side_to_move, row_length)
            checked_directions.append(d)

            # opposite direction
            if (d[0]*-1, d[1]*-1) in directions:
                row_length = self.how_many_in_row(added_move, (d[0]*-1, d[1]*-1), side_to_move, row_length)
                checked_directions.append((d[0]*-1, d[1]*-1))

            if row_length == 5:
                return True

    def how_many_in_row(self, added_move, d, side_to_move, row_length):
        next_square = added_move
        for i in range(5):
            next_square = (next_square[0] + d[0], next_square[1] + d[1])
            if next_square[0] < 0 or next_square[0] > 14 or next_square[1] < 0 or next_square[1] > 14:
                break

            if int(self.board[next_square[0]][next_square[1]]) == side_to_move:
                row_length += 1
            else:
                break

        return row_length

    def rotate_board(self):
        return np.fliplr(board.board)

    def print_board(self, board):
        for i in range(15):
            string = "|"
            for square in board[14-i]:
                if int(square) == 0:
                    string += "_|"
                if int(square) == 1:
                    string += "0|"
                if int(square) == -1:
                    string += "X|"
            print(string)


board = Board()


def play():
    while True:
        board.print_board(board.board)
        r = int(input("row: "))
        c = int(input("column: "))
        board.add_position(r, c, 1)
        board.is_there_five_in_row(1, (r,c))
        board.print_board(board.board)
        r = int(input("row: "))
        c = int(input("column: "))
        board.add_position(r, c, -1)
        board.is_there_five_in_row(-1, (r,c))
        print("rotating")
        board.print_board(board.rotate_board())

# play()

print(board.board.shape)
class Dataset:
    pass



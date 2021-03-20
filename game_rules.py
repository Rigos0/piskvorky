import numpy as np
import math
from tensorflow import keras
import tensorflow as tf
import time



gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
model = keras.models.load_model('value_network')
policy_model  = keras.models.load_model('policy_model')





class Board:
    def __init__(self):
        self.board = np.zeros((15, 15))

    def add_position(self, row, column, side_to_move):
        self.board[row][column] = side_to_move

    def index_to_move(self, index):
        row = index // 15
        column = index%15

        return (row, column)

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


class Search:
    def normalise_policy(self, policy):
        policy = policy
        # Remove illegal moves from policy
        index = 0
        for row in board.board:
            for column in row:
                if int(column) != 0:
                    policy[index] = 0

                index += 1

        return policy

    def get_matrix_representation(self, move_number):
        # if it is player 1 to move
        if move_number % 2 == 0:
            matrix = board.board
            yield matrix
        else:
            # thanks to the simple board representation, this is very elegant
            matrix = board.board * -1
            yield matrix
        # # rotaterotate rotate trotate rotate rotoaoteao and rottate
        # for i in range(1,4):
        #     yield np.rot90(matrix, k=i)
        # matrix = np.flip(matrix)
        # yield matrix
        # for i in range(1, 4):
        #     yield np.rot90(matrix, k=i)


search = Search()
board = Board()

def play():
    move_number = 0
    while True:
        board.print_board(board.board)
        r = int(input("row: "))
        c = int(input("column: "))
        board.add_position(r, c, 1)
        l = [board.board]
        l = np.asarray(l)
        l = l*-1
        # prediction = (model.predict(tf.expand_dims(l, axis=-1)))
        policy_prediction = (policy_model.predict(tf.expand_dims(l, axis=-1)))
        policy_prediction = np.ndarray.tolist(policy_prediction[0])



        # print(prediction[0][1], prediction[0][0])

        board.is_there_five_in_row(1, (r,c))


        board.print_board(board.board)
        r = int(input("row: "))
        c = int(input("column: "))
        board.add_position(r, c, -1)
        board.is_there_five_in_row(-1, (r,c))

        l = [board.board]
        l = np.asarray(l)
        policy_prediction = (policy_model.predict(tf.expand_dims(l, axis=-1)))
        policy_prediction = np.ndarray.tolist(policy_prediction[0])
        print(policy_prediction)
        print("best_move", policy_prediction.index(max(policy_prediction)))

        # print(model.predict(tf.expand_dims(l, axis=-1)))

def test_speed():
    l = [board.board]

    l = np.asarray(l)
    l = tf.expand_dims(l, axis=-1)
    value = policy_model(l, training=False)
    start = time.time()

    for x in range(1000):
        l = [board.board]

        l = np.asarray(l)
        l = tf.expand_dims(l, axis=-1)
        value = policy_model(l, training=False)



    end = time.time()
    print(end-start)
# test_speed()
# play()

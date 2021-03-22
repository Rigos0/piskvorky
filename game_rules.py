import numpy as np
import math
# from tensorflow import keras
# import tensorflow as tf
import time



# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         # Currently, memory growth needs to be the same across GPUs
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         # Memory growth must be set before GPUs have been initialized
#         print(e)
# model = keras.models.load_model('value_network')
# policy_model  = keras.models.load_model('policy_model')




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
    def __init__(self):
        self.root_node = None
        self.side_to_move = 1


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

    def get_policy(self):
        l = [board.board]
        l = np.asarray(l)
        # l = l * -1
        policy_prediction = (policy_model.predict(tf.expand_dims(l, axis=-1)))
        policy_prediction = np.ndarray.tolist(policy_prediction[0])

        return policy_prediction


    def initialise_search_tree(self):
        self.root_node = Node(None)


    def add_nodes(self, policy, parent):
        for row_index, row in enumerate(board.board):
            for column_index, column in enumerate(row):
                if not column:
                    node = Node(policy[row_index*15 + column_index], parent)
                    parent.children.append(node)

    # select which leaf node to expand
    def selection(self, node):
        for child in node.children:
            # find node with the highest ucb
            # pseudo - dodelat
            highest_ucb_node = node

        if highest_ucb_node.children:
            # the node is not a leaf node, so traverse the tree further
            highest_ucb_node = self.selection(highest_ucb_node)

        return highest_ucb_node

    #
    def expand(self, from_node):
        actions = get_actions()

        for action in actions:
            from_node.children.append(Node(policy, from_node.side*-1))


    def MCTS(self):
        self.selection(self.root_node)

        pass



class Node:
    def __init__(self, policy, move):
        self.number_of_visits = int
        self.number_of_wins = int
        self.initial_policy = policy
        self.children = []
        self.side_to_move = None
        self.move = move









class StaticEvaluation:
    def evaluate_position(self, position):
        streaks = {
            "three_half_open": 0,
            "three_open": 0,
            "four_half_open": 0,
            "four_open": 0,
            "five": 0,
            "three_half_open_2": 0,
            "three_open_2": 0,
            "four_half_open_2": 0,
            "four_open_2": 0,
            "five_2": 0
        }
        radky, diagonaly_nahoru = self.vrat_radky_a_diagonalu(position)
        position = np.rot90(position)
        sloupce, diagonaly_dolu = self.vrat_radky_a_diagonalu(position)
        vsechno = (radky, sloupce, diagonaly_dolu, diagonaly_nahoru)
        for i in vsechno:
            for x in i:
                streaks = self.one_dimension_eval(x, streaks)


        return streaks

    def vrat_radky_a_diagonalu(self, position):
        radky = []

        diagonaly_nahoru = []
        for i in range(29):
            diagonaly_nahoru.append([])
        for radek_index, radek in enumerate(position):
            radky.append(radek)
            for pozice_index, pozice in enumerate(radek):
                diagonaly_nahoru[self.do_jaky_diagonaly(radek_index, pozice_index)].append(pozice)

        return radky, diagonaly_nahoru

    def do_jaky_diagonaly(self, radek_index, pozice_index):
        index = pozice_index - radek_index
        if index < 0:
            index = 14 + index*-1
        return index



    def one_dimension_eval(self, list_of_positions, streaks):
        current_side = None
        row_length = 0


        for i, position in enumerate(list_of_positions):
            if position != 0:
                # prodluzujeme nalezeny streak
                if position == current_side:
                    row_length += 1
                # zacina novy streak
                else:
                    if row_length >= 3:
                        streaks = self.get_streak_info(list_of_positions, row_length,
                                                       streak_start_index, i-1, current_side, streaks)
                    current_side = position
                    streak_start_index = i
                    row_length = 1

            else:
                if row_length >= 3:
                    streaks = self.get_streak_info(list_of_positions, row_length, streak_start_index, i-1, current_side,
                                                   streaks)
                current_side = None
                row_length = 0

        if row_length >= 3:
            streaks = self.get_streak_info(list_of_positions, row_length, streak_start_index,
                                           len(list_of_positions) - 1, current_side, streaks)

        return streaks

    def get_streak_info(self, list_of_positions, streak_length, streak_start_index, streak_end_index, side,
                        streaks):
        list_length = len(list_of_positions)
        closed_in_front = False
        closed_from_back = False
        # zjistime jestli je streak uzavreny, polootevreny nebo otevreny
        if streak_length == 3 or streak_length == 4:
            # check for space in front
            if streak_start_index != 0:
                if list_of_positions[streak_start_index-1] != 0:
                    closed_in_front = True
            else:
                closed_in_front = True
            # check for space behind streak
            if streak_end_index != list_length-1:
                if list_of_positions[streak_end_index+1] != 0:
                    closed_from_back = True
            else:
                closed_from_back = True
        # vyhrava pouze presne 5 kamenu v rade, proto ted zjistime jestli nam streak dlouhy 4 nesousedi
        # ob jednu mezeru s dalsim kamenem te same barvy
        if streak_length == 4:
            # zkontrolujeme ob mezeru pred
            if streak_start_index > 1 and not closed_in_front:
                if list_of_positions[streak_start_index-2] == side:
                    closed_in_front = True
            # zkontrolujeme mezeru za streakem
            if streak_end_index < list_length - 2 and not closed_from_back:
                if list_of_positions[streak_end_index+2] == side:
                    closed_from_back = True

        if side == 1:
            if streak_length == 5:
                streaks["five"] += 1
            elif streak_length == 4 and (closed_from_back ^ closed_in_front):
                streaks["four_half_open"] += 1
            elif streak_length == 4 and (not closed_from_back and not closed_in_front):
                streaks["four_open"] += 1
            elif streak_length == 3 and (closed_from_back ^ closed_in_front):
                streaks["three_half_open"] += 1
            elif streak_length == 3 and (not closed_from_back and not closed_in_front):
                streaks["three_open"] += 1
        else:
            if streak_length == 5:
                streaks["five"] += 1
            elif streak_length == 4 and (closed_from_back ^ closed_in_front):
                streaks["four_half_open_2"] += 1
            elif streak_length == 4 and (not closed_from_back and not closed_in_front):
                streaks["four_open_2"] += 1
            elif streak_length == 3 and (closed_from_back ^ closed_in_front):
                streaks["three_half_open_2"] += 1
            elif streak_length == 3 and (not closed_from_back and not closed_in_front):
                streaks["three_open_2"] += 1

        return streaks


s = StaticEvaluation()
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
        # policy_prediction = (policy_model.predict(tf.expand_dims(l, axis=-1)))
        # policy_prediction = np.ndarray.tolist(policy_prediction[0])
        start = time.time()
        for i in range(1000):
            eval = s.evaluate_position(board.board)
        end = time.time()
        print(eval)
        print("time: ", end - start)

        # print(prediction[0][1], prediction[0][0])

        board.is_there_five_in_row(1, (r,c))


        board.print_board(board.board)
        r = int(input("row: "))
        c = int(input("column: "))
        board.add_position(r, c, -1)
        board.is_there_five_in_row(-1, (r,c))

        l = [board.board]
        l = np.asarray(l)
        # policy_prediction = (policy_model.predict(tf.expand_dims(l, axis=-1)))
        # policy_prediction = np.ndarray.tolist(policy_prediction[0])
        # print(policy_prediction)
        # print("best_move", policy_prediction.index(max(policy_prediction)))

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

play()

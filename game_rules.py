import numpy as np
import math
from tensorflow import keras
import tensorflow as tf
import time
import random

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
        column = index % 15

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
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

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
            if (d[0] * -1, d[1] * -1) in directions:
                row_length = self.how_many_in_row(added_move, (d[0] * -1, d[1] * -1), side_to_move, row_length)
                checked_directions.append((d[0] * -1, d[1] * -1))

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
            for square in board[14 - i]:
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
        self.tree_path = []
        self.how_many_moves_consider = 20
        self.policy_model = keras.models.load_model('policy_model')

    @staticmethod
    def normalise_policy(policy):
        # Remove illegal moves from policy
        index = 0
        for row in board.board:
            for column in row:
                if int(column) != 0:
                    policy[index] = 0

                index += 1

        return policy

    def get_policy(self, side_to_move):
        position = [board.board]
        position = np.asarray(position) * side_to_move
        policy_prediction = (self.policy_model.predict(tf.expand_dims(position, axis=-1)))
        policy_prediction = np.ndarray.tolist(policy_prediction[0])

        return policy_prediction

    # vyplivne x nejlepsich tahu podle neural networku
    def get_top_moves(self, policy):
        policies = []
        indicies = []
        for i in range(self.how_many_moves_consider):
            maxi = max(policy)
            max_index = policy.index(maxi)
            move = board.index_to_move(max_index)
            if board.board[move[0]][move[1]] == 0:
                policies.append(maxi)
                indicies.append(max_index)
            policy[max_index] = 0

        policies_sum = sum(policies)
        ratio = 100/policies_sum
        for i in range(len(policies)):
            policies[i] *= ratio
            yield policies[i], indicies[i]

    def return_to_root_position(self):
        for node in self.tree_path:
            board.board[node.move[0]][node.move[1]] = 0
        self.tree_path.clear()

    def initialise_search_tree(self, side_to_move):
        self.root_node = Node(None, None, side_to_move)

    # select which leaf node to expand
    def selection(self, node):
        if not node.children:
            if node.visited:
                self.expand(node)
                self.return_to_root_position()

            else:
                static_evaluation, win = static_eval.evaluate_position(board.board, node.side)
                self.backpropagate(static_evaluation,win)
                node.visited = True

        else:
            ucbs = []
            for child in node.children:
                # find node with the highest ucb
                ucb = child.get_upper_confidence_bound(self.root_node.visits)
                ucbs.append(ucb)

            # choose the node with the highest ucbs
            next_node = node.children[ucbs.index(max(ucbs))]
            self.tree_path.append(next_node)

            board.add_position(next_node.move[0], next_node.move[1], node.side)
            self.selection(next_node)

    # udelame deti a pridame je na konec stromu
    def expand(self, from_node):

        policies = self.get_policy(from_node.side)
        for policy, move_index in self.get_top_moves(policies):
            newborn_child = Node(policy, move_index, from_node.side * -1)
            from_node.children.append(newborn_child)

    # propagate the static eval value back to the tree
    def backpropagate(self, value, win):
        if win:
            print("I SEEEEEEEEEEEEEE WIIIIIIIIIIIIIIn")
            if self.tree_path:
                self.tree_path[-1].value = 1000
                if len(self.tree_path) > 1:
                    self.tree_path[-2].value = -1000

        for i in range(len(self.tree_path)):
            # colours are ascillating (win for one side is a lost position for the other)
            index = -i - 1
            self.tree_path[index].visits += 1
            self.tree_path[index].value += value
            value = 1 - value
            # return the move
            # (this could be done in the function return to root node, but it would
            # be slower
            board.board[self.tree_path[index].move[0]][self.tree_path[index].move[1]] = 0
        self.root_node.visits += 1
        self.tree_path.clear()

    def find_the_best_move(self):
        for node in self.root_node.children:
            pass

        return node.move

    def MCTS(self, side_to_move):
        self.initialise_search_tree(side_to_move)
        for i in range(500):
            self.selection(self.root_node)

        max_visits, best_move = 0, None
        for child in self.root_node.children:
            if child.visits > max_visits:
                max_visits = child.visits
                best_move = child.move
        board.add_position(best_move[0], best_move[1], side_to_move)
        return best_move


# z Nodes budeme stavet search tree
class Node:
    def __init__(self, policy, move, side):
        self.visits = 0.001
        self.value = 0
        self.initial_policy = policy
        self.children = []
        self.side = side
        if move:
            self.move = (move // 15, move % 15)
        self.visited = False

    # kazda node ma svuj upper confidence bound, podle ktereho se budeme rozhodovat jakou
    # node zvolit
    def get_upper_confidence_bound(self, N):
        expl_constant = 2
        exploration = expl_constant * math.sqrt((math.log(N)) / self.visits)

        return self.value/self.visits + exploration * self.initial_policy


class StaticEvaluation:
    def evaluate_position(self, position, side_to_move):
        # vysvetleni streaku
        # streaks = {
        #     "three_half_open": 0,
        #     "three_open": 0,
        #     "four_half_open": 0,
        #     "four_open": 0,
        #     "five": 0,
        #     "three_half_open_2": 0,
        #     "three_open_2": 0,
        #     "four_half_open_2": 0,
        #     "four_open_2": 0,
        #     "five_2": 0
        # }
        streaks = [[0,0,0,0,0], [0,0,0,0,0]]
        radky, diagonaly_nahoru = self.vrat_radky_a_diagonalu(position)
        position = np.rot90(position)
        sloupce, diagonaly_dolu = self.vrat_radky_a_diagonalu(position)
        vsechno = (radky, sloupce, diagonaly_dolu, diagonaly_nahoru)
        for i in vsechno:
            for x in i:
                streaks = self.one_dimension_eval(x, streaks)

        win = self.vyhrali_jsme_otaznik(side_to_move, streaks)
        cislo = self.udelej_cislo_z_nalezeneho_infa_o_pozici(streaks, side_to_move)
        return cislo, win

    def side_to_index(self, side):
        if side == 1:
            side_index = 0
        else:
            side_index = 1
        return side_index

    def udelej_cislo_z_nalezeneho_infa_o_pozici(self, info, side_to_move):
        evaluation = random.uniform(0,1)

        return 0.5

    def vyhrali_jsme_otaznik(self, side, info):
        index = self.side_to_index(side)
        vyhra = False
        if info[index][3] > 0 or info[index][4] > 0:
            vyhra = True
        return vyhra


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

    @staticmethod
    def do_jaky_diagonaly(radek_index, pozice_index):
        index = pozice_index - radek_index
        if index < 0:
            index = 14 + index * -1
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
                                                       streak_start_index, i - 1, current_side, streaks)
                    current_side = position
                    streak_start_index = i
                    row_length = 1

            else:
                if row_length >= 3:
                    streaks = self.get_streak_info(list_of_positions, row_length, streak_start_index, i - 1,
                                                   current_side,
                                                   streaks)
                current_side = None
                row_length = 0

        if row_length >= 3:
            streaks = self.get_streak_info(list_of_positions, row_length, streak_start_index,
                                           len(list_of_positions) - 1, current_side, streaks)

        return streaks

    @staticmethod
    def get_streak_info(list_of_positions, streak_length, streak_start_index, streak_end_index, side,
                        streaks):
        list_length = len(list_of_positions)
        closed_in_front = False
        closed_from_back = False
        # zjistime jestli je streak uzavreny, polootevreny nebo otevreny
        if streak_length == 3 or streak_length == 4:
            # check for space in front
            if streak_start_index != 0:
                if list_of_positions[streak_start_index - 1] != 0:
                    closed_in_front = True
            else:
                closed_in_front = True
            # check for space behind streak
            if streak_end_index != list_length - 1:
                if list_of_positions[streak_end_index + 1] != 0:
                    closed_from_back = True
            else:
                closed_from_back = True
        # vyhrava pouze presne 5 kamenu v rade, proto ted zjistime jestli nam streak dlouhy 4 nesousedi
        # ob jednu mezeru s dalsim kamenem te same barvy
        if streak_length == 4:
            # zkontrolujeme ob mezeru pred
            if streak_start_index > 1 and not closed_in_front:
                if list_of_positions[streak_start_index - 2] == side:
                    closed_in_front = True
            # zkontrolujeme mezeru za streakem
            if streak_end_index < list_length - 2 and not closed_from_back:
                if list_of_positions[streak_end_index + 2] == side:
                    closed_from_back = True

        if side == 1:
            index = 0
        else:
            index = 1

        if streak_length == 5:
            streaks[index][4] += 1
        elif streak_length == 4 and (closed_from_back ^ closed_in_front):
            streaks[index][2] += 1
        elif streak_length == 4 and (not closed_from_back and not closed_in_front):
            streaks[index][3] += 1
        elif streak_length == 3 and (closed_from_back ^ closed_in_front):
            streaks[index][0] += 1
        elif streak_length == 3 and (not closed_from_back and not closed_in_front):
            streaks[index][1] += 1
        return streaks


static_eval = StaticEvaluation()
search = Search()
board = Board()

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
    print(end - start)


def play(side):
    board.print_board(board.board)
    if side == 1:
        print(search.MCTS(side))
    else:
        r = int(input("row: "))
        c = int(input("column: "))
        board.add_position(r, c, side)
    # l = [board.board]
    # l = np.asarray(l)
    # l = l * -1
    # prediction = (model.predict(tf.expand_dims(l, axis=-1)))
    # policy_prediction = (policy_model.predict(tf.expand_dims(l, axis=-1)))

    # policy_prediction = np.ndarray.tolist(policy_prediction[0]
    # print(prediction[0][1], prediction[0][0])
    # search.MCTS(side)
    print(static_eval.evaluate_position(board.board, side))

def main():
    side = 1
    while True:
        play(side)
        side *= -1







main()

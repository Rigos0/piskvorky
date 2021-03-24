from game_rules import *
import os


class Dataset:
    # replay a game from given file, count number of moves
    def replay_game(self, file_name):
        board.new_game()
        game = open(directory_in_str + "/" + file_name)
        number_of_moves = 0
        side_to_move = 1
        for line in game:
            if line.startswith("P") or line.startswith("<"):
                continue
            # if not end of the game
            if "zip" not in line:
                try:
                    move = self.convert_notation(line)
                except UnboundLocalError:
                    return None, None
                board.add_position(move[0], move[1], side_to_move)
                # print(board.print_board(board.board))
                number_of_moves += 1

                # if the game ends decisively, add it to the dataset
                if board.is_there_five_in_row(side_to_move, move):
                    # reset the board, so we can replay the game again
                    board.new_game()
                    positions, values = self.add_game_to_dataset(game, side_to_move, number_of_moves)
                    yield positions, values
                    return None, None

                # flip the side to move after every move
                side_to_move *= -1
            # if the game does not end decisively, don't add it to the dataset
            else:
                board.new_game()
                break


    # get move notation in tuple from a string line (.psq files)
    def convert_notation(self, line):
        row = ""
        column = ""
        for index, char in enumerate(line):
            if char != ",":
                row += char
            else:
                i = index
                break
        column += line[i+1]
        if line[i+2] != ",":
            column += line[i+2]

        row = int(row)
        column = int(column)

        return (row - 1, column -1)

    # get labels for one game.
    # in every position, it will be players "1" move
    # so we will flip the pieces and the label in every position where it is player "-1" to play
    def get_values(self, number_of_moves, result):
        values = []
        oscillator = result*-1
        # create a list [1, -1, 1, -1 ....]
        # if player 1 wins, the list will start with -1
        if result == -1:
            result = 0
        else:
            result = 1
        for i in range(number_of_moves):
            for x in range(4): # z kazdy pozice vytvorime 8 diky rotacim
                values.append(result)

            if result == 0:
                result = 1
            else:
                result = 0
            # oscillator *= -1

        # we will assume that positions towards the end of the game are more winning, so they will have
        # # a higher value
        # divider = number_of_moves/10
        #
        #
        # for i in range(len(values)):
        #     if values[i] == -1:
        #         values[i] = - math.log10((i + 1) / divider + 1)
        #     else:
        #         values[i] =  math.log10((i + 1) / divider + 1)
        #     # values are also normalised to fit between 0 and 1
        return values



    # replay the game again and create an array with the positions
    def add_game_to_dataset(self, game, result, number_of_moves):
        positions = []
        policies = []
        game.seek(0)
        side_to_move = 1
        move_number = 0
        for line in game:
            # skip the first line
            if line.startswith("P"):
                continue
            if "zip" not in line:
                move = self.convert_notation(line)
                # play the move
                # do datasetu chceme az pozice od 11. tahu
                if move_number > 10:

                    # get an array representation of the position
                    for matrix in search.get_matrix_representation(move_number):
                        positions.append(matrix)
                    policy = self.move_to_category(move)
                    policies.append(policy)
                board.add_position(move[0], move[1], side_to_move)
                move_number += 1



                if board.is_there_five_in_row(side_to_move, move):
                    break

                side_to_move *= -1
            else:
                break

        board.new_game()

        values = np.asarray(policies)
        positions = np.asarray(positions)

        return positions, values

    def move_to_category(self, move):
        category = move[0] * 15 + move[1]

        return category




d = Dataset()

directory_in_str = r"C:\Users\Rigos\Documents\piskvorky\Standard"
directory = os.fsencode(directory_in_str)
files = []
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".psq"):
        files.append(filename)
        continue
    else:
        continue

print(files)

positions_mega_list = []
values_mega_list = []
for file in files:

    for positions, values in d.replay_game(file):
        for p in positions:
            positions_mega_list.append(p)

        for i in values:
            values_mega_list.append(i)

    if len(positions_mega_list) != len(values_mega_list):
        print(file)
        print(len(positions_mega_list), len(values_mega_list))


positions_mega_list = np.asarray(positions_mega_list)
values_mega_list = np.asarray(values_mega_list)
print(positions_mega_list.shape, values_mega_list.shape)
print(positions_mega_list)

np.save('positons_policies1.npy', positions_mega_list)
np.save('policies1.npy', values_mega_list)
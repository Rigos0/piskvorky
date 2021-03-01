from game_rules import *
import os


class Dataset:
    # replay a game from given file, count number of moves
    def replay_game(self, file_name):
        game = open(directory_in_str + "/" + file_name)
        number_of_moves = 0
        side_to_move = 1
        for line in game:
            if line.startswith("P") or line.startswith("<"):
                continue
            # if not end of the game
            if "zip" not in line:
                move = self.convert_notation(line)
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
        for i in range(number_of_moves):
            values.append(oscillator)
            oscillator *= -1

        # we will assume that positions towards the end of the game are more winning, so they will have
        # a higher value
        divider = number_of_moves/10


        for i in range(len(values)):
            if values[i] == -1:
                values[i] = - math.log10((i + 1) / divider + 1)
            else:
                values[i] =  math.log10((i + 1) / divider + 1)
            # values are also normalised to fit between 0 and 1

        return values

    def get_matrix_representation(self, move_number):
        # if it is player 1 to move
        if move_number % 2 == 0:
            return board.board
        else:
            # thanks to the simple board representation, this is very elegant
            return board.board * -1

    # replay the game again and create an array with the positions
    def add_game_to_dataset(self, game, result, number_of_moves):
        values = self.get_values(number_of_moves, result)
        positions = []

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
                board.add_position(move[0], move[1], side_to_move)
                move_number += 1

                # get an array representation of the position
                matrix = self.get_matrix_representation(move_number)
                positions.append(matrix)

                if board.is_there_five_in_row(side_to_move, move):
                    break

                side_to_move *= -1
            else:
                break

        board.new_game()

        values = np.asarray(values)
        positions = np.asarray(positions)

        return positions, values




d = Dataset()

directory_in_str = r"C:\Users\Rigos\Documents\piskvorky games\Standard"
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


positions_mega_list = np.asarray(positions_mega_list)
values_mega_list = np.asarray(values_mega_list)
print(positions_mega_list.shape, values_mega_list.shape)
print(positions_mega_list)


import os
import sys
import random
import csv


def generate_random_positions(num_positions: int, max_moves: int = 42):
    positions = set()
    metadata = dict()  # flat_board -> (move_count, turn)

    while len(positions) < num_positions:
        game = ConnectFour()
        move_count = 0

        while not game.is_over() and move_count < max_moves:
            legal = game.legal_moves()
            if not legal:
                break
            move = random.choice(legal)
            game.play(move)
            move_count += 1

            if game.is_over():
                continue  # skip final positions

            flat_board = tuple(game.board.flatten())
            if flat_board not in positions:
                positions.add(flat_board)
                metadata[flat_board] = (move_count, game.turn)

            if len(positions) >= num_positions:
                break

    return [(board, metadata[board][0], metadata[board][1]) for board in positions]

def save_positions_as_csv(positions, filename="positions.csv"):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        for flat_board, move_count, turn in positions:
            writer.writerow(list(flat_board) + [move_count, turn])

if __name__ == "__main__":

    from Game.ConnectFour import ConnectFour

    p = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.append(p)

    n = int(input("How many positions do you want to generate? "))
    while n <= 0:
        print("Invalid number Try again:")
        n = int(input())
    print(f"Generating {n} random ConnectFour positions...")

    positions = generate_random_positions(n)

    file_path = os.path.join(p, 'results', 'positions.csv')
    save_positions_as_csv(positions, file_path)

    print(f"Positions saved in {file_path}")

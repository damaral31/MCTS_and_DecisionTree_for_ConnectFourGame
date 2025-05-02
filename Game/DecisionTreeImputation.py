from Game.ConnectFour import ConnectFour
import utils.config as config
import numpy as np
import os
    
import pygame

class BoardEditor:

    def __init__(self):
        pygame.init()
        self.game = ConnectFour()
        self.width = config.WIDTH
        self.height = config.HEIGHT
        self.square_size = config.SQUARESIZE
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Connect Four Board Editor")
        self.font = pygame.font.SysFont("Arial", 30)
        self.dragging_piece = None  # Keeps track of the piece being dragged (1 for red, -1 for yellow)

    def draw_board(self):
        for c in range(config.COLUMN):
            for r in range(config.ROW):
                pygame.draw.rect(self.screen, config.BLACK, (c * config.SQUARESIZE, 0, config.SQUARESIZE, config.SQUARESIZE))
                if c == 0:
                    pygame.draw.circle(self.screen, config.RED, (int(c * config.SQUARESIZE + config.SQUARESIZE // 2), int(config.SQUARESIZE // 2)), config.RADIUS)

                elif c == config.COLUMN - 1:
                    pygame.draw.circle(self.screen, config.YELLOW, (int(c * config.SQUARESIZE + config.SQUARESIZE // 2), int(config.SQUARESIZE // 2)), config.RADIUS)
                
                
                text_surface = self.font.render("Press right-click on a piece to erase it", True, config.WHITE)
                text_rect = text_surface.get_rect(center=(self.width // 2, 30))
                text_enter = self.font.render("Press Enter to run the model", True, config.WHITE)
                text_rect_enter = text_enter.get_rect(center=(self.width // 2, 70))

                self.screen.blit(text_surface, text_rect)
                self.screen.blit(text_enter, text_rect_enter)
                pygame.draw.rect(self.screen, config.BLUE, (c * config.SQUARESIZE, (r + 1) * config.SQUARESIZE, config.SQUARESIZE, config.SQUARESIZE))
                piece = self.game.board[r][c]
                color = config.BLACK
                if piece == 1:
                    color = config.RED
                elif piece == -1:
                    color = config.YELLOW
                pygame.draw.circle(self.screen, color, (int(c * config.SQUARESIZE + config.SQUARESIZE // 2), int((r + 1) * config.SQUARESIZE + config.SQUARESIZE // 2)), config.RADIUS)
        pygame.display.update()

    def handle_drag_and_drop(self, debug=False):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                if y < config.SQUARESIZE:  # Top area for selecting pieces
                    if x < config.SQUARESIZE:
                        self.dragging_piece = 1  # Red piece
                    elif x > (config.COLUMN - 1) * config.SQUARESIZE:
                        self.dragging_piece = -1  # Yellow piece
                else:  # Check for erasing a piece
                    col = x // self.square_size
                    row = (y // self.square_size) - 1
                    if event.button == 3:  # Right-click to erase
                        if 0 <= row < config.ROW and 0 <= col < config.COLUMN:
                            self.game.board[row][col] = 0
            if event.type == pygame.MOUSEBUTTONUP:
                if self.dragging_piece is not None:
                    x, y = event.pos
                    col = x // self.square_size
                    row = (y // self.square_size) - 1
                    if 0 <= row < config.ROW and 0 <= col < config.COLUMN and self.game.board[row][col] == 0:
                        self.game.board[row][col] = self.dragging_piece
                self.dragging_piece = None
            if event.type == pygame.MOUSEMOTION and self.dragging_piece is not None:
                self.draw_board()
                x, y = event.pos
                color = config.RED if self.dragging_piece == 1 else config.YELLOW
                pygame.draw.circle(self.screen, color, (x, y), config.RADIUS)
                pygame.display.update()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:  # Check if Enter key is pressed
                    valid, message = self.is_valid_board()
                    if valid:
                        self.run_model(debug=debug)
                    else:
                        print(f"Invalid board: {message}")

    def is_valid_board(self):
        if not isinstance(self.game.board, np.ndarray):
            return False, 'Not a numpy array'
        if self.game.board.shape != (6, 7):
            return False, 'Invalid shape'
        if not np.all(np.isin(self.game.board, [-1, 0, 1])):
            return False, 'Invalid values in the array'
        
        if self.game.is_over() or self.game.check_win():
            return False, 'Game is over'

        for col in range(7):
            for row in range(5, 0, -1):  # de baixo (5) para cima (0)
                if self.game.board[row][col] == 0 and self.game.board[row-1][col] != 0:
                    return False, 'Floating piece'


        # Verificar número de peças de cada jogador
        p1_count = np.sum(self.game.board == 1)
        p2_count = np.sum(self.game.board == -1)
        if abs(p1_count - p2_count) > 1:
            return False, 'Invalid number of pieces'

        return True, 'Valid board'

    def make_player1_move(self):
        board = self.game.board.copy()
        p1_pices = np.sum(board == 1)
        p2_pices = np.sum(board == -1)
        if p1_pices > p2_pices:
            board = np.where(self.game.board == 1, -1, np.where(self.game.board == -1, 1, 0))
        return board

    def run_model(self, debug=False):
        # Here you can implement the logic to run your model on the current board state
        # For example, you can call a function that takes self.game.board as input and processes it
        
        board = self.make_player1_move()
        if debug:
            print(f"Model input:\n {board}\n")
        print("Please select a model to run:")
        print("1. ID3 Tree")
        print("2. Ruleset (Tree with pruning)")
        print("3. Bagging")

        while True:
            try:
                choice = int(input("Enter your choice (1 / 2 / 3 / 0 to exit): "))
                if choice in [1, 2, 3, 0]:
                    break
                else:
                    print("Invalid choice. Please select a valid model.")
            except ValueError:
                print("Invalid input. Please enter a number.")
        if choice == 0:
            return
        selected = ""
        if choice == 1:
            from DecisionTree.ID3Tree import ID3Tree
            file_path = os.path.join(os.getcwd(), "models", "id3_tree.pkl")
            model = ID3Tree.load_model(file_path)
            rules = model.build_rules()
            selected = "ID3 Tree"
        elif choice == 2:
            from DecisionTree.Ruleset import Ruleset
            file_path = os.path.join(os.getcwd(), "models", "ruleset.pkl")
            model = Ruleset.load_model(file_path)
            selected = "Ruleset"
        elif choice == 3:
            from DecisionTree.Bootstrap_Aggregating import Bagging
            file_path = os.path.join(os.getcwd(), "models", "bagging_model.pkl")
            model = Bagging.load_model(file_path)
            selected = "Bagging"

        row = board.flatten()
        number_of_pieces = np.count_nonzero(row)
        row = np.append(row, number_of_pieces)
        row = np.append(row, 0)
        row = row.tolist()
        model_pred = None

        if choice == 1:
            for rule in rules:
                model_pred = rule.predict(row)
                if model_pred is not None:
                    break
            if model_pred is None:
                model_pred = -1  # ERROR_CLASS
        else:
            model_pred, _ = model.predict(row)
        
        print(f"{selected} Prediction: Column {model_pred} (0-6)")
        if model_pred == -1:
            print("The model was not able to predict a column.")
        
    def run_editor(self, debug=False): 
        while True:
            self.draw_board()
            self.handle_drag_and_drop(debug=debug)

if __name__ == "__main__":
    editor = BoardEditor()
    editor.run_editor()
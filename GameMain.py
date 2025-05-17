from Game.ConnectFour import ConnectFour
from MCTS.MCTS_optimized import MonteCarlo
from MCTS.node import Node
from MCTS.MCTS import MonteCarlo_Single
import utils.config as config
import timeit
from utils.Visualize_MCtree import Drawer
from Game.DecisionTreeImputation import BoardEditor
import contextlib
import os
import random
import numpy as np
from DecisionTree.ID3Tree import ID3Tree
from DecisionTree.Ruleset import Ruleset
from DecisionTree.Bootstrap_Aggregating import Bagging

# Suppress pygame.init() stdout
with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
    import pygame
    pygame.init()

class ConnectFourGUI:
    """
    Main GUI class for the Connect Four game.
    Handles game logic, user interface, and event processing.
    """

    def __init__(self):
        # Initialize pygame and set up the game window and font
        self.game = ConnectFour()
        self.width = config.WIDTH
        self.height = config.HEIGHT
        self.square_size = config.SQUARESIZE
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Connect Four")
        self.font = pygame.font.SysFont("Arial", 40)

    def check_escape(self):
        """
        Checks for quit or escape events.
        If escape is pressed, resets the game and returns to the main menu.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.game.reset_game()
                self.screen.fill(config.BLACK)
                self.mainMenu()
                return True
        return False

    def draw_board(self):
        """
        Draws the current state of the game board on the screen.
        """
        for c in range(config.COLUMN):
            for r in range(config.ROW):
                # Draw the board background and pieces
                pygame.draw.rect(self.screen, config.BLACK, (c * config.SQUARESIZE, 0, config.SQUARESIZE, config.SQUARESIZE))
                pygame.draw.rect(self.screen, config.BLUE, (c * config.SQUARESIZE, (r + 1) * config.SQUARESIZE, config.SQUARESIZE, config.SQUARESIZE))
                piece = self.game.board[r][c]
                color = config.BLACK
                if piece == 1:
                    color = config.RED
                elif piece == -1:
                    color = config.YELLOW
                pygame.draw.circle(self.screen, color, (int(c * config.SQUARESIZE + config.SQUARESIZE // 2), int((r + 1) * config.SQUARESIZE + config.SQUARESIZE // 2)), config.RADIUS)
        # Draw the "go back" text
        go_back = self.font.render("Press ESC to go back to Main Menu", True, config.WHITE)
        self.screen.blit(go_back, (10, 10))
        pygame.display.update()

    def get_player_move(self):
        """
        Waits for the player to make a move by clicking a column.
        Returns the selected column index.
        """
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    self.game.reset_game()
                    self.screen.fill(config.BLACK)
                    self.mainMenu()
                    return None
                if event.type == pygame.MOUSEMOTION:
                    # Show a preview of the piece above the board
                    self.draw_board()
                    x = event.pos[0]
                    color = config.RED if self.game.turn == 1 else config.YELLOW
                    pygame.draw.circle(self.screen, color, (x, config.SQUARESIZE // 2), config.RADIUS)
                    pygame.display.update()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    # Player clicked, determine the column
                    x = event.pos[0]
                    col = x // self.square_size
                    if col in self.game.legal_moves():
                        return col

    def end_game_message(self):
        """
        Displays the end game message and resets the game.
        """
        if self.game.win == 0:
            text = self.font.render("Draw", True, config.WHITE)
        elif self.game.win == 1:
            text = self.font.render("Red Player wins!", True, config.GREEN)
        else:
            text = self.font.render("Yellow Player wins!", True, config.GREEN)
        self.screen.blit(text, (self.width // 2 - text.get_width() // 2, self.height // 2 - text.get_height() // 2))
        pygame.display.update()
        pygame.time.wait(3000)
        self.game.reset_game()
        self.screen.fill(config.BLACK)
        self.mainMenu()

    def run_pvp(self):
        """
        Runs a Player vs Player game loop.
        """
        while not self.game.is_over():
            if self.check_escape():
                return
            player_move = self.get_player_move()
            if player_move is None:
                return
            self.game.play(player_move)
            self.draw_board()
        self.end_game_message()

    def run_pva(self, iterations, debug):
        """
        Runs a Player vs AI game loop.
        The AI uses Monte Carlo Tree Search for its moves.
        """
        while not self.game.is_over():
            if self.check_escape():
                return
            self.draw_board()
            if self.game.turn == 1:
                # Player's turn
                player_move = self.get_player_move()
                if player_move is None:
                    return
                self.game.play(player_move)
            else:
                # AI's turn
                root = Node(self.game)
                monte_carlo = MonteCarlo(iteration=iterations, debug=debug) if iterations >= config.MEDIUMLEVEL else MonteCarlo_Single(iteration=iterations, debug=debug)
                start_time = timeit.default_timer()
                best_child, scores = monte_carlo.search(root)
                end_time = timeit.default_timer()
                self.game.play(best_child)
                if debug:
                    print(scores)
                    print(f"AI took {end_time - start_time:.2f} seconds to decide.")
                    drawer = Drawer()
                    G = drawer.build_tree_graph(root, depth=2, max_nodes=100)
                    drawer.draw_tree(G)
        self.draw_board()
        self.end_game_message()

    def run_pvd(self, model, rules=None):
        """
        Runs a Player vs Decision Tree game loop.
        The Decision Tree uses a pre-trained model to make its moves.
        """
        while not self.game.is_over():
            if self.check_escape():
                return
            self.draw_board()
            if self.game.turn == 1:
                # Player's turn
                player_move = self.get_player_move()
                if player_move is None:
                    return
                self.game.play(player_move)
            else:
                row = self.game.board.flatten()
                number_of_pieces = np.count_nonzero(row)

                player_1 = np.copy(row)
                # Replace -1 with 0
                player_1[player_1 == -1] = 0
                player_2 = np.copy(row)
                player_2[player_2 == 1] = 0
                player_2[player_2 == -1] = 1

                row = np.append(player_1, player_2)

                row = np.append(row, number_of_pieces)
                row = np.append(row, 0)
                row = row.tolist()
                model_pred = None

                # Run the selected model and print the prediction
                if rules is not None:
                    for rule in rules:
                        model_pred = rule.predict(row)
                        if model_pred is not None:
                            break
                    if model_pred is None:
                        model_pred = -1  # ERROR_CLASS
                else:
                    model_pred, _ = model.predict(row)
                
                legal_moves = self.game.legal_moves()
                if model_pred not in legal_moves:
                    model_pred = random.choice(self.game.legal_moves())
                self.game.play(model_pred)
        self.draw_board()
        self.end_game_message()

    def ai_or_dt_menu(self, debug=False):
        """
        Displays a menu to choose between playing against Monte Carlo Tree Search (AI) or a Decision Tree.
        """
        self.screen.fill(config.BLACK)
        title = self.font.render("Choose Opponent", True, config.WHITE)
        ai_button = self.font.render("Play vs Monte Carlo AI", True, config.WHITE)
        dt_button = self.font.render("Play vs Decision Tree", True, config.WHITE)
        back_button = self.font.render("Back", True, config.WHITE)

        ai_rect = pygame.Rect(self.width // 2 - 200, self.height // 2 - 40, 400, 60)
        dt_rect = pygame.Rect(self.width // 2 - 200, self.height // 2 + 40, 400, 60)
        back_rect = pygame.Rect(self.width // 2 - 60, self.height // 2 + 140, 120, 50)

        while True:
            self.screen.fill(config.BLACK)
            self.screen.blit(title, (self.width // 2 - title.get_width() // 2, self.height // 2 - 120))
            pygame.draw.rect(self.screen, config.WHITE, ai_rect, 2)
            pygame.draw.rect(self.screen, config.WHITE, dt_rect, 2)
            pygame.draw.rect(self.screen, config.WHITE, back_rect, 2)
            self.screen.blit(ai_button, (ai_rect.x + 20, ai_rect.y + 10))
            self.screen.blit(dt_button, (dt_rect.x + 20, dt_rect.y + 10))
            self.screen.blit(back_button, (back_rect.x + 10, back_rect.y + 5))
            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return
                if event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = event.pos
                    if ai_rect.collidepoint(x, y):
                        self.pva_menu(debug)
                        return
                    elif dt_rect.collidepoint(x, y):
                        self.pvd_menu(debug)
                        return
                    elif back_rect.collidepoint(x, y):
                        return

    def pvd_menu(self, debug=False):
        """
        Displays a menu to choose which Decision Tree model to play against.
        """

        self.screen.fill(config.BLACK)
        title = self.font.render("Choose Decision Tree Model", True, config.WHITE)
        id3_button = self.font.render("Play vs ID3", True, config.WHITE)
        ruleset_button = self.font.render("Play vs Ruleset", True, config.WHITE)
        bagging_button = self.font.render("Play vs Bagging", True, config.WHITE)
        back_button = self.font.render("Back", True, config.WHITE)

        id3_rect = pygame.Rect(self.width // 2 - 200, self.height // 2 - 60, 400, 60)
        ruleset_rect = pygame.Rect(self.width // 2 - 200, self.height // 2 + 20, 400, 60)
        bagging_rect = pygame.Rect(self.width // 2 - 200, self.height // 2 + 100, 400, 60)
        back_rect = pygame.Rect(self.width // 2 - 60, self.height // 2 + 200, 120, 50)

        while True:
            self.screen.fill(config.BLACK)
            self.screen.blit(title, (self.width // 2 - title.get_width() // 2, self.height // 2 - 140))
            pygame.draw.rect(self.screen, config.WHITE, id3_rect, 2)
            pygame.draw.rect(self.screen, config.WHITE, ruleset_rect, 2)
            pygame.draw.rect(self.screen, config.WHITE, bagging_rect, 2)
            pygame.draw.rect(self.screen, config.WHITE, back_rect, 2)
            self.screen.blit(id3_button, (id3_rect.x + 20, id3_rect.y + 10))
            self.screen.blit(ruleset_button, (ruleset_rect.x + 20, ruleset_rect.y + 10))
            self.screen.blit(bagging_button, (bagging_rect.x + 20, bagging_rect.y + 10))
            self.screen.blit(back_button, (back_rect.x + 10, back_rect.y + 5))
            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return
                if event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = event.pos
                    if id3_rect.collidepoint(x, y):
                        model = ID3Tree.load_model(os.path.join("models", "id3_analize.pkl"))
                        rules = model.build_rules()
                        self.run_pvd(model, rules=rules)
                        return
                    elif ruleset_rect.collidepoint(x, y):
                        model = Ruleset.load_model(os.path.join("models", "ruleset_analize.pkl"))
                        self.run_pvd(model)
                        return
                    elif bagging_rect.collidepoint(x, y):
                        model = Bagging.load_model(os.path.join("models", "bagging_analize.pkl"))
                        self.run_pvd(model)
                        return
                    elif back_rect.collidepoint(x, y):
                        return
        
    def run_ava(self, ai1_iter=config.ITERATION, ai2_iter=config.ITERATION, debug=False, save_path=None):
        """
        Runs an AI vs AI game loop.
        Both sides use Monte Carlo Tree Search.
        """
        while not self.game.is_over():
            if self.check_escape():
                return
            self.draw_board()
            root = Node(self.game)
            
            if(self.game.turn == 1):
                monte_carlo = MonteCarlo(iteration=ai1_iter, debug=debug) if ai1_iter >= config.MEDIUMLEVEL else MonteCarlo_Single(iteration=ai1_iter, debug=debug)
            else:
                monte_carlo = MonteCarlo(iteration=ai2_iter, debug=debug) if ai2_iter >= config.MEDIUMLEVEL else MonteCarlo_Single(iteration=ai2_iter, debug=debug)            
            
            start_time = timeit.default_timer()
            best_child, scores = monte_carlo.search(root)
            end_time = timeit.default_timer()

            if save_path is not None:
                linha = [self.game.board[i][j] for i in range(config.ROW) for j in range(config.COLUMN)] + [self.game.pieces] + [self.game.turn] + [best_child]
                linha = [str(x) for x in linha]  # Convert all elements to string to use join method
                linha = ';'.join(linha)
                with open(save_path, 'a') as f:
                    f.write(linha + '\n')

            if debug:
                print(scores)
                print(f"AI {self.game.turn} took {end_time - start_time:.2f} seconds to decide.")
                drawer = Drawer()
                G = drawer.build_tree_graph(root, depth=2, max_nodes=100)
                drawer.draw_tree(G)

            self.game.play(best_child)
        if save_path is not None:
            return
        
        self.draw_board()
        self.end_game_message()

    def pva_menu(self, debug=False):
        """
        Displays the difficulty selection menu for Player vs AI mode.
        """
        while not self.game.is_over():
            self.screen.fill(config.BLACK)
            title = self.font.render("Please select the difficulty level", True, config.WHITE)
            noob_button = self.font.render("Easy Mode", True, config.WHITE)
            pro_button = self.font.render("Medium Mode", True, config.WHITE)
            hacker_button = self.font.render("Hard Mode", True, config.WHITE)
            self.screen.blit(title, (self.width // 2 - title.get_width() // 2, self.height // 2 - 150))
            self.screen.blit(noob_button, (self.width // 2 - noob_button.get_width() // 2, self.height // 2 - 30))
            self.screen.blit(pro_button, (self.width // 2 - pro_button.get_width() // 2, self.height // 2 + 30))
            self.screen.blit(hacker_button, (self.width // 2 - hacker_button.get_width() // 2, self.height // 2 + 90))
            pygame.display.update()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return
                if event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = event.pos
                    # Check which difficulty button was clicked
                    if self.width // 2 - noob_button.get_width() // 2 < x < self.width // 2 + noob_button.get_width() // 2 and self.height // 2 - 30 < y < self.height // 2:
                        self.run_pva(config.EASYLEVEL, debug)
                    elif self.width // 2 - pro_button.get_width() // 2 < x < self.width // 2 + pro_button.get_width() // 2 and self.height // 2 + 30 < y < self.height // 2 + 60:
                        self.run_pva(config.MEDIUMLEVEL, debug)
                    elif self.width // 2 - hacker_button.get_width() // 2 < x < self.width // 2 + hacker_button.get_width() // 2 and self.height // 2 + 90 < y < self.height // 2 + 120:
                        self.run_pva(config.HARDLEVEL, debug)

    def models_menu(self, debug=False):
        """
        Displays a menu to choose AI vs AI, AI vs Decision Tree, or Decision Tree vs Decision Tree.
        """
        self.screen.fill(config.BLACK)
        title = self.font.render("Choose AI Matchup", True, config.WHITE)
        ava_button = self.font.render("Monte Carlo vs Monte Carlo", True, config.WHITE)
        avdt_button = self.font.render("Monte Carlo vs Decision Tree", True, config.WHITE)
        dtdt_button = self.font.render("Decision Tree vs Decision Tree", True, config.WHITE)
        back_button = self.font.render("Back", True, config.WHITE)

        ava_rect = pygame.Rect(self.width // 2 - 220, self.height // 2 - 60, 440, 60)
        avdt_rect = pygame.Rect(self.width // 2 - 220, self.height // 2 + 20, 440, 60)
        dtdt_rect = pygame.Rect(self.width // 2 - 220, self.height // 2 + 100, 440, 60)
        back_rect = pygame.Rect(self.width // 2 - 60, self.height // 2 + 200, 120, 50)

        while True:
            self.screen.fill(config.BLACK)
            self.screen.blit(title, (self.width // 2 - title.get_width() // 2, self.height // 2 - 140))
            pygame.draw.rect(self.screen, config.WHITE, ava_rect, 2)
            pygame.draw.rect(self.screen, config.WHITE, avdt_rect, 2)
            pygame.draw.rect(self.screen, config.WHITE, dtdt_rect, 2)
            pygame.draw.rect(self.screen, config.WHITE, back_rect, 2)
            self.screen.blit(ava_button, (ava_rect.x + 20, ava_rect.y + 10))
            self.screen.blit(avdt_button, (avdt_rect.x + 20, avdt_rect.y + 10))
            self.screen.blit(dtdt_button, (dtdt_rect.x + 20, dtdt_rect.y + 10))
            self.screen.blit(back_button, (back_rect.x + 10, back_rect.y + 5))
            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return
                if event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = event.pos
                    if ava_rect.collidepoint(x, y):
                        self.ava_menu(debug)
                        return
                    elif avdt_rect.collidepoint(x, y):
                        # Monte Carlo vs Decision Tree
                        # Ask user to select DT model for right side
                        self.avdt_menu(debug)
                        return
                    elif dtdt_rect.collidepoint(x, y):
                        # Decision Tree vs Decision Tree
                        self.dtdt_menu(debug)
                        return
                    elif back_rect.collidepoint(x, y):
                        return

    def avdt_menu(self, debug=False):
        """
        Menu for selecting Monte Carlo iterations and Decision Tree model for Monte Carlo vs Decision Tree.
        """
        self.screen.fill(config.BLACK)
        title = self.font.render("Monte Carlo vs Decision Tree", True, config.WHITE)

        # Monte Carlo iterations input
        mc_label = self.font.render("Monte Carlo Iterations:", True, config.WHITE)
        mc_input_rect = pygame.Rect(self.width // 2 + 100, self.height // 2 - 40, 200, 50)
        mc_input_active = False
        mc_input_text = str(config.ITERATION)

        # Decision Tree model selection
        dt_label = self.font.render("Decision Tree Model:", True, config.WHITE)
        dt_types = ["ID3", "Ruleset", "Bagging"]
        dt_idx = 0
        dt_rect = pygame.Rect(self.width // 2 + 100, self.height // 2 + 40, 200, 50)

        # Buttons
        start_button = self.font.render("Start", True, config.WHITE)
        start_rect = pygame.Rect(self.width // 2 - 60, self.height // 2 + 120, 120, 50)
        back_button = self.font.render("Back", True, config.WHITE)
        back_rect = pygame.Rect(self.width // 2 - 60, self.height // 2 + 190, 120, 50)

        error_message = ""

        while True:
            self.screen.fill(config.BLACK)
            self.screen.blit(title, (self.width // 2 - title.get_width() // 2, self.height // 2 - 120))

            # Monte Carlo input
            self.screen.blit(mc_label, (mc_input_rect.x - 370, mc_input_rect.y))
            pygame.draw.rect(self.screen, config.WHITE, mc_input_rect, 2)
            mc_input_surface = self.font.render(mc_input_text, True, config.WHITE)
            self.screen.blit(mc_input_surface, (mc_input_rect.x + 10, mc_input_rect.y + 10))
            if mc_input_active:
                pygame.draw.rect(self.screen, config.RED, mc_input_rect, 2)

            # Decision Tree selection
            self.screen.blit(dt_label, (dt_rect.x - 370, dt_rect.y))
            pygame.draw.rect(self.screen, config.WHITE, dt_rect, 2)
            dt_surface = self.font.render(dt_types[dt_idx], True, config.WHITE)
            self.screen.blit(dt_surface, (dt_rect.x + 10, dt_rect.y + 10))

            # Buttons
            pygame.draw.rect(self.screen, config.WHITE, start_rect, 2)
            self.screen.blit(start_button, (start_rect.x + 10, start_rect.y + 5))
            pygame.draw.rect(self.screen, config.WHITE, back_rect, 2)
            self.screen.blit(back_button, (back_rect.x + 10, back_rect.y + 5))

            if error_message:
                error_surface = self.font.render(error_message, True, config.RED)
                self.screen.blit(error_surface, (self.width // 2 - error_surface.get_width() // 2, self.height // 2 + 170))

            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return
                if event.type == pygame.KEYDOWN:
                    if mc_input_active:
                        if event.key == pygame.K_RETURN:
                            mc_input_active = False
                        elif event.key == pygame.K_BACKSPACE:
                            mc_input_text = mc_input_text[:-1]
                        elif event.unicode.isdigit():
                            mc_input_text += event.unicode
                if event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = event.pos
                    if mc_input_rect.collidepoint(x, y):
                        mc_input_active = True
                    else:
                        mc_input_active = False
                    if dt_rect.collidepoint(x, y):
                        dt_idx = (dt_idx + 1) % len(dt_types)
                    elif start_rect.collidepoint(x, y):
                        try:
                            mc_iter = int(mc_input_text)
                            if mc_iter < 1:
                                error_message = "Iterations must be positive."
                                continue
                        except ValueError:
                            error_message = "Enter a valid number."
                            continue
                        # Load selected DT model
                        if dt_types[dt_idx] == "ID3":
                            model = ID3Tree.load_model(os.path.join("models", "id3_analize.pkl"))
                            rules = model.build_rules()
                            self.run_avdt(model, rules=rules, debug=debug, ai1_iter=mc_iter)
                        elif dt_types[dt_idx] == "Ruleset":
                            model = Ruleset.load_model(os.path.join("models", "ruleset_analize.pkl"))
                            self.run_avdt(model, debug=debug, ai1_iter=mc_iter)
                        elif dt_types[dt_idx] == "Bagging":
                            model = Bagging.load_model(os.path.join("models", "bagging_analize.pkl"))
                            self.run_avdt(model, debug=debug, ai1_iter=mc_iter)
                        return
                    elif back_rect.collidepoint(x, y):
                        return

    def dtdt_menu(self, debug=False):
        """
        Menu for selecting Decision Tree models for both sides in Decision Tree vs Decision Tree.
        """
        self.screen.fill(config.BLACK)
        title = self.font.render("Select Decision Tree Models", True, config.WHITE)
        left_label = self.font.render("Red DT:", True, config.WHITE)
        right_label = self.font.render("Yellow DT:", True, config.WHITE)
        dt_types = ["ID3", "Ruleset", "Bagging"]
        left_idx = 0
        right_idx = 0
        start_button = self.font.render("Start", True, config.WHITE)
        back_button = self.font.render("Back", True, config.WHITE)
        left_rect = pygame.Rect(self.width // 2 - 200, self.height // 2 - 40, 180, 60)
        right_rect = pygame.Rect(self.width // 2 + 20, self.height // 2 - 40, 180, 60)
        start_rect = pygame.Rect(self.width // 2 - 60, self.height // 2 + 60, 120, 50)
        back_rect = pygame.Rect(self.width // 2 - 60, self.height // 2 + 130, 120, 50)

        while True:
            self.screen.fill(config.BLACK)
            self.screen.blit(title, (self.width // 2 - title.get_width() // 2, self.height // 2 - 160))
            self.screen.blit(left_label, (left_rect.x, left_rect.y - 40))
            self.screen.blit(right_label, (right_rect.x, right_rect.y - 40))
            pygame.draw.rect(self.screen, config.WHITE, left_rect, 2)
            pygame.draw.rect(self.screen, config.WHITE, right_rect, 2)
            left_dt = self.font.render(dt_types[left_idx], True, config.WHITE)
            right_dt = self.font.render(dt_types[right_idx], True, config.WHITE)
            self.screen.blit(left_dt, (left_rect.x + 20, left_rect.y + 10))
            self.screen.blit(right_dt, (right_rect.x + 20, right_rect.y + 10))
            pygame.draw.rect(self.screen, config.WHITE, start_rect, 2)
            pygame.draw.rect(self.screen, config.WHITE, back_rect, 2)
            self.screen.blit(start_button, (start_rect.x + 10, start_rect.y + 5))
            self.screen.blit(back_button, (back_rect.x + 10, back_rect.y + 5))
            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return
                if event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = event.pos
                    if left_rect.collidepoint(x, y):
                        left_idx = (left_idx + 1) % len(dt_types)
                    elif right_rect.collidepoint(x, y):
                        right_idx = (right_idx + 1) % len(dt_types)
                    elif start_rect.collidepoint(x, y):
                        left_model, left_rules = self._load_dt_model(dt_types[left_idx])
                        right_model, right_rules = self._load_dt_model(dt_types[right_idx])
                        self.run_dtdt(left_model, right_model, left_rules, right_rules, debug=debug)
                        return
                    elif back_rect.collidepoint(x, y):
                        return

    def _load_dt_model(self, dt_type):
        if dt_type == "ID3":
            model = ID3Tree.load_model(os.path.join("models", "id3_analize.pkl"))
            rules = model.build_rules()
            return model, rules
        elif dt_type == "Ruleset":
            model = Ruleset.load_model(os.path.join("models", "ruleset_analize.pkl"))
            return model, None
        elif dt_type == "Bagging":
            model = Bagging.load_model(os.path.join("models", "bagging_analize.pkl"))
            return model, None
        return None, None

    def run_avdt(self, dt_model, rules=None, debug=False, ai1_iter=config.ITERATION):
        """
        Runs Monte Carlo vs Decision Tree game loop.
        Monte Carlo is always player 1, Decision Tree is player -1.
        """
        while not self.game.is_over():
            if self.check_escape():
                return
            self.draw_board()
            if self.game.turn == 1:
                # Monte Carlo's turn
                root = Node(self.game)
                monte_carlo = MonteCarlo(iteration=ai1_iter, debug=debug) if ai1_iter >= config.MEDIUMLEVEL else MonteCarlo_Single(iteration=ai1_iter, debug=debug)
                start_time = timeit.default_timer()
                best_child, scores = monte_carlo.search(root)
                end_time = timeit.default_timer()
                if debug:
                    print(scores)
                    print(f"AI {self.game.turn} took {end_time - start_time:.2f} seconds to decide.")
                    drawer = Drawer()
                    G = drawer.build_tree_graph(root, depth=2, max_nodes=100)
                    drawer.draw_tree(G)

                self.game.play(best_child)
            else:
                # Decision Tree's turn
                row = self.game.board.flatten()
                number_of_pieces = np.count_nonzero(row)
                player_1 = np.copy(row)
                player_1[player_1 == -1] = 0
                player_2 = np.copy(row)
                player_2[player_2 == 1] = 0
                player_2[player_2 == -1] = 1
                row = np.append(player_1, player_2)
                row = np.append(row, number_of_pieces)
                row = np.append(row, 0)
                row = row.tolist()
                model_pred = None
                if rules is not None:
                    for rule in rules:
                        model_pred = rule.predict(row)
                        if model_pred is not None:
                            break
                    if model_pred is None:
                        model_pred = -1
                else:
                    model_pred, _ = dt_model.predict(row)
                legal_moves = self.game.legal_moves()
                if model_pred not in legal_moves:
                    model_pred = random.choice(legal_moves)
                self.game.play(model_pred)
        self.draw_board()
        self.end_game_message()

    def run_dtdt(self, red_model, yellow_model, red_rules=None, yellow_rules=None, debug=False):
        """
        Runs Decision Tree vs Decision Tree game loop.
        left_model is player 1, right_model is player -1.
        """
        while not self.game.is_over():
            if self.check_escape():
                return
            self.draw_board()
            row = self.game.board.flatten()
            number_of_pieces = np.count_nonzero(row)
            player_1 = np.copy(row)
            player_1[player_1 == -1] = 0
            player_2 = np.copy(row)
            player_2[player_2 == 1] = 0
            player_2[player_2 == -1] = 1
            row = np.append(player_1, player_2)
            row = np.append(row, number_of_pieces)
            row = np.append(row, 0)
            row = row.tolist()
            model_pred = None
            if self.game.turn == 1:
                if red_rules is not None:
                    for rule in red_rules:
                        model_pred = rule.predict(row)
                        if model_pred is not None:
                            break
                    if model_pred is None:
                        model_pred = -1
                else:
                    model_pred, _ = red_model.predict(row)
            else:
                if yellow_rules is not None:
                    for rule in yellow_rules:
                        model_pred = rule.predict(row)
                        if model_pred is not None:
                            break
                    if model_pred is None:
                        model_pred = -1
                else:
                    model_pred, _ = yellow_model.predict(row)
            legal_moves = self.game.legal_moves()
            if model_pred not in legal_moves:
                model_pred = random.choice(legal_moves)
            self.game.play(model_pred)
            pygame.time.wait(500)
        self.draw_board()
        self.end_game_message()

    def ava_menu(self, debug=False):
        """
        Displays a menu to set the number of iterations for each AI in AI vs AI mode.
        """
        input_active1 = False
        input_active2 = False
        input_text1 = ""
        input_text2 = ""
        error_message = ""
        ai1_rect = pygame.Rect(self.width // 2 - 100, self.height // 2 - 40, 200, 50)
        ai2_rect = pygame.Rect(self.width // 2 - 100, self.height // 2 + 40, 200, 50)
        start_button = self.font.render("Start", True, config.WHITE)
        start_rect = pygame.Rect(self.width // 2 - 60, self.height // 2 + 120, 120, 50)

        while True:
            self.screen.fill(config.BLACK)
            title = self.font.render("Set Iterations for Each AI", True, config.WHITE)
            ai1_label = self.font.render("AI 1 Iterations:", True, config.WHITE)
            ai2_label = self.font.render("AI 2 Iterations:", True, config.WHITE)
            pygame.draw.rect(self.screen, config.WHITE, ai1_rect, 2)
            pygame.draw.rect(self.screen, config.WHITE, ai2_rect, 2)

            self.screen.blit(title, (self.width // 2 - title.get_width() // 2, self.height // 2 - 120))
            self.screen.blit(ai1_label, (ai1_rect.x - 220, ai1_rect.y + 10))
            self.screen.blit(ai2_label, (ai2_rect.x - 220, ai2_rect.y + 10))

            ai1_surface = self.font.render(input_text1, True, config.WHITE)
            ai2_surface = self.font.render(input_text2, True, config.WHITE)
            self.screen.blit(ai1_surface, (ai1_rect.x + 10, ai1_rect.y + 10))
            self.screen.blit(ai2_surface, (ai2_rect.x + 10, ai2_rect.y + 10))
            pygame.draw.rect(self.screen, config.WHITE, start_rect, 2)
            self.screen.blit(start_button, (start_rect.x + 10, start_rect.y + 5))
            if input_active1:
                pygame.draw.rect(self.screen, config.RED, ai1_rect, 2)
            if input_active2:
                pygame.draw.rect(self.screen, config.YELLOW, ai2_rect, 2)


            if error_message:
                error_surface = self.font.render(error_message, True, config.RED)
                self.screen.blit(error_surface, (self.width // 2 - error_surface.get_width() // 2, self.height // 2 + 180))
            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return
                if event.type == pygame.KEYDOWN:
                    if input_active1:
                        if event.key == pygame.K_RETURN:
                            input_active1 = False
                        elif event.key == pygame.K_BACKSPACE:
                            input_text1 = input_text1[:-1]
                        elif event.unicode.isdigit():
                            input_text1 += event.unicode
                    elif input_active2:
                        if event.key == pygame.K_RETURN:
                            input_active2 = False
                        elif event.key == pygame.K_BACKSPACE:
                            input_text2 = input_text2[:-1]
                        elif event.unicode.isdigit():
                            input_text2 += event.unicode
                    elif event.key == pygame.K_ESCAPE:
                        return
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if ai1_rect.collidepoint(event.pos):
                        input_active1 = True
                        input_active2 = False
                    elif ai2_rect.collidepoint(event.pos):
                        input_active1 = False
                        input_active2 = True

                    elif start_rect.collidepoint(event.pos):
                        try:
                            iter1 = int(input_text1)
                            iter2 = int(input_text2)
                            if iter1 < 0 or iter2 < 0:
                                error_message = "Iterations must be positive numbers."
                                continue
                            if debug:
                                print(f"AI 1 Iterations: {iter1}, AI 2 Iterations: {iter2}")
                            self.run_ava(iter1, iter2, debug)
                            return
                        except ValueError:
                            error_message = "Please enter valid numbers."

    def mainMenu(self):
        """
        Displays the main menu and handles navigation to different game modes and options.
        """
        debug = False
        while not self.game.is_over():
            self.screen.fill(config.BLACK)
            # Render menu buttons and texts
            title = self.font.render("Connect Four", True, config.WHITE)
            pvp_button = self.font.render("Player vs Player", True, config.WHITE)
            pva_button = self.font.render("Player vs AI", True, config.WHITE)
            ava_button = self.font.render("AI vs AI", True, config.WHITE)
            quit_button = self.font.render("Quit", True, config.WHITE)
            debug_text = self.font.render("Enable Debugging", True, config.WHITE)
            editor_text = self.font.render("Board Editor", True, config.WHITE)
            checkbox_rect = pygame.Rect(self.width // 2 - 135, self.height // 2 + 265, 20, 20)
            # Place buttons on the screen
            self.screen.blit(title, (self.width // 2 - title.get_width() // 2, self.height // 2 - 250))
            self.screen.blit(pvp_button, (self.width // 2 - pvp_button.get_width() // 2, self.height // 2 - 110))
            self.screen.blit(pva_button, (self.width // 2 - pva_button.get_width() // 2, self.height // 2 - 50))
            self.screen.blit(ava_button, (self.width // 2 - ava_button.get_width() // 2, self.height // 2 + 10))
            self.screen.blit(editor_text, (self.width // 2 - editor_text.get_width() // 2, self.height // 2 + 70))
            self.screen.blit(quit_button, (self.width // 2 - quit_button.get_width() // 2, self.height // 2 + 130))
            self.screen.blit(debug_text, (self.width // 2 - debug_text.get_width() // 2 + 30, self.height // 2 + 250))
            # Draw debug checkbox
            pygame.draw.rect(self.screen, config.WHITE, checkbox_rect, 2)
            if debug:
                pygame.draw.rect(self.screen, config.WHITE, checkbox_rect.inflate(-4, -4))
            pygame.display.update()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = event.pos
                    # Handle menu button clicks
                    if checkbox_rect.collidepoint(x, y):
                        debug = not debug
                    elif self.width // 2 - pvp_button.get_width() // 2 < x < self.width // 2 + pvp_button.get_width() // 2 and self.height // 2 - 110 < y < self.height // 2 - 70:
                        self.run_pvp()
                    elif self.width // 2 - pva_button.get_width() // 2 < x < self.width // 2 + pva_button.get_width() // 2 and self.height // 2 - 50 < y < self.height // 2 - 10:
                        self.ai_or_dt_menu(debug)
                    elif self.width // 2 - ava_button.get_width() // 2 < x < self.width // 2 + ava_button.get_width() // 2 and self.height // 2 + 10 < y < self.height // 2 + 50:
                        self.models_menu(debug)
                    elif self.width // 2 - editor_text.get_width() // 2 < x < self.width // 2 + editor_text.get_width() // 2 and self.height // 2 + 70 < y < self.height // 2 + 110:
                        editor = BoardEditor()
                        editor.run_editor(debug)
                    elif self.width // 2 - quit_button.get_width() // 2 < x < self.width // 2 + quit_button.get_width() // 2 and self.height // 2 + 130 < y < self.height // 2 + 170:
                        pygame.quit()
                        exit()

if __name__ == "__main__":
    gui = ConnectFourGUI()
    gui.mainMenu()
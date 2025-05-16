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

            self.game.play(best_child)
            if debug:
                print(scores)
                print(f"AI {self.game.turn} took {end_time - start_time:.2f} seconds to decide.")
                drawer = Drawer()
                G = drawer.build_tree_graph(root, depth=2, max_nodes=100)
                drawer.draw_tree(G)

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
                        self.pva_menu(debug)
                    elif self.width // 2 - ava_button.get_width() // 2 < x < self.width // 2 + ava_button.get_width() // 2 and self.height // 2 + 10 < y < self.height // 2 + 50:
                        self.ava_menu(debug)
                    elif self.width // 2 - editor_text.get_width() // 2 < x < self.width // 2 + editor_text.get_width() // 2 and self.height // 2 + 70 < y < self.height // 2 + 110:
                        editor = BoardEditor()
                        editor.run_editor(debug)
                    elif self.width // 2 - quit_button.get_width() // 2 < x < self.width // 2 + quit_button.get_width() // 2 and self.height // 2 + 130 < y < self.height // 2 + 170:
                        pygame.quit()
                        exit()

if __name__ == "__main__":
    gui = ConnectFourGUI()
    gui.mainMenu()
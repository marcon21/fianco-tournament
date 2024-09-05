import numpy as np
import pygame
from pygame.locals import *
import string
from random import random
import time

CELL_SIZE = 100
RADIUS = CELL_SIZE / 2 - 5

GAME_RES = WIDTH, HEIGHT = CELL_SIZE * 9 + 300, CELL_SIZE * 9
FPS = 60
GAME_TITLE, SCREEN_COLOR = "Fianco", "white"

pygame.init()
window = pygame.display.set_mode(GAME_RES, HWACCEL | HWSURFACE | DOUBLEBUF)
clock = pygame.time.Clock()
dt = 0
pygame.display.set_caption(GAME_TITLE)

pygame.font.init()
font = pygame.font.SysFont("Arial", 24)

all_legal_moves_one = {}
all_legal_moves_two = {}


class Board:
    def __init__(self):
        self.board = self.new_board()
        self.legal_moves = {}
        self.current_player = 1
        self.past_moves = []

        self.calculate_legal_moves()

    def new_board(self):
        board = np.zeros((9, 9), dtype=int)

        # Black
        board[0, :] = 2
        board[1, 1] = 2
        board[1, 7] = 2
        board[2, 2] = 2
        board[2, 6] = 2
        board[3, 3] = 2
        board[3, 5] = 2

        # White
        board[8, :] = 1
        board[7, 1] = 1
        board[7, 7] = 1
        board[6, 2] = 1
        board[6, 6] = 1
        board[5, 3] = 1
        board[5, 5] = 1

        return board

    def __str__(self):
        return str(self.board)

    def set_board(self, board):
        self.board = board

    def calculate_legal_moves(self):
        # if self.current_player == 1:
        #     if self.board.tobytes() in all_legal_moves_one:
        #         self.legal_moves = all_legal_moves_one[self.board.tobytes()]
        #     else:
        #         self.legal_moves = self.get_all_possible_moves(self.current_player)
        #         all_legal_moves_one[self.board.tobytes()] = self.legal_moves

        #     if len(all_legal_moves_one) > 250000:
        #         all_legal_moves_one.pop(next(iter(all_legal_moves_one)))
        # else:
        #     if self.board.tobytes() in all_legal_moves_two:
        #         self.legal_moves = all_legal_moves_two[self.board.tobytes()]
        #     else:
        #         self.legal_moves = self.get_all_possible_moves(self.current_player)
        #         all_legal_moves_two[self.board.tobytes()] = self.legal_moves
        #     if len(all_legal_moves_two) > 250000:
        #         all_legal_moves_two.pop(next(iter(all_legal_moves_two)))
        self.legal_moves = self.get_all_possible_moves(self.current_player)

    def load_moves(self, file: string):
        with open(file, "r") as f:
            lines = f.readlines()
            for line in lines:
                self.move(line.strip())

    def save_moves(self, file: string):
        with open(file, "w") as f:
            for move in self.past_moves:
                f.write(f"{move}\n")

    def get_all_possible_moves(self, player):
        legal_moves = {}
        self.capturers = []

        for i in range(9):
            for j in range(9):
                if self.board[i, j] == player:
                    moves = self.get_possible_moves([i, j])
                    legal_moves[f"{i}{j}"] = moves

                    capture_move = []
                    for move in moves:
                        if abs(i - move[0]) > 1:
                            capture_move.append(move)

                    if capture_move:
                        self.capturers.append([i, j])
                        legal_moves[f"{i}{j}"] = capture_move

        if self.capturers:
            only_captures = {}
            for capturer in self.capturers:
                only_captures[f"{capturer[0]}{capturer[1]}"] = legal_moves[
                    f"{capturer[0]}{capturer[1]}"
                ]
            return only_captures

        return legal_moves

    # @profile
    def get_possible_moves(self, piece):
        if type(piece) == str:
            piece = self.convert_coord_to_abs(piece)

        player = self.board[piece[0], piece[1]]

        # Empty cell
        if player == 0:
            return []

        # Sides moves
        moves = [
            [piece[0], piece[1] + 1],
            [piece[0], piece[1] - 1],
        ]

        # Forward move
        if player == 1:
            moves.append([piece[0] - 1, piece[1]])
            # Capture moves
            if piece[0] > 1 and piece[1] < 7:
                if self.board[piece[0] - 1, piece[1] + 1] == 2:
                    moves.append([piece[0] - 2, piece[1] + 2])
            if piece[0] > 1 and piece[1] > 1:
                if self.board[piece[0] - 1, piece[1] - 1] == 2:
                    moves.append([piece[0] - 2, piece[1] - 2])
        elif player == 2:
            moves.append([piece[0] + 1, piece[1]])
            # Capture moves
            if piece[0] < 7 and piece[1] < 7:
                if self.board[piece[0] + 1, piece[1] + 1] == 1:
                    moves.append([piece[0] + 2, piece[1] + 2])
            if piece[0] < 7 and piece[1] > 1:
                if self.board[piece[0] + 1, piece[1] - 1] == 1:
                    moves.append([piece[0] + 2, piece[1] - 2])

        valid_moves = []
        for move in moves:
            # Check if move is within bounds
            if 0 <= move[0] < 9 and 0 <= move[1] < 9:
                # End position is empty
                if self.board[move[0], move[1]] == 0:
                    valid_moves.append(move)

        return valid_moves

    def move(self, move):
        # String in format A1-A2
        if type(move) == str:
            start = self.convert_coord_to_abs(move[:2])
            end = self.convert_coord_to_abs(move[3:])
        else:
            start = move[0]
            end = move[1]

        if not self.is_move_valid(start, end):
            print("Invalid move")
            return

        # Capture
        if abs(start[0] - end[0]) > 1:
            self.board[(start[0] + end[0]) // 2, (start[1] + end[1]) // 2] = 0

        self.board[end[0], end[1]] = self.board[start[0], start[1]]
        self.board[start[0], start[1]] = 0

        self.past_moves.append(move)
        self.current_player = self.current_player % 2 + 1

        self.calculate_legal_moves()

    def undo_move(self):
        if not self.past_moves:
            return

        move = self.past_moves.pop()
        if type(move) == str:
            start = self.convert_coord_to_abs(move[:2])
            end = self.convert_coord_to_abs(move[3:])
        else:
            start = move[0]
            end = move[1]

        if abs(start[0] - end[0]) > 1:
            self.board[(start[0] + end[0]) // 2, (start[1] + end[1]) // 2] = (
                self.current_player
            )

        self.board[start[0], start[1]] = self.board[end[0], end[1]]
        self.board[end[0], end[1]] = 0

        self.current_player = self.current_player % 2 + 1
        self.calculate_legal_moves()

    def convert_coord_to_abs(self, pos: string):
        # Pos in format A1 to (8, 0)
        pos = pos.upper()
        return [9 - int(pos[1]), string.ascii_uppercase.index(pos[0])]

    def convert_coord_to_str(self, pos):
        # Pos in format (8, 0) to A1
        return f"{string.ascii_uppercase[pos[1]]}{9 - pos[0]}"

    def is_game_over(self):
        if 1 in self.board[0, :]:
            return 1
        if 2 in self.board[8, :]:
            return -1
        return 0

    def is_move_valid(self, start, end):
        if self.board[start[0], start[1]] != self.current_player:
            return False

        if self.board[end[0], end[1]] != 0:
            return False

        if end not in self.legal_moves[f"{start[0]}{start[1]}"]:
            return False

        return True

    def draw_board(self, window, active_piece=None):
        for i in range(9):
            pygame.draw.line(
                window,
                (0, 0, 0),
                (i * CELL_SIZE + CELL_SIZE / 2, CELL_SIZE / 2),
                (i * CELL_SIZE + CELL_SIZE / 2, CELL_SIZE * 9 - CELL_SIZE / 2),
            )
            pygame.draw.line(
                window,
                (0, 0, 0),
                (CELL_SIZE / 2, i * CELL_SIZE + CELL_SIZE / 2),
                (CELL_SIZE * 9 - CELL_SIZE / 2, i * CELL_SIZE + CELL_SIZE / 2),
            )

        for capturer in self.capturers:
            # Draw capturer
            pygame.draw.circle(
                window,
                (0, 0, 255),
                (
                    capturer[1] * CELL_SIZE + CELL_SIZE / 2,
                    capturer[0] * CELL_SIZE + CELL_SIZE / 2,
                ),
                RADIUS + 3,
            )

        if active_piece:
            pygame.draw.circle(
                window,
                (255, 0, 0),
                (
                    active_piece[1] * CELL_SIZE + CELL_SIZE / 2,
                    active_piece[0] * CELL_SIZE + CELL_SIZE / 2,
                ),
                RADIUS + 4,
            )

        for i in range(9):
            for j in range(9):
                # Empty cell
                if self.board[i, j] == 0:
                    pygame.draw.circle(
                        window,
                        (255, 255, 255),
                        (j * CELL_SIZE + CELL_SIZE / 2, i * CELL_SIZE + CELL_SIZE / 2),
                        RADIUS / 2,
                    )
                elif self.board[i, j] == 1:
                    # Draw white piece
                    pygame.draw.circle(
                        window,
                        (0, 0, 0),
                        (j * CELL_SIZE + CELL_SIZE / 2, i * CELL_SIZE + CELL_SIZE / 2),
                        RADIUS + 1,
                    )
                    pygame.draw.circle(
                        window,
                        (255, 255, 255),
                        (j * CELL_SIZE + CELL_SIZE / 2, i * CELL_SIZE + CELL_SIZE / 2),
                        RADIUS,
                    )
                elif self.board[i, j] == 2:
                    # Draw black piece
                    pygame.draw.circle(
                        window,
                        (0, 0, 0),
                        (j * CELL_SIZE + CELL_SIZE / 2, i * CELL_SIZE + CELL_SIZE / 2),
                        RADIUS,
                    )

                # Write the coordinates
                text_c = (128, 128, 128)
                if self.board[i, j] == 1:
                    text_c = (0, 0, 0)
                if self.board[i, j] == 2:
                    text_c = (255, 255, 255)

                text = font.render(f"{string.ascii_uppercase[j]}{9-i}", True, text_c)
                # text = font.render(
                #     f"{string.ascii_uppercase[j]}{9-i} ({i},{j})", True, text_c
                # )

                window.blit(
                    text,
                    (
                        j * CELL_SIZE + CELL_SIZE / 2 - text.get_width() / 2,
                        i * CELL_SIZE + CELL_SIZE / 2 - text.get_height() / 2,
                    ),
                )

    def print_history(self, window):
        for i, move in enumerate(self.past_moves):
            if i % 2 == 0:
                p = "W"
            else:
                p = "B"

            move = f"{p}: {move}"
            text = font.render(move, True, (0, 0, 0))
            window.blit(
                text,
                (WIDTH - 200, i * 20 + 10),
            )

    def draw_possible_moves(self, window, piece):
        if type(piece) == str:
            piece = self.convert_coord_to_abs(piece)

        try:
            moves = self.legal_moves[f"{piece[0]}{piece[1]}"]
        except KeyError:
            moves = []
        for move in moves:
            pygame.draw.circle(
                window,
                (0, 255, 0),
                (
                    move[1] * CELL_SIZE + CELL_SIZE / 2,
                    move[0] * CELL_SIZE + CELL_SIZE / 2,
                ),
                CELL_SIZE / 8,
            )


class Engine:
    def __init__(self, player=1):
        self.player = player
        self.max_eval = 15

    def evaluate(self, board: Board, randomize=False, debug=False):
        unique, counts = np.unique(board.board, return_counts=True)
        count_dict = dict(zip(unique, counts))
        material_diff = count_dict[1] - count_dict[2]

        ones, _ = np.where(board.board == 1)
        twos, _ = np.where(board.board == 2)
        ones = 9 - ones
        twos = twos + 1
        avg_ones = np.mean(ones) / 9
        avg_twos = np.mean(twos) / 9

        game_over = board.is_game_over() * 100

        if debug:
            print(
                f"Material diff: {material_diff}, Avg 1: {avg_ones}, Avg 2: {avg_twos}, Game over: {game_over}, ones: {ones}, twos: {twos}"
            )

        return (
            (material_diff * 0.5) + (avg_ones * 0.1) - (avg_twos * 0.1) + game_over
        ) * (1 if self.player == 1 else -1)

    def get_best_move(self, board: Board, depth=3):
        best_eval = -np.inf
        best_move = None
        alpha = -np.inf
        beta = np.inf
        for piece, moves in board.legal_moves.items():
            piece = [int(piece[0]), int(piece[1])]
            for move in moves:
                board.move((piece, move))
                eval = self.minimax(board, depth - 1, False, alpha, beta)
                if eval > best_eval:
                    best_eval = eval
                    best_move = f"{board.convert_coord_to_str(piece)}-{board.convert_coord_to_str(move)}"
                board.undo_move()

        return best_move

    def minimax(self, board: Board, depth, is_maximizing, alpha, beta):
        if depth == 0:
            return self.evaluate(board)

        if is_maximizing:
            best_eval = -np.inf
            for piece, moves in board.legal_moves.items():
                piece = [int(piece[0]), int(piece[1])]
                for move in moves:
                    board.move((piece, move))
                    eval = self.minimax(board, depth - 1, False, alpha, beta)
                    best_eval = max(best_eval, eval)
                    alpha = max(alpha, eval)
                    board.undo_move()
                    if beta <= alpha:
                        break
            return best_eval
        else:
            best_eval = np.inf
            for piece, moves in board.legal_moves.items():
                piece = [int(piece[0]), int(piece[1])]
                for move in moves:
                    board.move((piece, move))
                    eval = self.minimax(board, depth - 1, True, alpha, beta)
                    best_eval = min(best_eval, eval)
                    beta = min(beta, eval)
                    board.undo_move()
                    if beta <= alpha:
                        break
            return best_eval


board = Board()
engine = Engine(player=1)
engine_black = Engine(player=2)
current_selection = None
current_board_eval = engine.evaluate(board)
move_count = 0

while not board.is_game_over():
    for event in pygame.event.get():
        if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
            pygame.quit()
            exit()
        if event.type == KEYDOWN:
            if event.key == K_r:
                board = Board()
                current_selection = None
                current_board_eval = engine.evaluate(board)
            elif event.key == K_e:
                print("Evaluating...")
                start_time = time()
                best_move = engine.get_best_move(board, depth=3)
                print(f"Time taken: {time()-start_time}")
                print(best_move)
                board.move(best_move)
                current_board_eval = engine.evaluate(board)
            elif event.key == K_u:
                board.undo_move()
                current_board_eval = engine.evaluate(board)

        if event.type == MOUSEBUTTONDOWN:
            pos = pygame.mouse.get_pos()
            x, y = pos
            x = x // CELL_SIZE
            y = y // CELL_SIZE

            if y > 8 or y < 0 or x > 8 or x < 0:
                continue

            new_selection = [y, x]

            if current_selection is not None:
                if (
                    new_selection
                    in board.legal_moves[
                        f"{current_selection[0]}{current_selection[1]}"
                    ]
                ):
                    board.move(
                        f"{board.convert_coord_to_str(current_selection)}-{board.convert_coord_to_str(new_selection)}"
                    )
                    current_board_eval = engine.evaluate(board)

                current_selection = None
            else:
                if board.board[y, x] == board.current_player:
                    current_selection = [y, x]

    window.fill(SCREEN_COLOR)

    # Draw board
    board.draw_board(window, active_piece=current_selection)
    board.print_history(window)

    if current_selection:
        board.draw_possible_moves(window, current_selection)

    # Current player text
    player = "White" if board.current_player == 1 else "Black"
    text = font.render(f"Current player: {player}", True, (0, 0, 0))
    window.blit(text, (WIDTH - 270, HEIGHT - 40))

    # Evaluation Bar
    # mapped_value = np.interp(
    #     -current_board_eval, [-engine.max_eval, engine.max_eval], [100, (HEIGHT - 100)]
    # )
    # pygame.draw.rect(
    #     window,
    #     (0, 255, 0),
    #     (9 * CELL_SIZE + 20, mapped_value, 50, (HEIGHT - 100) - (mapped_value + 0)),
    # )
    # pygame.draw.rect(window, (0, 0, 0), (9 * CELL_SIZE + 20, 100, 50, HEIGHT - 200), 5)
    text = font.render(
        f"{'+' if current_board_eval>=0 else ' '}{current_board_eval}", True, (0, 0, 0)
    )
    window.blit(text, (9 * CELL_SIZE + 30, HEIGHT - 90))

    pygame.display.flip()
    time.sleep(1)

    if move_count < 10:
        depth = 3
    elif move_count < 20:
        depth = 4
    else:
        depth = 5

    print(f"Player {player} Evaluating...")
    start_time = time.time()

    if board.current_player == 1:
        best_move = engine.get_best_move(board, depth=depth)
    else:
        best_move = engine_black.get_best_move(board, depth=depth)

    print(f"Time taken: {time.time()-start_time}")
    print(best_move)
    board.move(best_move)

    current_board_eval = engine.evaluate(board)

    # exit()
    move_count += 1

    dt = clock.tick(FPS) / 1000

print("Game Over")
print(f"Player {player} wins")
board.save_moves("moves.txt")

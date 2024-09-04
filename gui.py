import numpy as np
import pygame
from pygame.locals import *
import string
from copy import deepcopy

CELL_SIZE = 100

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


class Board:
    def __init__(self):
        self.board = self.new_board()
        self.current_player = 1
        self.past_moves = []

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

    def get_all_possible_moves(self, player):
        pass

    def get_possible_moves(self, piece):
        if type(piece) == str:
            piece = self.convert_coordinates(piece)

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
            if self.board[piece[0] - 1, piece[1] + 1] == 2:
                moves.append([piece[0] - 2, piece[1] + 2])
            if self.board[piece[0] - 1, piece[1] - 1] == 2:
                moves.append([piece[0] - 2, piece[1] - 2])
        elif player == 2:
            moves.append([piece[0] + 1, piece[1]])
            # Capture moves
            if self.board[piece[0] + 1, piece[1] + 1] == 1:
                moves.append([piece[0] + 2, piece[1] + 2])
            if self.board[piece[0] + 1, piece[1] - 1] == 1:
                moves.append([piece[0] + 2, piece[1] - 2])

        valid_moves = []
        for move in moves:
            # Check if move is within bounds
            if 0 < move[0] < 9 and 0 < move[1] < 9:
                # End position is empty
                if self.board[move[0], move[1]] == 0:
                    valid_moves.append(move)

        only_captures = []
        for move in valid_moves:
            if abs(piece[0] - move[0]) == 2:
                only_captures.append(move)

        if only_captures:
            return only_captures

        return valid_moves

    def move(self, move: string):
        # String in format A1-A2
        start = self.convert_coordinates(move[:2])
        end = self.convert_coordinates(move[3:])

        if self.board[start[0], start[1]] != self.current_player:
            print("Invalid move")
            return

        if not self.is_move_valid(start, end):
            print("Invalid move")
            return

        # Check if move is a capture
        if abs(start[0] - end[0]) == 2:
            self.board[
                start[0] + (end[0] - start[0]) // 2, start[1] + (end[1] - start[1]) // 2
            ] = 0

        self.board[end[0], end[1]] = self.board[start[0], start[1]]
        self.board[start[0], start[1]] = 0

        self.past_moves.append(move)
        self.current_player = self.current_player % 2 + 1

    def convert_coordinates(self, pos: string):
        # Pos in format A1 to (8, 0)
        pos = pos.upper()
        return [9 - int(pos[1]), string.ascii_uppercase.index(pos[0])]

    def is_game_over(self):
        if 1 in self.board[0, :]:
            return 1
        if 2 in self.board[8, :]:
            return 2

    def is_move_valid(self, start, end):
        moves = self.get_possible_moves(start)
        if end not in moves:
            return False

        return True

    def draw_board(self, window):
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

        for i in range(9):
            for j in range(9):
                radius = CELL_SIZE / 2 - 5
                if self.board[i, j] == 0:
                    pygame.draw.circle(
                        window,
                        (255, 255, 255),
                        (j * CELL_SIZE + CELL_SIZE / 2, i * CELL_SIZE + CELL_SIZE / 2),
                        radius / 2,
                    )
                elif self.board[i, j] == 1:
                    pygame.draw.circle(
                        window,
                        (0, 0, 0),
                        (j * CELL_SIZE + CELL_SIZE / 2, i * CELL_SIZE + CELL_SIZE / 2),
                        radius + 1,
                    )
                    pygame.draw.circle(
                        window,
                        (255, 255, 255),
                        (j * CELL_SIZE + CELL_SIZE / 2, i * CELL_SIZE + CELL_SIZE / 2),
                        radius,
                    )
                elif self.board[i, j] == 2:
                    pygame.draw.circle(
                        window,
                        (0, 0, 0),
                        (j * CELL_SIZE + CELL_SIZE / 2, i * CELL_SIZE + CELL_SIZE / 2),
                        radius,
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
            piece = self.convert_coordinates(piece)

        moves = self.get_possible_moves(piece)
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


board = Board()
current_selection = None

while True:
    for event in pygame.event.get():
        if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
            pygame.quit()
            exit()
        if event.type == MOUSEBUTTONDOWN:
            pos = pygame.mouse.get_pos()
            x, y = pos
            x = x // CELL_SIZE
            y = y // CELL_SIZE

            if current_selection is not None:
                if [y, x] in board.get_possible_moves(current_selection):
                    board.move(
                        f"{string.ascii_uppercase[current_selection[1]]}{9 - current_selection[0]}-{string.ascii_uppercase[x]}{9 - y}"
                    )

                current_selection = None
            else:
                if board.board[y, x] == board.current_player:
                    current_selection = [y, x]

    window.fill(SCREEN_COLOR)

    # Draw board
    board.draw_board(window)
    board.print_history(window)
    if current_selection:
        board.draw_possible_moves(window, current_selection)

    # Current player
    player = "White" if board.current_player == 1 else "Black"
    text = font.render(f"Current player: {player}", True, (0, 0, 0))
    window.blit(text, (WIDTH - 250, HEIGHT - 50))

    pygame.display.flip()

    dt = clock.tick(FPS) / 1000

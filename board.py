import numpy as np
import pygame
from pygame.locals import *
import string
from parameters import *


pygame.font.init()
font = pygame.font.SysFont("Arial", 24)


class Board:
    def __init__(self):
        self.board = self.new_board()
        self.legal_moves = {}
        self.current_player = 1
        self.past_moves = []
        self.past_legal_moves = []

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

    def __eq__(self, other: object) -> bool:
        return self.board.tobytes() == other.board.tobytes() and isinstance(
            other, Board
        )

    def __hash__(self) -> int:
        return hash(self.board.tobytes())

    def calculate_legal_moves(self):
        if self.is_game_over():
            self.legal_moves = {}
            return
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
        self.past_legal_moves.append(self.legal_moves)
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

        self.legal_moves = self.past_legal_moves.pop()

        # self.calculate_legal_moves()

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
            return 2
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
        self.calculate_legal_moves()

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

        # Draw last move
        if self.past_moves:
            move = self.past_moves[-1]
            start = self.convert_coord_to_abs(move[:2])
            end = self.convert_coord_to_abs(move[3:])
            pygame.draw.line(
                window,
                (0, 255, 0),
                (
                    start[1] * CELL_SIZE + CELL_SIZE / 2,
                    start[0] * CELL_SIZE + CELL_SIZE / 2,
                ),
                (
                    end[1] * CELL_SIZE + CELL_SIZE / 2,
                    end[0] * CELL_SIZE + CELL_SIZE / 2,
                ),
                5,
            )
            pygame.draw.circle(
                window,
                (0, 255, 0),
                (
                    start[1] * CELL_SIZE + CELL_SIZE / 2,
                    start[0] * CELL_SIZE + CELL_SIZE / 2,
                ),
                RADIUS / 3 * 2,
            )
            pygame.draw.circle(
                window,
                (0, 255, 0),
                (
                    end[1] * CELL_SIZE + CELL_SIZE / 2,
                    end[0] * CELL_SIZE + CELL_SIZE / 2,
                ),
                RADIUS + 4,
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
        for i, move in enumerate(self.past_moves[-35:]):
            j = i + (len(self.past_moves) + 1 if len(self.past_moves) > 35 else 0)
            if j % 2 == 0:
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

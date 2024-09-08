import numpy as np
from time import time, sleep
from parameters import *
from board import Board


class Engine:
    def __init__(self, player=1, depth=3, max_time=0):
        self.player = player
        self.transposition_table = {}
        self.depth = depth
        self.max_time = max_time
        self.start_time = time()

    def evaluate(self, board: Board):
        player_prospective = (1 if board.current_player == self.player else -1) * (
            1 if self.player == 1 else -1
        )

        ones, _ = np.where(board.board == 1)
        twos, _ = np.where(board.board == 2)

        material_diff = ones.size - twos.size
        if 0 in ones:
            return 1000 * player_prospective
        elif 8 in twos:
            return -1000 * player_prospective

        ones = 9 - ones
        twos = twos + 1
        twos = twos[::-1]

        avg_ones = np.average(ones, weights=range(ones.size + 1, 1, -1))
        avg_twos = np.average(twos, weights=range(twos.size + 1, 1, -1))

        return (
            (material_diff * 5) + (avg_ones * 1) - (avg_twos * 1)
        ) * player_prospective

    def get_best_move(self, board: Board):
        self.start_time = time()
        best_eval = -np.inf
        best_move = None
        best_move_sequence = []
        alpha = -np.inf
        beta = np.inf

        for piece, moves in board.legal_moves.items():
            piece_coords = [int(piece[0]), int(piece[1])]
            for move in moves:
                board.move((piece_coords, move))
                eval, move_sequence = self.negamax(board, self.depth - 1, -beta, -alpha)
                eval = -eval
                if eval > best_eval:
                    best_eval = eval
                    best_move = (piece_coords, move)
                    best_move_sequence = [
                        f"{board.convert_coord_to_str(piece_coords)}-{board.convert_coord_to_str(move)}",
                    ] + move_sequence
                alpha = max(alpha, eval)
                board.undo_move()
                if alpha >= beta:
                    break

        # while len(self.transposition_table) > 100000:
        #     self.transposition_table.pop(next(iter(self.transposition_table)))
        self.transposition_table = {}

        if best_move:
            best_move_str = f"{board.convert_coord_to_str(best_move[0])}-{board.convert_coord_to_str(best_move[1])}"
            # pprint(
            #     f"Expected eval: {best_eval}, Best move: {best_move_str}, Best move sequence: {best_move_sequence}"
            # )
            for move in best_move_sequence:
                board.move(move)
                print(
                    f"{move}: {self.evaluate(board) * (1 if self.player == board.current_player else -1) * (1 if self.player == 1 else -1)}"
                )

            for _ in best_move_sequence:
                board.undo_move()

            return best_move_str

        return None

    def negamax(self, board: Board, depth, alpha, beta, hash_table=True):
        board_hash = str(board.board.tobytes())
        if hash_table and board_hash in self.transposition_table:
            return self.transposition_table[board_hash]

        if (
            depth == 0
            or board.is_game_over()
            or (self.max_time and (time() - self.start_time) > self.max_time)
        ):
            eval = self.evaluate(board)
            self.transposition_table[board_hash] = (eval, [])
            return eval, []

        best_eval = -np.inf
        best_move_sequence = []

        for piece, moves in board.legal_moves.items():
            piece_coords = [int(piece[0]), int(piece[1])]
            for move in moves:
                board.move((piece_coords, move))
                eval, move_sequence = self.negamax(board, depth - 1, -beta, -alpha)
                eval = -eval
                if eval > best_eval:
                    best_eval = eval
                    best_move_sequence = [
                        f"{board.convert_coord_to_str(piece_coords)}-{board.convert_coord_to_str(move)}",
                    ] + move_sequence
                alpha = max(alpha, eval)
                board.undo_move()
                if alpha >= beta:
                    break

        self.transposition_table[board_hash] = (best_eval, best_move_sequence)
        return best_eval, best_move_sequence

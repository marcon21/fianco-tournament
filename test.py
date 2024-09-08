import fianco_tournament as ft
import numpy as np
from time import time
from board import Board

board = Board()
# board.board[5, 5] = 1
# board.board[0, 0] = 1

# print(ft.get_possible_moves(board.board, (5, 5)))
# print(ft.get_possible_moves(board.board, (0, 0)))

# print(board.get_possible_moves((5, 5)))
# print(board.get_possible_moves((0, 0)))

# print(board.get_all_possible_moves(player=1))

# aaa = ft.get_all_possible_moves(board.board, 1)

t = time()
for i in range(100000):
    # ft.get_all_possible_moves(board.board, 1)
    ft.empty(board.board)
print("Rust", time() - t)

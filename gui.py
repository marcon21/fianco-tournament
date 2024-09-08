import numpy as np
import pygame
from pygame.locals import *
from time import time, sleep
from parameters import *
from board import Board
from engine import Engine

pygame.init()
window = pygame.display.set_mode(GAME_RES, HWACCEL | HWSURFACE | DOUBLEBUF)
clock = pygame.time.Clock()
dt = 0
pygame.display.set_caption(GAME_TITLE)

pygame.font.init()
font = pygame.font.SysFont("Arial", 24)


board = Board()
engine_white = Engine(player=1, depth=DEPTH, max_time=MAX_TIME)
engine_black = Engine(player=2, depth=DEPTH, max_time=MAX_TIME)
current_board_eval = engine_white.evaluate(board)

current_selection = None
move_count = 0


import fianco_tournament as ft


def think_best_move(board):
    print(f"Player {player} Evaluating...")
    start_time = time()
    if board.current_player == 1:
        # best_move = engine_white.get_best_move(board)
        best_move = ft.get_best_move(board.board, board.current_player, DEPTH)
    else:
        # best_move = engine_black.get_best_move(board)
        best_move = ft.get_best_move(board.board, board.current_player, DEPTH)
    print(f"Time taken: {time()-start_time:.2f}, Move: {best_move}")
    return best_move


while True:
    player = "White" if board.current_player == 1 else "Black"

    # assert engine_white.evaluate(board) == -engine_black.evaluate(board), (
    #     engine_white.evaluate(board),
    #     engine_black.evaluate(board),
    # )

    for event in pygame.event.get():
        if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
            pygame.quit()
            exit()
        if event.type == KEYDOWN:
            if event.key == K_r:
                # Reset board
                board = Board()
                current_selection = None
                current_board_eval = engine_white.evaluate(board)
            elif event.key == K_e:
                # Evaluate with engine and make move
                best_move = think_best_move(board)
                board.move(best_move)
                current_board_eval = engine_white.evaluate(board)
            elif event.key == K_t:
                # Think and print best move
                best_move = think_best_move(board)
            elif event.key == K_u:
                # Undo move
                board.undo_move()
                current_board_eval = engine_white.evaluate(board)
                current_selection = None
                move_count -= 1
            elif event.key == K_UP:
                DEPTH += 1
                print(f"Depth set to: {DEPTH}")
                engine_white.depth = DEPTH
                engine_black.depth = DEPTH
            elif event.key == K_DOWN:
                DEPTH -= 1
                print(f"Depth set to: {DEPTH}")
                engine_white.depth = DEPTH
                engine_black.depth = DEPTH

        if event.type == MOUSEBUTTONDOWN:
            # Manual Move
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
                    current_board_eval = engine_white.evaluate(board)

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
    text = font.render(f"Current player: {player}", True, (0, 0, 0))
    window.blit(text, (WIDTH - 270, HEIGHT - 40))

    # Evaluation Bar
    text = font.render(
        f"{'+' if current_board_eval>=0 else ' '}{current_board_eval}", True, (0, 0, 0)
    )
    window.blit(text, (9 * CELL_SIZE + 30, HEIGHT - 90))

    pygame.display.flip()

    if board.is_game_over():
        print("Game Over")
        player = "White" if board.current_player == 2 else "Black"
        print(f"Player {player} wins")
        while True:
            pass

    best_move = think_best_move(board)
    board.move(best_move)
    current_board_eval = engine_white.evaluate(board)
    sleep(1)

    dt = clock.tick(FPS) / 1000

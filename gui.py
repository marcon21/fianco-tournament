import numpy as np
import pygame
from pygame.locals import *
from time import time, sleep
from parameters import *
from board import Board

import fianco_tournament as ft
from functools import lru_cache

pygame.init()
window = pygame.display.set_mode(GAME_RES, HWACCEL | HWSURFACE | DOUBLEBUF)
clock = pygame.time.Clock()
dt = 0
pygame.display.set_caption(GAME_TITLE)

pygame.font.init()
font = pygame.font.SysFont("Arial", 24)


board = Board()

evaluate_function = lru_cache(maxsize=1000)(
    lambda board: round(ft.evaluate_stand_alone(board.board, 1, 1), 3)
)

current_board_eval = evaluate_function(board)

current_selection = None
move_count = 0
white_time = 0
black_time = 0


def think_best_move(board, time_it=True):
    global white_time, black_time
    print(f"Player {player} Evaluating...")
    start_time = time()
    if board.current_player == 1:
        best_move, expected_res = ft.get_best_move(
            board.board, board.current_player, 1, DEPTH, MAX_TIME
        )
        if time_it:
            white_time += time() - start_time
        print(f"White time: {white_time:.2f}")
    else:
        best_move, expected_res = ft.get_best_move(
            board.board, board.current_player, 2, DEPTH, MAX_TIME
        )
        if time_it:
            black_time += time() - start_time
        print(f"Black time: {black_time:.2f}")

    print(
        f"Time taken: {time()-start_time:.2f}, Move: {best_move}, Expected Eval: {expected_res}, Total time: {white_time+black_time:.2f}"
    )

    return best_move


while True:
    player = "White" if board.current_player == 1 else "Black"
    current_board_eval = evaluate_function(board)
    try:
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                exit()
            if event.type == KEYDOWN:
                if event.key == K_r:
                    pass
                #     # Reset board
                #     board = Board()
                #     current_selection = None
                elif event.key == K_e:
                    # Evaluate with engine and make move
                    current_selection = None
                    best_move = think_best_move(board)
                    board.move(best_move)
                elif event.key == K_t:
                    # Think and print best move
                    best_move = think_best_move(board)
                elif event.key == K_u:
                    # Undo move
                    board.undo_move()
                    current_selection = None
                elif event.key == K_UP:
                    DEPTH += 1
                    print(f"Depth set to: {DEPTH}")
                elif event.key == K_DOWN:
                    DEPTH -= 1
                    print(f"Depth set to: {DEPTH}")

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

                    current_selection = None
                else:
                    if board.board[y, x] == board.current_player:
                        current_selection = [y, x]
    except Exception as e:
        print(e)

    window.fill(SCREEN_COLOR)

    # Draw board
    board.draw_board(window, active_piece=current_selection)
    board.print_history(window)

    if current_selection:
        board.draw_possible_moves(window, current_selection)

    # Time text
    text = font.render(f"Total: {white_time+black_time:.2f}s", True, (0, 0, 0))
    window.blit(text, (WIDTH - 270, HEIGHT - 190))

    text = font.render(f"Black: {black_time:.2f}s", True, (0, 0, 0))
    window.blit(text, (WIDTH - 270, HEIGHT - 160))

    text = font.render(f"White: {white_time:.2f}s", True, (0, 0, 0))
    window.blit(text, (WIDTH - 270, HEIGHT - 130))

    # Depth text
    text = font.render(f"Depth: {DEPTH}", True, (0, 0, 0))
    window.blit(text, (WIDTH - 270, HEIGHT - 100))

    # Evaluation Text
    text = font.render(
        f"Eval: {'+' if current_board_eval>=0 else ' '}{current_board_eval}",
        True,
        (0, 0, 0),
    )
    window.blit(text, (WIDTH - 270, HEIGHT - 70))

    # Current player text
    text = font.render(f"Current player: {player}", True, (0, 0, 0))
    window.blit(text, (WIDTH - 270, HEIGHT - 40))

    pygame.display.flip()

    if board.is_game_over():
        print("Game Over")
        player = "White" if board.current_player == 2 else "Black"
        print(f"Player {player} wins")
        board.save_moves("last_game.txt")

    # current_selection = None
    # best_move = think_best_move(board)
    # board.move(best_move)
    # sleep(1)

    # if board.current_player == 2:
    #     current_selection = None
    #     best_move = think_best_move(board)
    #     board.move(best_move)
    #     sleep(1)

    dt = clock.tick(FPS) / 1000

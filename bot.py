"""
The Brandeis Quant Club ML/AI Competition (November 2023)

Author: @Ephraim Zimmerman
Email: quants@brandeis.edu
Website: brandeisquantclub.com; quants.devpost.com

Description:

For any technical issues or questions please feel free to reach out to
the "on-call" hackathon support member via email at quants@brandeis.edu

Website/GitHub Repository:
You can find the latest updates, documentation, and additional resources for this project on the
official website or GitHub repository: https://github.com/EphraimJZimmerman/chess_hackathon_23

License:
This code is open-source and released under the MIT License. See the LICENSE file for details.
"""

import random
import chess
import time
from collections.abc import Iterator
from contextlib import contextmanager
import test_bot


@contextmanager
def game_manager() -> Iterator[None]:
    """Creates context for game."""

    print("===== GAME STARTED =====")
    ping: float = time.perf_counter()
    try:
        # DO NOT EDIT. This will be replaced w/ judging context manager.
        yield
    finally:
        pong: float = time.perf_counter()
        total = pong - ping
        print(f"Total game time = {total:.3f} seconds")
    print("===== GAME ENDED =====")


def count_pawn_structure(board) -> []:
    doubled_pawns = 0
    blocked_pawns = 0
    isolated_pawns = 0
    for square in chess.SQUARES:
        if board.piece_at(square) == chess.PAWN:
            pawn_color = board.color_at(square)

            file = chess.square_file(square)

            if any(board.piece_at(chess.square(file, i)) == chess.PAWN and board.color_at(
                    chess.square(file, i)) == pawn_color for i in range(8)):
                doubled_pawns += 1

            if (pawn_color == chess.WHITE and board.piece_at(square + 8) is not None) or \
                    (pawn_color == chess.BLACK and board.piece_at(square - 8) is not None):
                blocked_pawns += 1

            adjacent_files = [file - 1, file + 1]
            for adj_file in adjacent_files:
                if 0 <= adj_file < 8 and \
                        any(board.piece_at(chess.square(adj_file, i)) == chess.PAWN and board.color_at(
                            chess.square(adj_file, i)) == pawn_color for i in range(8)):
                    break
            else:
                isolated_pawns += 1

    return doubled_pawns, blocked_pawns, isolated_pawns


def order_moves(moves):
    # gonna implement move importance
    return moves


# class pos_info:
#     def __init__(self, key, depth, evaluation, flag):
#         self.key = key
#         self.depth = depth
#         self.evaluation = evaluation
#         self.flag = flag


# class transposition_table:
#     def __init__(self, capacity):
#         self.capacity = capacity
#         self.size = 0
#         self.hashtable = [None] * capacity
#         self.table = [[[random.getrandbits(64) for k in range(12)] for j in range(8)] for i in range(8)]

#     def zobrist(self, board):
#         h = 0
#         for i in range(8):
#             for j in range(8):
#                 if board.piece_at(chess.square(i, j)) is not None:
#                     piece = board.piece_at(chess.square(i, j)).piece_type
#                     h ^= self.table[i][j][piece]
#         return h

#     def hash_z(self, z):
#         return z % self.capacity

#     def add(self, pos_info):
#         z = pos_info.key
#         pos = self.hash_z(z)
#         if self.hashtable[pos] is None:
#             self.hashtable[pos] = pos_info
#             self.size += 1
#         else:
#             while self.hashtable[pos] is not None:
#                 if self.hashtable[pos].key == z:
#                     if pos_info.depth > self.hashtable[pos].depth:
#                         self.hashtable[pos] = pos_info
#                     return
#                 pos += 1
#                 if pos >= self.capacity:
#                     pos = 0
#             self.hashtable[pos] = pos_info
#             self.size += 1

#     def get(self, key):
#         pos = self.hash_z(key)
#         if self.hashtable[pos] is None:
#             return None
#         if self.hashtable[pos] == key:
#             return self.hashtable[pos]
#         while self.hashtable[pos] != key:
#             if self.hashtable[pos] is None:
#                 return None
#             pos += 1
#         return self.hashtable[pos]


class Bot:
    def __init__(self, fen=None):
        self.board = chess.Board(fen if fen else "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        self.count = 0

    def check_move_is_legal(self, initial_position, new_position) -> bool:
        """
            To check if, from an initial position, the new position is valid.

            Args:
                initial_position (str): The starting position given chess notation.
                new_position (str): The new position given chess notation.

            Returns:
                bool: If this move is legal
        """

        return chess.Move.from_uci(initial_position + new_position) in self.board.legal_moves

    def next_move(self) -> str:
        """
            The main call and response loop for playing a game of chess.

            Returns:
                str: The current location and the next move.
        """
        depth = 4

        # move = str(random.choice([_ for _ in self.board.legal_moves]))
        legal_moves = list(self.board.legal_moves)

        if legal_moves:
            # Check if all available moves lead to stalemate
            all_stalemate_moves = all(self.results_in_stalemate(move) for move in legal_moves)

            if all_stalemate_moves:
                # If all moves lead to stalemate, proceed with any move
                move = str(legal_moves[0])
            else:
                # Otherwise, find a move that doesn't result in stalemate
                move = self.find_non_stalemate_move(legal_moves, depth)

            print("My move: " + move)
            print("score: " + str(self.evaluate_pos(self.board)))
            print("positions evaluated: " + str(self.count))
            self.count = 0
            return move

    def find_non_stalemate_move(self, legal_moves, depth):
        if legal_moves:
            scores = []
            new_board = self.board.copy()
            for move in legal_moves:
                new_board.push(move)
                # child_node = TreeNode(board=new_board, parent=None, move=move)
                # build_game_tree(child_node, depth - 1)
                scores.append(self.minimax(new_board, depth - 1))
                new_board.pop()

            best_move = legal_moves[scores.index(max(scores))]
            move = str(best_move)
            return move

    def results_in_stalemate(self, move):
        new_board = self.board.copy()
        new_board.push(move)
        return new_board.is_stalemate()

    def minimax(self, board, depth, alpha=float('-inf'), beta=float('inf'), maximizing_player=True):
        # print("minimax")
        # tt = transposition_table(capacity=200000)
        # z = tt.zobrist(board)
        # key = tt.hash_z(z)
        # if tt.get(key) is not None:
        #     return tt.get(key).flag

        legalmoves = list(board.legal_moves)
        if depth == 0 or not legalmoves:
            evaluation = self.evaluate_pos(board)
            # if maximizing_player:
            #     tt.add(pos_info(key, depth, evaluation, alpha))
            # else:
            #     tt.add(pos_info(key, depth, evaluation, beta))
            return evaluation

        if maximizing_player:
            max_eval = float('-inf')
            for child in order_moves(legalmoves):  # Order moves for better alpha-beta pruning
                board.push(child)
                eval_child = self.minimax(board, depth - 1, alpha, beta, False)
                board.pop()
                max_eval = max(max_eval, eval_child)
                alpha = max(alpha, eval_child)
                if beta <= alpha:
                    break  # Beta cut-off
            return max_eval
        else:
            min_eval = float('inf')
            for child in order_moves(legalmoves):
                board.push(child)
                eval_child = self.minimax(board, depth - 1, alpha, beta, True)
                board.pop()
                min_eval = min(min_eval, eval_child)
                beta = min(beta, eval_child)
                if beta <= alpha:
                    break  # Alpha cut-off
            return min_eval

    def evaluate_pos(self, board) -> int:
        score = 0
        self.count += 1
        if self.board.turn:
            piece_values = [1, 3, 3, 5, 9, 1000]
            for piece_type, value in zip(range(1, 7), piece_values):
                score += value * (len(board.pieces(piece_type, True)) - len(board.pieces(piece_type, False)))

            # Reward central pawns
            for square in board.pieces(chess.PAWN, chess.WHITE):
                rank = chess.square_rank(square) - 2
                file = chess.square_file(square)
                # Assign higher scores to central pawns
                if file in [3, 4]:
                    score += 0.2 * rank  # Adjusted the weight for central pawns
                else:
                    score += 0.1 * rank

        else:
            piece_values = [1, 3, 3, 5, 9, 1000]
            for piece_type, value in zip(range(1, 7), piece_values):
                score += value * (len(board.pieces(piece_type, False)) - len(board.pieces(piece_type, True)))

            # Reward central pawns
            for square in board.pieces(chess.PAWN, chess.BLACK):
                rank = 8 - chess.square_rank(square) - 2
                file = chess.square_file(square)
                # Assign higher scores to central pawns
                if file in [3, 4]:
                    score += 0.2 * rank  # Adjusted the weight for central pawns
                else:
                    score += 0.1 * rank

        # Penalize pawn structure
        score += -.5 * sum(count_pawn_structure(board))
        if board.is_stalemate():
            score = 0
        if board.is_checkmate() and self.board.turn:
            score = 9999999999
        if board.is_check():
            score += .1

        return score


if __name__ == "__main__":

    chess_bot = Bot()  # you can enter a FEN here, like Bot("...")
    with game_manager():

        """
        
        Feel free to make any adjustments as you see fit. The desired outcome 
        is to generate the next best move, regardless of whether the bot 
        is controlling the white or black pieces. The code snippet below 
        serves as a useful testing framework from which you can begin 
        developing your strategy.

        """

        playing = True

        while playing:
            if chess_bot.board.turn:
                chess_bot.board.push_san(test_bot.get_move(chess_bot.board))
            else:
                chess_bot.board.push_san(chess_bot.next_move())
            print(chess_bot.board, end="\n\n")

            if chess_bot.board.is_game_over():
                if chess_bot.board.is_stalemate():
                    print("Is stalemate")
                elif chess_bot.board.is_insufficient_material():
                    print("Is insufficient material")

                # EX: Outcome(termination=<Termination.CHECKMATE: 1>, winner=True)
                print(chess_bot.board.outcome())

                playing = False

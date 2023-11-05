
import chess
import requests
import logging
import random

# DO NOT MODIFY


def get_move(board: chess.Board, best_move=False) -> str:
    """

    Gets a move for the current white piece.

    **Bot is not smart -- not intended to be used for training, only for testing
    game logic.

    Args:
        board (chess.Board): The entire chess board, which will be evaluated.
        best_move (bool): If the bot should, when applicable, find the best move.

    Returns:
        str: The next move.

    """

    if not best_move:
        if board.is_checkmate() or board.is_game_over():
            board.is_game_over()
        else:
            move = str(random.choice([_ for _ in board.legal_moves]))
            return move
    else:

        url = "https://www.chessdb.cn/cdb.php?action=querybest&board=" + board.fen().replace(" ", "%")

        try:
            if board.turn:
                response = requests.get(url, timeout=10.0)
                if response.status_code == 200:
                    content = response.text[5:9]
                    if content == "tmov":
                        if board.is_checkmate() or board.is_game_over():
                            board.is_game_over()
                        else:
                            move = str(random.choice([_ for _ in board.legal_moves]))
                            return move
                    return content
            else:
                logging.exception("The bot can only play offense.")
        except requests.Timeout as timout_error:
            logging.error(timout_error)
            move = str(random.choice([_ for _ in board.legal_moves]))
            return move
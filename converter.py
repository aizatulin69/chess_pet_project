import io
import zstandard as zstd
from engine import Engine
import chess
import chess.pgn
import pandas as pd
import time

e = Engine()

TOTAL_SAMPLES = 1000.0


def stream_zst_pgn(file_path):
    dctx = zstd.ZstdDecompressor()
    
    with open(file_path, 'rb') as f:
        with dctx.stream_reader(f) as reader:
            text = io.TextIOWrapper(reader, encoding='utf-8', errors='replace')
            print(text)
            
            while True:
                game = chess.pgn.read_game(text)
                if game is None:
                    break
                
                if game.headers["Termination"] != "Normal":
                    continue
                
                we = int(game.headers["WhiteElo"])
                be = int(game.headers["BlackElo"])
                if we<1000 or be<1000: continue
                    
                if game.headers["Result"] == "1-0": 
                    result = 1
                elif game.headers["Result"] == "0-1": 
                    result = -1
                else: continue
                
                board = game.board()
                samples = []
                moves = []
                boards = []
                turns = []
                results = []
                i = 0

                for move in game.mainline_moves():
                    move_encoded = e.encode_single_move(
                        (move.from_square // 8 +1) * 10 + (move.from_square % 8 +1),
                        (move.to_square // 8 +1) * 10 + (move.to_square % 8 +1),
                        {chess.ROOK: 2, chess.KNIGHT: 3, chess.BISHOP: 4, chess.QUEEN: 6}.get(move.promotion, 0))
                    moves.append(move_encoded)
                    boards.append(e.return_board_list())
                    turns.append(e.turn)
                    e.move_simplified(move_encoded)
                    board.push(move)
                    if result == 1:
                        results.append((-1)**i)
                    else:
                        results.append((-1)**(i+1))
                    i += 1
                    
                e.reset()

                for r, m, b, t in zip (results, moves, boards, turns):
                    samples.append((r, m, b, t))
                yield from samples


def main():
    file_path = "lichess_db_standard_rated_2017-02.pgn.zst"
    results = []
    encoded_moves = []
    boards = []
    turns = []
    
    
    total_w = 0.0
    total_b = 0.0

    games_generator = stream_zst_pgn(file_path)
    start_time = time.time()

    for sample in games_generator:
        if sample[0] == 1:
            if total_w == TOTAL_SAMPLES:
                continue
            total_w += 1
        else:
            total_b += 1
            if total_b == TOTAL_SAMPLES:
                break
        results.append(sample[0])
        encoded_moves.append(sample[1])
        boards.append(sample[2])
        turns.append(sample[3])

    current_time = time.time()
    elapsed = current_time - start_time
    print(f"{(elapsed//60):.0f}:{(elapsed%60):.0f}")
    print(total_w, total_b)

    df = pd.DataFrame({"value_target": results, 
                       "policy_target": encoded_moves, 
                       "board": boards,
                       "turn": turns})
    
    df.to_csv("dataset.csv", index=False)

if __name__ == "__main__":
    main()
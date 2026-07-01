from chess_NN import Engine, PlayerModel, ChessModel
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from multiprocessing import Pool
import pandas as pd
import time
from numpy import e


MAX_GAMES = 100000
SAVE_EVERY = 5000
PRINT_EVERY = 64
N_ENVS = 8
GAMES_PLAYED = 0


def init_worker():
    global game, model, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChessModel()
    model.eval()
    game = Engine()


def board_to_tensor(board):
    board_tensor = torch.tensor(board, dtype=torch.float32).view(8, 8)
    tensor = torch.zeros(12, 8, 8, dtype=torch.float32)
    for piece in range(1, 7):
        tensor[piece - 1] = (board_tensor == piece).float()
    for piece in range(1, 7):
        tensor[piece + 5] = (board_tensor == -piece).float()
    return tensor.unsqueeze(0)

def play_game_worker(state_dict):
    global model, game, GAMES_PLAYED
    GAMES_PLAYED += 1
    model.load_state_dict(state_dict)
    game.reset()

    states = []
    legal_actions_history = []
    action_indices = []

    done = False
    moves = 0

    while not done:
        legal_actions = game.return_alm()
        x = board_to_tensor(game.return_board_list()).to(device)

        with torch.no_grad():
            logits, _ = model(x)
        logits = logits[0]

        logits_valid = logits[legal_actions]
        temperature = 0.5 + 4.5*(1-(GAMES_PLAYED)/50000)**2
        if temperature < 0.5: temperature = 1.0
        probs = F.softmax(logits_valid / temperature, dim=0)
        dist = Categorical(probs)
        action_idx = dist.sample()
        action = legal_actions[action_idx]

        action_indices.append(action_idx)
        legal_actions_history.append(legal_actions)
        states.append(board_to_tensor(game.return_board_list()))

        game.move(action)
        moves += 1

        if moves == 200 or game.return_ending() != 0:
            game.reset()
            done = True

    return {
        "states": states,
        "action_indices": action_indices,
        "legal_actions_history": legal_actions_history,
        "result": game.return_ending(),
        "moves": moves
    }



def main():
    player = PlayerModel()

    results = []
    moves_stats = []
    policy_loss = []
    value_loss = []
    entropy = []

    total_games = 0

    def avg(arr):
        return sum(arr) / len(arr) if len(arr) > 0 else 0
    
    start_time = time.time()

    with Pool(processes=N_ENVS, initializer=init_worker) as pool:
        while total_games < MAX_GAMES:
            batch = pool.map(play_game_worker, [player.model.state_dict()] * N_ENVS )
            player.model.train()
            total_policy_loss = 0
            total_value_loss = 0

            batch_policy_losses = []
            batch_value_losses = []
            batch_entropies = []

            player.opt.zero_grad() 
 
            for episode in batch:
                states = episode["states"]
                moves = episode["moves"]

                results.append(episode["result"])
                moves_stats.append(moves)

                if episode["result"] in (1, -1):
                    returns = [(-1)**i for i in range(moves)][::-1]
                else:
                    returns = [0.0 for _ in range(moves)]

                returns = torch.tensor(returns, dtype=torch.float32)
                x = torch.cat(states, dim=0)
                
                logits, values = player.model(x)
                values = values.squeeze(-1)

                episode_policy_loss = 0
                episode_entropy = 0

                for t in range(len(states)):
                    step_logits = logits[t]
                    legal_actions = episode["legal_actions_history"][t]
                    valid_logits = step_logits[legal_actions]
                    valid_probs = F.softmax(valid_logits, dim=0)

                    log_probs = torch.log(valid_probs + 1e-8)   
                    e = -torch.sum(valid_probs * log_probs)
                    episode_entropy += e.item()

                    action_idx = episode["action_indices"][t]
                    log_prob = torch.log(valid_probs[action_idx] + 1e-8)
                    advantage = returns[t] - values[t].detach()
                    episode_policy_loss -= advantage * log_prob

                avg_episode_policy_loss = episode_policy_loss / len(states)
                avg_episode_entropy = episode_entropy / len(states)

                total_policy_loss += episode_policy_loss / len(states)
                total_value_loss += F.huber_loss(values, returns)
                episode_value_loss = F.huber_loss(values, returns)

                batch_policy_losses.append(avg_episode_policy_loss.item())
                batch_value_losses.append(episode_value_loss.item())
                batch_entropies.append(avg_episode_entropy)

                loss = (avg_episode_policy_loss + episode_value_loss - 0.001 * avg_episode_entropy)/N_ENVS
                
                player.opt.zero_grad()
                loss.backward()

            torch.nn.utils.clip_grad_norm_(player.model.parameters(), 1.0)
            player.opt.step()


            total_games += N_ENVS
            if total_games % PRINT_EVERY == 0:
                res = results[-PRINT_EVERY:]
                endings = [len([x for x in res if x == i]) for i in (0, 1, -1, 2)]
                moves_avg = avg(moves_stats[-PRINT_EVERY:])
                avg_p_loss = avg(batch_policy_losses)
                avg_v_loss = avg(batch_value_losses)
                avg_entr = avg(batch_entropies)

                policy_loss.append(avg_p_loss)
                value_loss.append(avg_v_loss)
                entropy.append(avg_entr)

                current_time = time.time()
                elapsed_time = current_time - start_time

                print(f"\n=== {total_games - PRINT_EVERY} -- {total_games} ===")
                print(f"Время: {elapsed_time:.2f}")
                print(f"Итог: {endings}")
                print(f"Ходы: {moves_avg:.2f}")
                print(f"Policy_losses: {avg_p_loss:.4f}")
                print(f"Value_losses: {avg_v_loss:.4f}")
                print(f"Entropy: {avg_entr:.4f}")
                

                start_time = current_time

            if total_games % SAVE_EVERY == 0:
                player.save(f"models/m_{total_games}.pt")
                print(f"модели сохранены ", f"на партии {total_games}")

    df = pd.DataFrame({"results": results, 
                       "moves": moves_stats, 
                       "policy_loss": policy_loss,
                       "value_loss": value_loss,
                       "entropy": entropy})
    df.to_csv("stats.csv", index=False)


if __name__ == "__main__":
    main()
from engine import Engine
from models import PlayerModel, ChessModel
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from multiprocessing import Pool, set_start_method
import pandas as pd
import time
import os
from numpy import mean



MAX_GAMES = 200_000
SAVE_EVERY = 5000
PRINT_EVERY = 400
N_ENVS = 8
ENTROPY_COEF = 0.05
VALUE_COEF = 0.3
MAX_MOVES = 100
REPEAT_PENALTY = 0.05
CHECKPOINT_PATH = "pretrain_models/pretrain_model10"




# Инициализация каждого воркера в пуле.
def init_worker():
    global game, model, device
    device = torch.device("cuda")
    model = ChessModel(n_blocks=8, channels=128).to(device)
    model.eval()
    game = Engine()


# Конвертирует плоский список board в тензор [1, 12, 8, 8]
def board_to_tensor(board, player):

    if player == -1:
        # Для черных переворачиваем доску, чтобы всегда "играть за белых"
        board = board[::-1]

    board_tensor = torch.tensor(board, dtype=torch.float32).view(8, 8)
    tensor = torch.zeros(12, 8, 8, dtype=torch.float32)

    # Свои фигуры (каналы 0-5)
    for piece in range(1, 7):
        target_value = piece * player
        tensor[piece - 1] = (board_tensor == target_value).float()

    # Вражеские фигуры (каналы 6-11)
    for piece in range(1, 7):
        target_value = piece * player * -1
        tensor[piece + 5] = (board_tensor == target_value).float()

    return tensor.unsqueeze(0)  # [1, 12, 8, 8]


# Температура убывает от 2.0 до 0.5 по мере партии
def get_temperature(move_number):
    progress = move_number / 100
    return max(0.5, 2.0 - 1.5 * progress)


# Играет одну партию self-play и возвращает траекторию
def play_game_worker(state_dict):
    global model, game, device

    model.load_state_dict(state_dict)
    model.eval()
    game.__init__()

    # Траектория партии
    states = [] # Состояния доски (тензоры)
    legal_actions_history = [] # Доступные ходы на каждом шаге
    action_indices = [] # Индексы выбранных ходов в legal_actions
    actions = [] # Закодированные ходы (для штрафа повторов)
    log_probs = [] # Логарифмы вероятностей (для policy gradient)
    values = [] # Предсказанные value (для advantage)
    rewards = [] # Промежуточные награды (взятия, шахи)
    turns = [] # Список сторон, которые делали ходы

    done = False
    move_number = 0
                    
    while not done:
        legal_actions, board, turn = game.return_game()

        # Конвертируем доску в тензор
        x = board_to_tensor(board, turn).to(device)

        with torch.no_grad():
            logits, value_pred = model(x)

        logits = logits[0]           # [4672]
        value_pred = value_pred[0, 0].item()  # скаляр
        values.append(value_pred)

        # Берем только легальные ходы
        legal_logits = logits[legal_actions]

        # Softmax с температурой
        probs = F.softmax(legal_logits / 1.5, dim=0)

        # Сэмплируем ход
        dist = Categorical(probs)
        action_idx = dist.sample().item()
        action = legal_actions[action_idx]

        # Сохраняем для обучения
        states.append(list(game.board.values()))
        legal_actions_history.append(legal_actions)
        action_indices.append(action_idx)
        actions.append(action)
        turns.append(game.turn)

        # Лог-вероятность выбранного действия (нужна для policy gradient)
        log_prob = torch.log(probs[action_idx] + 1e-10)
        log_probs.append(log_prob)

        # Делаем ход
        game.move(action)
        move_number += 1

        # Награда: базовая + штраф за повтор
        base_reward = game.reward
        repeat_penalty = sum(1 for a in actions[:-1] if a == action) * REPEAT_PENALTY
        reward = base_reward - repeat_penalty
        rewards.append(reward)

        # Условия окончания партии
        ending = game.ending
        if move_number >= MAX_MOVES or ending != 0:
            done = True

    # ============ РАСЧЕТ RETURNS ============
    returns = []
    if ending in (1, -1):
        G = 0.99
        for i, r in enumerate(rewards):
            rew = G * ending * (-1)**i + r
            returns.append(rew)
            G *= 0.99
    else:
        returns = [-0.5 + r for r in rewards]

    return {
    "states": states,
    "action_indices": action_indices,
    "legal_actions_history": legal_actions_history,
    "log_probs": log_probs,
    "values": values,
    "returns": returns,
    "result": ending,
    "moves": move_number,
    "rewards": rewards
    }


# Вычисляет advantage: A = G - V(s)
def compute_advantages(returns, values):
    returns_t = torch.tensor(returns, dtype=torch.float32)
    values_t = torch.tensor(values, dtype=torch.float32)
    advantages = returns_t - values_t

    # Нормализуем advantage для стабильности
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return returns_t, advantages


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("stats", exist_ok=True)
    
    if CHECKPOINT_PATH:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cuda")
        player = PlayerModel(lr=3e-4, n_blocks=8, channels=128, checkpoint = checkpoint)
    else:
        player = PlayerModel(lr=3e-4, n_blocks=8, channels=128)
        
    # Статистика
    results_history = []
    stats = []
    moves_stats = []
    policy_losses = []
    value_losses = []
    entropies = []

    total_games = 0
    total_time = 0
    start_time = time.time()

    with Pool(processes=N_ENVS, initializer=init_worker) as pool:
        while total_games < MAX_GAMES:
            
            # Генерируем батч из N_ENVS партий
            args = [player.model.state_dict() for _ in range(N_ENVS)]
            batch = pool.map(play_game_worker, args)

            player.model.train()
            player.opt.zero_grad()

            total_policy_loss = 0.0
            total_value_loss = 0.0
            total_entropy = 0.0

            for episode in batch:
                states_list = episode["states"]
                moves = episode["moves"]
                result = episode["result"]

                results_history.append(result)
                moves_stats.append(moves)
                
                states = [board_to_tensor(s, (-1)**i) for i, s in enumerate(states_list)]

                # Собираем states в батч
                x = torch.cat(states, dim=0).to(player.device)  # [moves, 12, 8, 8]

                # Forward pass
                logits, values_pred = player.model(x)
                values_pred = values_pred.squeeze(-1)  # [moves]

                # Рассчитываем returns и advantages
                returns_t, advantages = compute_advantages(
                    episode["returns"], episode["values"]
                )
                returns_t = returns_t.to(player.device)
                advantages = advantages.to(player.device)

                episode_policy_loss = 0.0
                episode_entropy = 0.0

                for t in range(moves):
                    step_logits = logits[t]  # [4672]
                    legal_actions = episode["legal_actions_history"][t]

                    # Маскируем только легальные ходы
                    legal_logits = step_logits[legal_actions]
                    legal_probs = F.softmax(legal_logits, dim=0)

                    # Энтропия: -sum(p * log(p))
                    log_legal_probs = torch.log(legal_probs + 1e-10)
                    entropy = -torch.sum(legal_probs * log_legal_probs)
                    episode_entropy += entropy

                    # Policy loss: -advantage * log_prob
                    action_idx = episode["action_indices"][t]
                    log_prob = log_legal_probs[action_idx]
                    advantage = advantages[t]
                    episode_policy_loss -= advantage * log_prob

                # Value loss: Huber между предсказанным и фактическим return
                episode_value_loss = F.smooth_l1_loss(values_pred, returns_t)

                # Нормализуем на длину партии
                episode_policy_loss = episode_policy_loss / moves
                episode_value_loss = episode_value_loss
                episode_entropy = episode_entropy / moves

                # Полный loss для эпизода
                loss = (episode_policy_loss + VALUE_COEF * episode_value_loss - ENTROPY_COEF * episode_entropy)
                loss = loss / N_ENVS  # Делим на количество параллельных игр
                loss.backward()

                total_policy_loss += episode_policy_loss.item()
                total_value_loss += episode_value_loss.item()
                total_entropy += episode_entropy.item()

                policy_losses.append(episode_policy_loss.item())
                value_losses.append(episode_value_loss.item())
                entropies.append(episode_entropy.item())

            # Обрезаем градиенты и делаем шаг
            torch.nn.utils.clip_grad_norm_(player.model.parameters(), max_norm=1.0)
            player.opt.step()

            total_games += N_ENVS


            # вывод метрик в консоль и их сохранение в память
            if total_games % PRINT_EVERY == 0:
                # Считаем статистику окончаний
                endings = [len([r for r in results_history if r == i]) for i in (0, 1, -1, 2, 3)]

                avg_moves = mean(moves_stats)
                avg_p_loss = mean(policy_losses)
                avg_v_loss = mean(value_losses)
                avg_ent = mean(entropies)

                current_time = time.time()
                elapsed = current_time - start_time
                total_time += elapsed

                stats.append({
                    "games": total_games,
                    "time": f"{elapsed:.2f}",
                    "endings": endings,
                    "moves": f"{avg_moves:.2f}",
                    "policy_loss": f"{avg_p_loss:.4f}",
                    "value_loss": f"{avg_v_loss:.4f}",
                    "entropy": f"{avg_ent:.4f}"
                })

                print(  f"{total_games:10d} | ",
                        f"Time: {(total_time//60):.0f}:{(total_time%60):.0f} | ",
                        f"Endings: {endings} | ",
                        f"Avg moves: {avg_moves:.2f} | "
                        f"Policy loss: {avg_p_loss:.4f} | "
                        f"Value loss: {avg_v_loss:.4f} | "
                        f"Entropy: {avg_ent:.4f}")

                results_history = []
                moves_stats = []
                policy_losses = []
                value_losses = []
                entropies = []
                start_time = current_time

            if total_games % SAVE_EVERY == 0:
                player.save(f"train_models/m_{total_games}.pt")
                df = pd.DataFrame(stats)
                df.to_csv(f"train_stats/stats_{total_games}.csv", index=False)

    print("\n=== Обучение завершено ===")


if __name__ == "__main__":
    set_start_method("spawn", force=True)
    main()
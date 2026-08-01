import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from ast import literal_eval
from models import ChessModel
from time import time
import gc


def board_to_tensor(board, player):
    if player == -1:
        board = board[::-1]
    
    board_tensor = torch.tensor(board, dtype=torch.float32).view(8, 8)
    tensor = torch.zeros(12, 8, 8, dtype=torch.float32)
    
    for piece in range(1, 7):
        target_value = piece * player
        tensor[piece - 1] = (board_tensor == target_value).float()
    
    for piece in range(1, 7):
        target_value = piece * player * -1
        tensor[piece + 5] = (board_tensor == target_value).float()
    
    return tensor


class CustomDataset(Dataset):
    def __init__(self, path):
        self.df = pd.read_csv(path)
        print(f"Датасет: {len(self.df)} записей")
        print(f"Policy target range: [{self.df['policy_target'].min()}, {self.df['policy_target'].max()}]")
        print(f"Value distribution:\n{self.df['value_target'].value_counts()}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        return {
            "value_target": float(row['value_target']),
            "policy_target": int(row['policy_target']),
            "board": board_to_tensor(literal_eval(row['board']), row['turn'])
        }\
            

def main():
    DATA_PATH = 'dataset.csv'
    EPOCHS = 10
    IS_REPEAT = False
    all_stats = []
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    dataset = CustomDataset(DATA_PATH)
    epoch_len = dataset.__len__() / 256
    dataloader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    model = ChessModel(n_blocks=8, channels=128).to(device)
    
    if IS_REPEAT:
        # Загружаем существующую модель
        checkpoint_path = 'pretrain_models/pretrain_model10'  # или укажи нужный путь
        print(f"\nЗагрузка чекпоинта: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Загружена модель, эпоха: {checkpoint.get('epoch', 'unknown')}")
        
        # Оптимизатор с меньшим lr для дообучения
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
    else:
        # Создаём новую модель с нуля
        print("\nИнициализация новой модели")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    
    for epoch in range(1, EPOCHS + 1):
        print(f"\n{'='*50}")
        print(f"Эпоха {epoch}/{EPOCHS} | LR: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"{'='*50}")
        
        start = time()
        
        model.train()
        total_policy_loss = 0
        total_value_loss = 0
        total_samples = 0
        total_time = 0
        start_time = time()
        correct_predictions = 0
        stats = []
        
        for i, batch in enumerate(dataloader):
            boards = batch['board'].to(device)
            policy_targets = batch['policy_target'].to(device).long()
            value_targets = batch['value_target'].to(device).float()
            
            optimizer.zero_grad()
            policy_logits, values = model(boards)
            
            policy_loss = F.cross_entropy(policy_logits, policy_targets, label_smoothing=0.1)
            
            value_loss = F.mse_loss(values.squeeze(), value_targets)
            
            loss = policy_loss + value_loss*2.0
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            pred_moves = policy_logits.argmax(dim=1)
            correct_predictions += (pred_moves == policy_targets).sum().item()
            
            batch_size = boards.size(0)
            total_policy_loss += policy_loss.item() * batch_size
            total_value_loss += value_loss.item() * batch_size
            total_samples += batch_size
            
            if i % 1000:
                acc = correct_predictions / total_samples if total_samples > 0 else 0
                stats.append({
                    "ploss": f"{policy_loss.item():.3f}",
                    "vloss": f"{value_loss.item():.3f}",
                    "accuracy": f"{acc:.4f}",
                    "value": f"{values.mean().item():.3f}"
                })
            
            if i % 1000 == 0:
                current_time = time()
                elapsed = current_time-start_time
                total_time += elapsed
                start_time = current_time
                acc = correct_predictions / total_samples if total_samples > 0 else 0
                print(f"  Progress: {i}/{epoch_len:.0f} | Time: {(total_time//60):.0f}:{(total_time%60):.0f}"
                    f" | P: {policy_loss.item():.3f} | V: {value_loss.item():.3f} | "
                    f"Acc: {acc:.4f} | Value: {values.mean().item():.3f}")
        
        p_loss = total_policy_loss / total_samples
        v_loss = total_value_loss / total_samples
        
        all_stats.append(stats)
        del stats
        gc.collect()
        
        
        elapsed = time() - start
        
        print(f"\nИтог эпохи {epoch} ({elapsed:.0f}с):")
        print(f"  Policy Loss: {p_loss:.4f} | Value Loss: {v_loss:.4f} | Accuracy: {acc:.4f}")
        
        scheduler.step()
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'policy_loss': p_loss,
            'value_loss': v_loss,
        }, f"pretrain_models/pretrain_model{epoch}")
        
    stats_df = pd.DataFrame({"stats": all_stats})
    stats_df.to_csv("pretrain_stats/stats.csv")
    
    print("\nОбучение завершено!")


if __name__ == '__main__':
    main()
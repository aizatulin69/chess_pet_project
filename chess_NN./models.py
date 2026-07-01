import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x += residual
        x = F.relu(x)
        return x


class ChessModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.start = nn.Conv2d(12, 64, 3, padding=1, bias=False)
        self.bn_start = nn.BatchNorm2d(64)
        self.res_blocks = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64)
        )

        self.policy_conv = nn.Conv2d(64, 32, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 4672)
        )

        self.value_conv = nn.Conv2d(64, 32, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = F.relu(self.bn_start(self.start(x)))
        features = self.res_blocks(x)
        p_feats = F.relu(self.policy_bn(self.policy_conv(features)))
        policy = self.policy_head(p_feats)
        v_feats = F.relu(self.value_bn(self.value_conv(features)))
        value = self.value_head(v_feats)
        return policy, value



# Общий базовый класс
class PlayerModel():
    
    # определяется в подклассах
    TURN_FLAG = None   # 1 для белых, -1 для чёрных
    C_VALUES = None   # индекс типа фигуры -> код фигуры в chess_lib

    def __init__(self, lr: float = 3e-4):
        self.model = ChessModel()
        self.opt   = optim.Adam(self.model.parameters(), lr=lr)

    # сохранение / загрузка
    def save(self, path: str):
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.opt.state_dict(),
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location="cpu")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.opt.load_state_dict(checkpoint["optimizer_state_dict"])
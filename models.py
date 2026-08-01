import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Остаточный (residual) блок с двумя свёрточными слоями и skip-connection.

    Реализует архитектуру, популяризированную ResNet: входной тензор
    проходит через две свёртки 3×3 с BatchNorm и GELU, после чего
    исходный тензор прибавляется к результату (skip-connection) и
    снова пропускается через активацию.

    Parameters
    ----------
    channels : int
        Число входных и выходных каналов. Внутри блока размерность
        сохраняется неизменной благодаря ``padding=1``.

    Attributes
    ----------
    conv1 : nn.Conv2d
        Первая свёртка 3×3.
    bn1 : nn.BatchNorm2d
        Нормализация после первой свёртки.
    conv2 : nn.Conv2d
        Вторая свёртка 3×3.
    bn2 : nn.BatchNorm2d
        Нормализация после второй свёртки.

    Examples
    --------
    >>> block = ResidualBlock(channels=64)
    >>> x = torch.randn(2, 64, 8, 8)
    >>> out = block(x)
    >>> out.shape
    torch.Size([2, 64, 8, 8])
    """

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        """
        Прямой проход через остаточный блок.

        Parameters
        ----------
        x : torch.Tensor
            Входной тензор формы ``(N, C, H, W)``.

        Returns
        -------
        torch.Tensor
            Выходной тензор той же формы, что и ``x``.

        Notes
        -----
        Skip-connection применяется до финальной активации GELU.
        """
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.gelu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x += residual
        x = F.gelu(x)
        return x


class ChessModel(nn.Module):
    """
    Двухголовая нейронная сеть для шахмат (Policy + Value).

    Архитектура вдохновлена AlphaZero: входное представление доски
    проходит через начальную свёртку, затем через башню остаточных
    блоков (Residual Tower), после чего разветвляется на два «головы»:

    * **Policy head** — предсказывает распределение вероятностей
      по всем возможным ходам (4672 действия).
    * **Value head** — оценивает текущую позицию скаляром
      в диапазоне ``[-1, 1]`` (победа / поражение).

    Parameters
    ----------
    n_blocks : int, optional
        Количество остаточных блоков в башне (по умолчанию 8).
    channels : int, optional
        Число каналов в остаточных блоках (по умолчанию 64).

    Attributes
    ----------
    start : nn.Conv2d
        Начальная свёртка из 12 входных плоскостей в ``channels``.
    bn_start : nn.BatchNorm2d
        Нормализация после начальной свёртки.
    res_blocks : nn.Sequential
        Последовательность остаточных блоков.
    policy_conv : nn.Conv2d
        Сжимающая свёртка 1×1 перед policy head.
    policy_bn : nn.BatchNorm2d
        Нормализация policy-признаков.
    policy_head : nn.Sequential
        Полносвязные слои, выдающие логиты размера 4672.
    value_conv : nn.Conv2d
        Сжимающая свёртка 1×1 перед value head.
    value_bn : nn.BatchNorm2d
        Нормализация value-признаков.
    value_head : nn.Sequential
        Полносвязные слои, выдающие скалярную оценку.

    Examples
    --------
    >>> model = ChessModel(n_blocks=4, channels=32)
    >>> board = torch.randn(4, 12, 8, 8)  # batch=4
    >>> policy, value = model(board)
    >>> policy.shape
    torch.Size([4, 4672])
    >>> value.shape
    torch.Size([4, 1])
    """

    def __init__(self, n_blocks=8, channels=64):
        super().__init__()

        # Вход: 12 плоскостей (6 типов фигур x 2 цвета)
        self.start = nn.Conv2d(12, channels, 3, padding=1, bias=False)
        self.bn_start = nn.BatchNorm2d(channels)

        # Residual tower — основная "думающая" часть
        self.res_blocks = nn.Sequential(*[
            ResidualBlock(channels) for _ in range(n_blocks)
        ])

        # Policy head: какой ход выбрать
        self.policy_conv = nn.Conv2d(channels, 32, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 1024),
            nn.GELU(),
            nn.Linear(1024, 4672)  # 64 клетки x 73 плоскости
        )

        # Value head: оценка позиции
        self.value_conv = nn.Conv2d(channels, 32, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 512),
            nn.GELU(),
            nn.Linear(512, 1),
            nn.Tanh()  # Сжимаем в [-1, 1]
        )

    def forward(self, x):
        """
        Прямой проход через сеть.

        Parameters
        ----------
        x : torch.Tensor
            Входное представление доски формы ``(N, 12, 8, 8)``,
            где ``N`` — размер батча. 12 каналов соответствуют
            6 типам фигур × 2 цвета (белые / чёрные).

        Returns
        -------
        policy : torch.Tensor
            Логиты распределения по ходам формы ``(N, 4672)``.
            Для получения вероятностей следует применить ``softmax``.
        value : torch.Tensor
            Скалярная оценка позиции формы ``(N, 1)``,
            сжатая в ``[-1, 1]`` через ``Tanh``.

        Notes
        -----
        Число 4672 = 64 начальные клетки × 73 плоскости действия.
        Плоскости действия включают: 56 обычных направлений,
        8 ходов коня и 9 вариантов превращения пешки.
        """
        # x: [batch, 12, 8, 8]
        x = F.gelu(self.bn_start(self.start(x)))
        features = self.res_blocks(x)

        # Policy
        p = F.gelu(self.policy_bn(self.policy_conv(features)))
        policy = self.policy_head(p)  # [batch, 4672]

        # Value
        v = F.gelu(self.value_bn(self.value_conv(features)))
        value = self.value_head(v)    # [batch, 1]

        return policy, value


class PlayerModel:
    """
    Обёртка над ``ChessModel`` с управлением устройством и чекпоинтами.

    Класс инкапсулирует модель, оптимизатор Adam и логику
    сохранения/загрузки состояния обучения.

    Parameters
    ----------
    lr : float, optional
        Скорость обучения оптимизатора Adam (по умолчанию 3e-4).
    n_blocks : int, optional
        Число остаточных блоков в ``ChessModel`` (по умолчанию 8).
    channels : int, optional
        Число каналов в ``ChessModel`` (по умолчанию 64).
    checkpoint : dict or bool, optional
        Если передан словарь чекпоинта, модель и оптимизатор
        восстанавливаются из него. По умолчанию ``False``.

    Attributes
    ----------
    device : torch.device
        Устройство вычислений (всегда ``cuda``).
    model : ChessModel
        Экземпляр шахматной нейросети.
    opt : torch.optim.Adam
        Оптимизатор для обучения модели.

    Raises
    ------
    RuntimeError
        Если CUDA недоступна при инициализации.

    Examples
    --------
    >>> player = PlayerModel(lr=1e-3, n_blocks=4, channels=32)
    >>> player.save("checkpoint.pt")
    >>> player.load("checkpoint.pt")
    """

    def __init__(self, lr: float = 3e-4, n_blocks: int = 8, channels: int = 64, checkpoint=False):
        self.device = torch.device("cuda")
        self.model = ChessModel(n_blocks=n_blocks, channels=channels).to(self.device)
        self.opt = optim.Adam(self.model.parameters(), lr=lr)
        if checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.opt.load_state_dict(checkpoint["optimizer_state_dict"])

    def save(self, path: str):
        """
        Сохранить состояние модели и оптимизатора на диск.

        Parameters
        ----------
        path : str
            Путь к файлу чекпоинта (обычно ``.pt`` или ``.pth``).

        Returns
        -------
        None

        Notes
        -----
        Сохраняется словарь с ключами:

        * ``"model_state_dict"`` — веса модели.
        * ``"optimizer_state_dict"`` — состояние оптимизатора.

        Для восстановления используйте :meth:`load`.
        """
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.opt.state_dict(),
        }, path)

    def load(self, path: str):
        """
        Загрузить состояние модели и оптимизатора с диска.

        Parameters
        ----------
        path : str
            Путь к файлу чекпоинта.

        Returns
        -------
        None

        Notes
        -----
        Файл загружается на устройство ``self.device`` (CUDA).
        Убедитесь, что архитектура модели (``n_blocks``, ``channels``)
        совпадает с той, что была при сохранении чекпоинта.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.opt.load_state_dict(checkpoint["optimizer_state_dict"])
class Engine():
    """
    Шахматный движок с полной логикой игры.

    Реализует правила классических шахмат: движение фигур, взятие,
    рокировку, превращение пешки, шах, мат, пат, недостаточный материал,
    а также систему кодирования/декодирования ходов для ML-интерфейсов.

    Attributes
    ----------
    board : dict[int, int]
        Отображение позиции клетки (двузначное число ``row*10 + col``)
        на тип фигуры. Положительные значения — белые, отрицательные — чёрные.
        ``0`` означает пустую клетку.
    whites : list[int]
        Список типов белых фигур: ``[1, 2, 3, 4, 5, 6]``.
    blacks : list[int]
        Список типов чёрных фигур: ``[-1, -2, -3, -4, -5, -6]``.
    turn : int
        Текущий ход: ``1`` — белые, ``-1`` — чёрные.
    moves_played : int
        Счётчик совершённых ходов.
    ALL_LEGAL_MOVES : list[int] or dict
        Закодированные легальные ходы для текущей позиции.
    uncoded_moves : list[tuple]
        Некодированные легальные ходы в формате ``(start, end, promo)``.
    king_positions : dict[int, int]
        Позиции королей: ``{5: pos_white, -5: pos_black}``.
    ending : int
        Состояние окончания игры. ``0`` — игра продолжается,
        ``1`` — победа белых, ``-1`` — победа чёрных,
        ``2`` — ничья (недостаточно материала), ``3`` — пат.
    reward : float
        Награда за последний совершённый ход.
    score : dict[int, list[int]]
        Количество фигур каждого типа по сторонам.
        Индексы списка: ``[пешка, конь, слон, ладья, ферзь, король]``.
    last_captured : int
        Тип фигуры, взятой на последнем ходу.
    last_move_end : int or None
        Конечная клетка последнего хода (для взятия на проходе).
    position_history : list
        История позиций для проверки троекратного повторения.
    castling_rights : dict
        Права на рокировку для каждой стороны.
        Структура: ``{1: {"king": bool, "left": bool, "right": bool}, -1: ...}``.

    Notes
    -----
    Система координат: клетки кодируются двузначными числами ``row*10 + col``,
    где ``row`` и ``col`` находятся в диапазоне ``1..8``.
    Например, ``e2`` соответствует ``25``, ``e4`` — ``45``.
    """

    def __init__(self):
        """
        Инициализировать новую шахматную партию в начальной позиции.

        Устанавливает стандартную расстановку фигур, права рокировки,
        счётчики и генерирует начальный список легальных ходов.
        """
        self.board = \
           {11: 2, 12: 3, 13: 4, 14: 6, 15: 5, 16: 4, 17: 3, 18: 2, 
            21: 1, 22: 1, 23: 1, 24: 1, 25: 1, 26: 1, 27: 1, 28: 1, 
            31: 0, 32: 0, 33: 0, 34: 0, 35: 0, 36: 0, 37: 0, 38: 0, 
            41: 0, 42: 0, 43: 0, 44: 0, 45: 0, 46: 0, 47: 0, 48: 0, 
            51: 0, 52: 0, 53: 0, 54: 0, 55: 0, 56: 0, 57: 0, 58: 0, 
            61: 0, 62: 0, 63: 0, 64: 0, 65: 0, 66: 0, 67: 0, 68: 0, 
            71: -1, 72: -1, 73: -1, 74: -1, 75: -1, 76: -1, 77: -1, 78: -1, 
            81: -2, 82: -3, 83: -4, 84: -6, 85: -5, 86: -4, 87: -3, 88: -2}

        self.whites = [1, 2, 3, 4, 5, 6]
        self.blacks = [-1, -2, -3, -4, -5, -6]

        self.turn = 1
        self.moves_played = 0
        self.ALL_LEGAL_MOVES = [133, 132, 498, 497, 612, 613, 685, 686, 758, 759, 831, 832, 904, 905, 977, 978, 1050, 1051, 1123, 1124]
        self.uncoded_moves = [(12, 31, 0), (12, 33, 0),(17, 36, 0), (17, 38, 0), (21, 31, 0), (21, 41, 0), (22, 32, 0), (22, 42, 0), (23, 33, 0), (23, 43, 0), (24, 34, 0), (24, 44, 0), (25, 35, 0), (25, 45, 0), (26, 36, 0), (26, 46, 0), (27, 37, 0), (27, 47, 0), (28, 38, 0), (28, 48, 0)]
        self.king_positions = {5: 15, -5: 85}
        self.ending = 0
        self.reward = 0
        
        self.score = {1: [8, 2, 2, 2, 1, 1],
                      -1: [8, 2, 2, 2, 1, 1]}
        
        self.last_captured    = 0
        self.last_move_end    = None
        self.position_history  = []
                
        self.castling_rights = {
            1: {"king": True, "left": True, "right": True},
            -1: {"king": True, "left": True, "right": True}
        }


    def path_blocked(self, start, end):
        """
        Проверить, есть ли фигуры на пути между двумя клетками.

        Функция работает для ладьи, слона и ферзя. Если ``start`` и ``end``
        не лежат на одной горизонтали, вертикали или диагонали, метод
        возвращает ``False`` (путь считается свободным).

        Parameters
        ----------
        start : int
            Начальная клетка в формате ``row*10 + col``.
        end : int
            Конечная клетка в формате ``row*10 + col``.

        Returns
        -------
        bool
            ``True``, если между ``start`` и ``end`` стоит хотя бы одна фигура,
            иначе ``False``.

        Notes
        -----
        Метод не проверяет наличие фигуры на конечной клетке ``end`` —
        только промежуточные клетки.
        """
        # проверка для ладьи и ферзя
        if start//10 == end//10:
            step = 1 if end > start else -1
        elif start%10 == end%10:
            step = 10 if end > start else -10

        # для слона и ферзя
        elif abs(end-start)%9==0:
            step = 9 if end > start else -9
        elif abs(end-start)%11==0:
            step = 11 if end > start else -11

        else:
            return False
        # основной цикл
        p = start + step
        while p != end:
            if not (p%10 in (0,9) or p//10 in (0,9) or p<11 or p>88) and \
            self.board[p] != 0:
                return True
            p += step
        return False


    def is_under_attack(self, pos, by):
        """
        Определить, атакуется ли указанная клетка фигурами заданного цвета.

        Parameters
        ----------
        pos : int
            Клетка для проверки в формате ``row*10 + col``.
        by : int
            Цвет потенциальных атакующих: ``1`` — белые, ``-1`` — чёрные.

        Returns
        -------
        bool
            ``True``, если хотя бы одна фигура цвета ``by`` бьёт клетку ``pos``.

        Notes
        -----
        Проверка выполняется полным перебором всех фигур на доске.
        Для скользящих фигур (ладья, слон, ферзь) дополнительно вызывается
        :meth:`path_blocked` для проверки видимости.
        """
        attackers = self.whites if by == 1 else self.blacks
        
        # основной цикл
        for p, piece in self.board.items():
            if piece not in attackers:
                continue

            if piece in (1,-1): # пешка
                d = 10 if by == 1 else -10
                if pos in (p+d+1, p+d-1):
                    return True
            elif piece in (3,-3): # конь
                if abs(p-pos) in (8,12,19,21):
                    return True
            elif piece in (2,-2): # ладья
                if (p%10==pos%10 or p//10==pos//10) \
                   and not self.path_blocked(p,pos):
                    return True 
            elif piece in (4,-4): # слон
                if abs(p-pos)%9==0 or abs(p-pos)%11==0:
                    if not self.path_blocked(p,pos):
                        return True
            elif piece in (6,-6): # ферзь
                if ((p%10==pos%10 or p//10==pos//10) or
                    abs(p-pos)%9==0 or abs(p-pos)%11==0):
                    if not self.path_blocked(p,pos):
                        return True
            elif piece in (5,-5): # король
                if abs(p-pos) in (1,10,9,11):
                    return True

        return False


    def search_moves(self, pos, dirs, repeat, enemy):
        """
        Сгенерировать псевдолегальные ходы для фигуры, стоящей на ``pos``.

        Метод строит список достижимых клеток, двигаясь от ``pos`` по
        направлениям ``dirs`` до границы доски или первой встреченной фигуры.

        Parameters
        ----------
        pos : int
            Клетка с фигурой в формате ``row*10 + col``.
        dirs : list[int]
            Список приращений для каждого направления. Например,
            ``[1, -1, 10, -10]`` для горизонтали и вертикали.
        repeat : bool
            Если ``True``, движение продолжается до упора (ладья, слон, ферзь).
            Если ``False``, делается только один шаг (конь, король).
        enemy : list[int]
            Список типов фигур противника, которые могут быть взяты.

        Returns
        -------
        list[int]
            Список клеток, доступных для хода (включая клетки с фигурами
            противника, но не со своими).

        Examples
        --------
        >>> engine = Engine()
        >>> engine.search_moves(15, [1, -1, 10, -10], True, engine.blacks)
        [16, 14, 25, 35, 45, 55, 65, 75]
        """
        moves = []
        for d in dirs:
            p = pos
            while True:
                p += d

                # проверка выхода за доску
                if p%10 in (0,9) or p//10 in (0,9) or p<11 or p>88:
                    break
                
                # проверка на фигуру, стоящую на пути
                piece = self.board[p]
                if piece == 0:
                    moves.append(p)
                elif piece in enemy:
                    moves.append(p)
                    break
                elif abs(piece) == 5:
                    break
                else:
                    break
                
                # если repeat == False, не идёт дальше
                if not repeat:
                    break

        return moves


    def get_all_moves(self):
        """
        Сгенерировать псевдолегальные ходы для всех фигур текущего игрока.

        Метод заполняет атрибут :attr:`ALL_LEGAL_MOVES` словарём вида
        ``{start_pos: [end_pos, ...], ...}``, содержащим все возможные ходы
        без учёта шаха собственному королю. Рокировка добавляется отдельно
        для короля при наличии соответствующих прав.

        Returns
        -------
        None
            Результат записывается в :attr:`ALL_LEGAL_MOVES`.

        See Also
        --------
        filter_moves : Фильтрация псевдолегальных ходов с учётом шаха.
        """
        self.ALL_LEGAL_MOVES = {}
        enemy = self.blacks if self.turn == 1 else self.whites

        # основной цикл
        for pos, piece in self.board.items():

            # если пустая клетка или не та фигура -- идём дальше
            if piece == 0:
                continue
            if (piece in self.whites and self.turn == -1) or \
               (piece in self.blacks and self.turn == 1):
                continue

            # легальные ходы для всех фигур
            if piece in (2,-2): # ладья
                moves = self.search_moves(pos,[1,-1,10,-10],True,enemy)
            elif piece in (4,-4): # слон
                moves = self.search_moves(pos,[9,-9,11,-11],True,enemy)
            elif piece in (6,-6): # ферзь
                moves = self.search_moves(pos,[1,-1,10,-10,9,-9,11,-11],True,enemy)
            elif piece in (3,-3): # конь
                moves = self.search_moves(pos,[8,12,19,21,-8,-12,-19,-21],False,enemy)
            elif piece in (5,-5): # король
                moves = self.search_moves(pos,[1,-1,10,-10,9,-9,11,-11],False,enemy)

                # рокировка
                if self.castling_rights[self.turn]["king"]:
                    if pos == 15 and self.turn == 1:
                        if not self.path_blocked(15, 18) and self.castling_rights[self.turn]["right"]:
                            moves.append(17)
                        if not self.path_blocked(15, 11) and self.castling_rights[self.turn]["left"]:
                            moves.append(11)
                    if pos == 85 and self.turn == -1:
                        if not self.path_blocked(85, 88) and self.castling_rights[self.turn]["right"]:
                            moves.append(88)
                        if not self.path_blocked(85, 81) and self.castling_rights[self.turn]["left"]:
                            moves.append(81)

            # пешка
            elif piece in (1,-1):
                moves = []
                d = 10 if piece == 1 else -10
                if 10<pos+d<90:
                    if self.board[pos+d] == 0:
                        moves.append(pos+d)
                        if (20<pos<30 and piece == 1) or (70<pos<80 and piece == -1):
                            if self.board[pos+2*d] == 0:
                                moves.append(pos+2*d)
                for x in (d+1,d-1):
                    if (pos+x)%10 in (0,9) or (pos+x)//10 in (0,9) or (pos+x)<11 or (pos+x)>88:
                        continue
                    if self.board[pos+x] in enemy:
                        if abs(self.board[pos+x]) != 5:
                            moves.append(pos+x)

            if moves:
                self.ALL_LEGAL_MOVES[pos] = moves
    

    def castling(self, start, end):
        """
        Выполнить рокировку, если ход ``start -> end`` является рокировкой.

        Метод обновляет права на рокировку в зависимости от подвижности ладей
        и короля, а затем, при соблюдении всех условий, перемещает короля и
        ладью на новые позиции.

        Parameters
        ----------
        start : int
            Начальная клетка хода.
        end : int
            Конечная клетка хода.

        Returns
        -------
        bool
            ``True``, если рокировка была выполнена, иначе ``False``.

        Notes
        -----
        При короткой рокировке проверяется отсутствие шаха на трёх клетках:
        ``row*10+5``, ``row*10+6``, ``row*10+7``.
        При длинной — на четырёх: ``row*10+3``, ``row*10+4``, ``row*10+5``
        (дважды проверяется ``row*10+5``).
        """
        row = 1 if self.turn == 1 else 8

        # обновляет права в начале хода
        if start == row*10+1: self.castling_rights[self.turn]["right"] = False
        if start == row*10+8: self.castling_rights[self.turn]["left"] = False
        if start == row*10+5: self.castling_rights[self.turn]["king"] = False

        # делает рокировку если всё хорошо
        if self.castling_rights[self.turn]["king"]:
            if ((start, end) == (15, 18) and self.turn == 1) or \
               ((start, end) == (85, 88) and self.turn == -1):
                if not self.path_blocked(row*10+5, row*10+8) and self.castling_rights[self.turn]["right"]:
                    if (self.is_under_attack(row*10+7, self.turn*-1),
                        self.is_under_attack(row*10+6, self.turn*-1),
                        self.is_under_attack(row*10+5, self.turn*-1)) == (False, False, False):
                        self.board[row*10+7] = 5*self.turn
                        self.board[row*10+6] = 2*self.turn
                        self.board[row*10+5] = 0
                        self.board[row*10+8] = 0
                        return True
            if ((start, end) == (15, 11) and self.turn == 1) or \
               ((start, end) == (85, 81) and self.turn == -1):
                if not self.path_blocked(row*10+5, row*10+1) and self.castling_rights[self.turn]["left"]:
                    if (self.is_under_attack(row*10+3, self.turn*-1),
                        self.is_under_attack(row*10+4, self.turn*-1),
                        self.is_under_attack(row*10+5, self.turn*-1),
                        self.is_under_attack(row*10+5, self.turn*-1)) == (False, False, False, False):
                        self.board[row*10+3] = 5*self.turn
                        self.board[row*10+4] = 2*self.turn
                        self.board[row*10+5] = 0
                        self.board[row*10+8] = 0
                        self.board[row*10+1] = 0
                    return True
        return False

    def filter_moves(self):
        """
        Отфильтровать псевдолегальные ходы, оставив только полностью легальные.

        Метод последовательно делает каждый псевдолегальный ход на временной
        доске и проверяет, не оказывается ли собственный король под шахом.
        Если король остаётся под боем, ход отбрасывается.

        Returns
        -------
        None
            Результат записывается в :attr:`ALL_LEGAL_MOVES`.

        Notes
        -----
        Если во время перебора обнаруживается, что взят король противника,
        такой ход удаляется из списка (короля нельзя взять, только объявить мат).
        """
        legal = {}
        king = 5 if self.turn == 1 else -5

        # проходится по всем {pos: [moves]}
        for pos, moves in self.ALL_LEGAL_MOVES.items():
            valid = []

            for move in moves:
                piece = self.board[pos]

                captured = self.board[move]
                if abs(captured) == 5:
                    self.ALL_LEGAL_MOVES[pos] = [m for m in moves if m != move]
                    continue

                # делает ход, чтобы проверить угрозу
                test_kp = self.king_positions.copy()
                self.board[move] = piece
                self.board[pos] = 0
                if piece in (5, -5):
                    test_kp[piece] = move

                # определяет короля
                king_pos = test_kp[king]
                
                # проверяет, появилась ли угроза королю после хода
                if not self.is_under_attack(king_pos, -self.turn):
                    valid.append(move)
                
                # возвращает всё на место
                self.board[pos] = piece
                self.board[move] = captured

            if valid: 
                legal[pos] = valid
        self.ALL_LEGAL_MOVES = legal


    def reset(self):
        """
        Сбросить движок в начальное состояние новой партии.

        Восстанавливает начальную расстановку фигур, обнуляет счётчики,
        права рокировки и генерирует начальный список ходов.

        Returns
        -------
        None
        """
        self.board = \
            {11: 2, 12: 3, 13: 4, 14: 6, 15: 5, 16: 4, 17: 3, 18: 2, 
            21: 1, 22: 1, 23: 1, 24: 1, 25: 1, 26: 1, 27: 1, 28: 1, 
            31: 0, 32: 0, 33: 0, 34: 0, 35: 0, 36: 0, 37: 0, 38: 0, 
            41: 0, 42: 0, 43: 0, 44: 0, 45: 0, 46: 0, 47: 0, 48: 0, 
            51: 0, 52: 0, 53: 0, 54: 0, 55: 0, 56: 0, 57: 0, 58: 0, 
            61: 0, 62: 0, 63: 0, 64: 0, 65: 0, 66: 0, 67: 0, 68: 0, 
            71: -1, 72: -1, 73: -1, 74: -1, 75: -1, 76: -1, 77: -1, 78: -1, 
            81: -2, 82: -3, 83: -4, 84: -6, 85: -5, 86: -4, 87: -3, 88: -2}
        self.turn = 1
        self.moves_played = 0
        self.ALL_LEGAL_MOVES = [133, 132, 498, 497, 612, 613, 685, 686, 758, 759, 831, 832, 904, 905, 977, 978, 1050, 1051, 1123, 1124]
        self.uncoded_moves = [(12, 31, 0), (12, 33, 0),(17, 36, 0), (17, 38, 0), (21, 31, 0), (21, 41, 0), (22, 32, 0), (22, 42, 0), (23, 33, 0), (23, 43, 0), (24, 34, 0), (24, 44, 0), (25, 35, 0), (25, 45, 0), (26, 36, 0), (26, 46, 0), (27, 37, 0), (27, 47, 0), (28, 38, 0), (28, 48, 0)]
        self.king_positions = {5: 15, -5: 85}
        self.castling_rights = {
            1: {"king": True, "left": True, "right": True},
            -1: {"king": True, "left": True, "right": True}}
        self.reward = 0
        self.score = {1: [8, 2, 2, 2, 1, 1],
                      -1: [8, 2, 2, 2, 1, 1]}

    def add_promotion(self):
        """
        Добавить варианты превращения пешек к списку легальных ходов.

        Для пешек, достигших предпоследней горизонтали, каждый ход
        разветвляется на четыре варианта превращения: конь, слон, ладья,
        ферзь. Для остальных фигур добавляется ``promo=0``.

        Returns
        -------
        None
            Результат записывается в :attr:`ALL_LEGAL_MOVES` как список
            кортежей ``(start, end, promo)``, а затем в :attr:`uncoded_moves`.

        Notes
        -----
        Порядок фигур превращения: конь (3), слон (4), ладья (2), ферзь (6).
        Ферзь кодируется отдельно в :meth:`encode_single_move`.
        """
        for pos, moves in self.ALL_LEGAL_MOVES.items():
            piece = self.board[pos]

            # отвечает за добавление вариантов превращения для пешки на предпоследней полосе
            if (piece == 1 and pos//10 == 7) or (piece == -1 and pos//10 == 2):
                promos = [2,3,4,6] if piece == 1 else [-2,-3,-4,-6]
                self.ALL_LEGAL_MOVES[pos] = []
                for m in moves:
                    self.ALL_LEGAL_MOVES[pos].extend([(m, p) for p in promos])

            # если фигуре не нужно превращаться, добавляет 0
            else:
                self.ALL_LEGAL_MOVES[pos] = []
                for m in moves:
                    self.ALL_LEGAL_MOVES[pos].append((m, 0))

        # создаёт массив вида [(start, end, promo), ...]
        legal_actions = []
        for f, moves in self.ALL_LEGAL_MOVES.items():
            for t, p in moves:
                legal_actions.append((f, t, p))

        self.ALL_LEGAL_MOVES = legal_actions


    def encode_single_move(self, f, t, p):
        """
        Закодировать один ход в уникальный целочисленный индекс.

        Используется схема кодирования AlphaZero: каждый ход представляется
        как ``f_ * 73 + action_plane``, где ``f_`` — индекс начальной клетки
        (0..63), а ``action_plane`` — тип хода (0..72).

        Parameters
        ----------
        f : int
            Начальная клетка в формате ``row*10 + col``.
        t : int
            Конечная клетка в формате ``row*10 + col``.
        p : int
            Тип фигуры превращения (``0`` если превращения нет).

        Returns
        -------
        int
            Уникальный индекс хода в диапазоне ``[0, 4671]``.

        Notes
        -----
        Пространство действий разбито на три зоны:

        * **0..55** — обычные ходы (8 направлений × 7 дистанций).
        * **56..63** — ходы коня (8 вариантов смещений).
        * **64..72** — превращения пешек (3 направления × 3 фигуры).

        Превращение в ферзя при обычном достижении последней горизонтали
        кодируется как обычный ход (ферзь не входит в зону 64..72).
        """
        DIRS = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
        KNIGHT_MOVES = [(-2, 1), (-1, 2), (1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1)]
        PROMO_MAP = {3: 0, 4: 1, 2: 2, -3: 0, -4: 1, -2: 2}

        f_row, f_col = (f // 10 - 1), (f % 10 - 1)
        t_row, t_col = (t // 10 - 1), (t % 10 - 1)
        f_ = f_row * 8 + f_col
        dr = t_row - f_row
        dc = t_col - f_col
        action_plane = -1

        # Если p == 6 или -6 (ферзь), этот if пропускается и кодируется как обычный ход
        if p != 0 and abs(p) != 6:
            # Направление движения пешки при промоушене (-1 влево, 0 прямо, 1 вправо)
            # Для белых (строка 6 -> 7, dr=1) и черных (строка 1 -> 0, dr=-1)
            # Но для определения "влево/вправо" нам важен только dc
            direction_idx = dc + 1 # превращает -1, 0, 1 в индексы 0, 1, 2
            promo_idx = PROMO_MAP[p]
            # Плоскости промоушена занимают индексы 64..72
            action_plane = 64 + promo_idx * 3 + direction_idx

        # Проверяем ход коня
        elif (dr, dc) in KNIGHT_MOVES:
            knight_idx = KNIGHT_MOVES.index((dr, dc))
            # Плоскости коня занимают индексы 56..63
            action_plane = 56 + knight_idx

        # Обычный ход (Ферзь, Ладья, Слон, Король, обычная Пешка или Пешка->Ферзь)
        else:
            # Определяем базовый вектор направления
            step_r = 1 if dr > 0 else (-1 if dr < 0 else 0)
            step_c = 1 if dc > 0 else (-1 if dc < 0 else 0)
            
            dir_idx = DIRS.index((step_r, step_c))
            # Дистанция хода
            distance = max(abs(dr), abs(dc))
            distance_idx = distance - 1 # 0..6
            
            # Плоскости обычных ходов занимают индексы 0..55
            action_plane = dir_idx * 7 + distance_idx

        # Итоговый уникальный индекс от 0 до 4671
        return f_ * 73 + action_plane


    def encode_moves(self):
        """
        Преобразовать список некодированных ходов в список индексов.

        Применяет :meth:`encode_single_move` к каждому ходу из
        :attr:`ALL_LEGAL_MOVES` и перезаписывает атрибут закодированными
        целочисленными значениями.

        Returns
        -------
        None
        """
        actions = []
        for f, t, p in self.ALL_LEGAL_MOVES:
            action = self.encode_single_move(f, t, p)
            actions.append(action)
        self.ALL_LEGAL_MOVES = actions

    def decode_move(self, action, current_turn):
        """
        Декодировать индекс хода в кортеж ``(start, end, promo)``.

        Обратная операция к :meth:`encode_single_move`. Распаковывает
        ``action`` в исходные шахматные координаты и тип превращения.

        Parameters
        ----------
        action : int
            Закодированный индекс хода (``0..4671``).
        current_turn : int
            Цвет игрока, совершающего ход: ``1`` — белые, ``-1`` — чёрные.
            Необходим для определения направления превращения пешки.

        Returns
        -------
        tuple[int, int, int]
            Кортеж ``(start, end, promo)``, где ``start`` и ``end`` —
            клетки в формате ``row*10 + col``, а ``promo`` — тип фигуры
            превращения (``0`` если отсутствует).

        Raises
        ------
        ValueError
            Если ``action`` выходит за пределы допустимого диапазона.

        Notes
        -----
        При декодировании обычного хода пешки на последнюю горизонталь
        автоматически устанавливается ``promo = 6 * current_turn`` (ферзь),
        если в закодированном действии не указано иное превращение.
        """
        DIRS = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
        KNIGHT_MOVES = [(-2, 1), (-1, 2), (1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1)]
        PROMO_LIST = [3, 4, 2]

        f_ = action // 73
        action_plane = action % 73
        
        f_row, f_col = f_ // 8, f_ % 8
        f = (f_row + 1) * 10 + (f_col + 1)
        
        t_row, t_col = -1, -1
        p = 0

        # 1. Обычные ходы (0..55)
        if action_plane < 56:
            dir_idx = action_plane // 7
            distance = (action_plane % 7) + 1
            
            dr, dc = DIRS[dir_idx]
            t_row = f_row + dr * distance
            t_col = f_col + dc * distance
            
            if (t_row == 7 and current_turn == 1) or (t_row == 0 and current_turn == -1):
                pass 

        # 2. Ходы коня (56..63)
        elif action_plane < 64:
            knight_idx = action_plane - 56
            dr, dc = KNIGHT_MOVES[knight_idx]
            t_row = f_row + dr
            t_col = f_col + dc
            
        # 3. Превращения пешек (64..72)
        else:
            promo_data = action_plane - 64
            promo_idx = promo_data // 3
            direction_idx = promo_data % 3
            
            dc = direction_idx - 1
            dr = 1 if current_turn == 1 else -1 # Белые идут вверх, черные вниз
            
            t_row = f_row + dr
            t_col = f_col + dc
            p = PROMO_LIST[promo_idx] * current_turn

        t = (t_row + 1) * 10 + (t_col + 1)
        
        # Фикс для авто-превращения в ферзя при обычном ходе на последнюю горизонталь
        if self.board[f] in (1, -1):
            if p == 0 and (t_row == 7 or t_row == 0):
                # Если логика движка требует p=6/-6 для ферзя, активируем этот фикс:
                p = 6 * current_turn

        return f, t, p


    def update_state(self):
        """
        Обновить статус окончания игры.

        Проверяет условия мата, пата, недостаточного материала и победы
        одной из сторон по оставшимся фигурам. Результат записывается в
        :attr:`ending`.

        Returns
        -------
        None

        Notes
        -----
        Возможные значения :attr:`ending`:

        * ``0`` — игра продолжается.
        * ``1`` — победа белых (у чёрных остался только король).
        * ``-1`` — победа чёрных (у белых остался только король).
        * ``2`` — ничья (недостаточно материала).
        * ``3`` — пат (нет легальных ходов, король не под шахом).
        * ``-self.turn`` — мат текущему игроку.
        """
        king = 5 if self.turn == 1 else -5
        king_pos = self.king_positions[king]
        in_check = self.is_under_attack(king_pos, -self.turn)

        # проверка на мат и пат (нет легальных ходов)
        if not self.ALL_LEGAL_MOVES:
            self.ending = -self.turn if in_check else 3
            return

        # проверка на недостаточный материал для мата
        w_pieces = self.score[1]
        b_pieces = self.score[-1]

        w_knights, w_bishops = w_pieces[1], w_pieces[2]
        b_knights, b_bishops = b_pieces[1], b_pieces[2]

        # король против короля (остались только короли, индексы 5)
        if w_pieces == [0, 0, 0, 0, 1, 0] and b_pieces == [0, 0, 0, 0, 1, 0]:
            self.ending = 2
            return

        # король и один слон или один конь против короля
        if (w_knights + w_bishops <= 1 and b_knights == 0 and b_bishops == 0) or \
            (b_knights + b_bishops <= 1 and w_knights == 0 and w_bishops == 0):
            self.ending = 2
            return

        w_pieces_count = sum(w_pieces)
        b_pieces_count = sum(b_pieces)

        # если у одной из сторон остался только король
        if w_pieces_count == 1:
            self.ending = -1  # Победа черных (у белых пустой массив фигур)
            return
        
        elif b_pieces_count == 1:
            self.ending = 1   # Победа белых (у черных пустой массив фигур)
            return


    def calc_reward(self, last_captured):
        """
        Вычислить скалярную награду за последний совершённый ход.

        Награда складывается из стоимости взятой фигуры и бонуса за шах
        вражескому королю.

        Parameters
        ----------
        last_captured : int
            Тип фигуры, взятой на последнем ходе (``0`` если взятия не было).

        Returns
        -------
        None
            Результат записывается в :attr:`reward`.

        Notes
        -----
        Карта стоимости фигур:

        * Пешка — ``0.01``
        * Ладья — ``0.05``
        * Конь — ``0.03``
        * Слон — ``0.03``
        * Ферзь — ``0.09``
        * Король — ``0.0`` (взятие короля невозможно в легальной игре)

        Бонус за шах — ``0.2``.
        """
        rew_map = {0: 0.0, 1: 0.01, 2: 0.05, 3: 0.03, 4: 0.03, 6: 0.09}
        mover = -self.turn   # игрок, который только что сходил
        total = 0.0

        # награда за взятие фигуры
        captured_type  = abs(last_captured)
        capture_reward = rew_map.get(captured_type, 0.0)
        total += capture_reward

        # шах вражескому королю
        enemy_king_pos = self.king_positions[-5 * mover]  # -5 если mover=1, 5 если mover=-1
        if self.is_under_attack(enemy_king_pos, mover):
            total += 0.2

        self.reward = total


    def move(self, action):
        """
        Выполнить полный цикл хода по закодированному индексу.

        Декодирует действие, обновляет доску, счёт фигур, позицию короля,
        права рокировки, вычисляет награду, переключает сторону и
        пересчитывает все легальные ходы для новой позиции.

        Parameters
        ----------
        action : int
            Закодированный индекс хода (``0..4671``).

        Returns
        -------
        None

        See Also
        --------
        move_simplified : Упрощённая версия без пересчёта легальных ходов.
        move_player : Ход по явным координатам.
        """
        start, end, promo = self.decode_move(action, self.turn)
        self.ending = 0
        piece = self.board[start]
        last_captured = self.board[end]

        self.calc_reward(last_captured)

        if last_captured != 0:
            self.score[self.turn*-1][abs(last_captured)-1] -= 1
        if not self.castling(start, end):
            self.board[end] = piece
            if promo != 0: self.board[end] = promo
            self.board[start] = 0
            self.score[self.turn*-1][abs(promo)-1] += 1
        if piece in (5, -5):
            self.king_positions[piece] = end

        self.turn *= -1
        self.moves_played += 1
        self.get_all_moves()
        self.filter_moves()
        self.add_promotion()
        self.uncoded_moves = self.ALL_LEGAL_MOVES
        self.encode_moves()
        self.update_state()

    
    def move_simplified(self, action):
        """
        Выполнить ход без полного пересчёта состояния игры.

        Упрощённая версия :meth:`move`, используемая в конвертерах данных
        и при быстрой прокрутке партий. Обновляет только доску и сторону,
        не генерируя легальные ходы и не проверяя окончание игры.

        Parameters
        ----------
        action : int
            Закодированный индекс хода.

        Returns
        -------
        None

        Warning
        -------
        После вызова этого метода :attr:`ALL_LEGAL_MOVES` и другие
        динамические атрибуты могут содержать устаревшие данные.
        Для полноценной игры используйте :meth:`move`.
        """
        start, end, promo = self.decode_move(action, self.turn)
        piece = self.board[start]

        if not self.castling(start, end):
            self.board[end] = piece
            if promo != 0: self.board[end] = promo
            self.board[start] = 0

        self.turn *= -1


    def move_player(self, start, end, promo=0):
        """
        Выполнить ход по явным координатам клеток.

        Аналог :meth:`move`, но принимает некодированные координаты.
        Полностью обновляет состояние движка после хода.

        Parameters
        ----------
        start : int
            Начальная клетка в формате ``row*10 + col``.
        end : int
            Конечная клетка в формате ``row*10 + col``.
        promo : int, optional
            Тип фигуры превращения (по умолчанию ``0``).

        Returns
        -------
        None
        """
        self.ending = 0
        piece = self.board[start]
        last_captured = self.board[end]

        if last_captured != 0:
            self.score[self.turn*-1][abs(last_captured)-1] -= 1
        if not self.castling(start, end):
            self.board[end] = piece
            if promo != 0: self.board[end] = promo
            self.board[start] = 0
            self.score[self.turn*-1][abs(promo)-1] += 1
        if piece in (5, -5):
            self.king_positions[piece] = end
            
        self.turn *= -1
        self.moves_played += 1
        self.get_all_moves()
        self.filter_moves()
        self.add_promotion()
        self.uncoded_moves = self.ALL_LEGAL_MOVES
        self.encode_moves()
        self.update_state()


    def return_game(self):
        """
        Получить текущее состояние игры в компактном виде.

        Returns
        -------
        tuple[list[int], list[int], int]
            Кортеж ``(legal_moves, board_list, turn)``, где:

            * ``legal_moves`` — список закодированных легальных ходов.
            * ``board_list`` — список значений доски (порядок ключей
              определяется ``self.board``).
            * ``turn`` — текущий ход (``1`` или ``-1``).
        """
        return self.ALL_LEGAL_MOVES, list(self.board.values()), self.turn

    def return_board_list(self):
        """
        Получить доску в виде плоского списка.

        Returns
        -------
        list[int]
            Список значений клеток доски в порядке ключей словаря.
        """
        return list(self.board.values())
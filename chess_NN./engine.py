class Engine():
    def __init__(self):
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
        
        self.castling_rights = {
            1: {"king": True, "left": True, "right": True},
            -1: {"king": True, "left": True, "right": True}
        }


    # проверяет, заблокирован ли путь
    def path_blocked(self, start, end):

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


    # проверяет, есть ли угроза фигуре
    def is_under_attack(self, pos, by):
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


    # функция поиска законных ходов для фигуры
    def search_moves(self, pos, dirs, repeat, enemy):
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


    # создаёт легальные ходы для всех фигур нужного цвета
    def get_all_moves(self):
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
        row = 1 if self.turn == 1 else 8

        # обновляет права в начале хода
        if start == row*10+1: self.castling_rights[self.turn]["right"] = False
        if start == row*10+8: self.castling_rights[self.turn]["left"] = False
        if start == row*10+5: self.castling_rights[self.turn]["king"] = False

        # делает рокировку если всё хорошо
        if self.castling_rights[self.turn]["king"]:
            if ((start, end) == (15, 18) and self.turn == 1) or \
               ((start, end) == (85, 88) and self.turn == -1):
                if not self.path_blocked(15, 18) and self.castling_rights[self.turn]["right"]:
                    self.board[row*10+7] = 5*self.turn
                    self.board[row*10+6] = 2*self.turn
                    self.board[row*10+5] = 0
                    self.board[row*10+8] = 0
                    return True
            if ((start, end) == (15, 11) and self.turn == 1) or \
               ((start, end) == (85, 81) and self.turn == -1):
                if not self.path_blocked(15, 11) and self.castling_rights[self.turn]["left"]:
                    self.board[row*10+3] = 5*self.turn
                    self.board[row*10+4] = 2*self.turn
                    self.board[row*10+5] = 0
                    self.board[row*10+8] = 0
                    self.board[row*10+1] = 0
                    return True
        return False

    # фильтрует псевдолегальные ходы
    def filter_moves(self):
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
                
                if not self.is_under_attack(king_pos, -self.turn):
                    valid.append(move)
                
                # возвращает всё на место
                self.board[pos] = piece
                self.board[move] = captured

            if valid: 
                legal[pos] = valid
        self.ALL_LEGAL_MOVES = legal


    def reset(self):
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


    # добавляет варианты превращения для других фигур
    def add_promotion(self):
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


    # кодирует ОДИН ход
    def encode_single_move(self, f, t, p):
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


    # превращает self.ALL_LEGAL_MOVES в единый массив, где каждый ход имеет уникальный индекс
    def encode_moves(self):
        actions = []
        for f, t, p in self.ALL_LEGAL_MOVES:
            action = self.encode_single_move(f, t, p)
            actions.append(action)
        self.ALL_LEGAL_MOVES = actions

    # декодирует ОДИН ход
    def decode_single_move(self, action, current_turn):
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


    def decode_move(self, action):
        return self.decode_single_move(action, self.turn)
    

    # обновляет self.endgame 
    def update_state(self):
        king = 5 if self.turn == 1 else -5
        king_pos = self.king_positions[king]
        
        in_check = self.is_under_attack(king_pos, -self.turn)

        # вывод результата игры
        if not self.ALL_LEGAL_MOVES:
            self.ending = -self.turn if in_check else 2


    # основная функция хода
    def move(self, action):
        start, end, promo = self.decode_move(action)
        self.ending = 0
        piece = self.board[start]

        if not self.castling(start, end):
            self.board[end] = piece
            if promo != 0: self.board[end] = promo
            self.board[start] = 0
        if piece in (5, -5):
            self.king_positions[piece] = end
        for p, _ in self.board.items():
            if self.board[p] == 5:
                wK = p
            elif self.board[p] == -5:
                bK = p
        if wK != self.king_positions[5] or bK != self.king_positions[-5]:
            raise

        self.turn *= -1
        self.moves_played += 1
        self.get_all_moves()
        self.filter_moves()
        self.add_promotion()
        self.uncoded_moves = self.ALL_LEGAL_MOVES
        self.encode_moves()
        self.update_state()
    

    def move_player(self, start, end, promo=0):
        self.ending = 0
        if not self.castling(start, end):
            self.board[end] = self.board[start]
            if promo != 0: self.board[end] = promo
            self.board[start] = 0
        self.turn *= -1
        self.moves_played += 1
        self.get_all_moves()
        self.filter_moves()
        self.add_promotion()
        self.uncoded_moves = self.ALL_LEGAL_MOVES
        self.update_state()


    def return_game(self):
        return self.board, self.ALL_LEGAL_MOVES, self.moves_played, self.ending
    def return_board_list(self): return list(self.board.values())
    def return_board(self): return self.board
    def return_moves(self): return self.moves_played
    def return_ending(self): return self.ending
    def return_alm(self): return self.ALL_LEGAL_MOVES
    def return_und(self): return self.uncoded_moves

class CliffWalkingEnv:
    """
    확률적 Cliff Walking 그리드월드.

    State:
        state = row * ncol + col

    Actions:
        0: up, 1: down, 2: left, 3: right

    Transition model:
        P[s][a] = [(prob, next_state, reward, done), ...]

    (1 - slip_chance)의 확률로 의도한 행동을 수행합니다.
    slip_chance의 확률로 에이전트는 다른 행동 중 하나로
    무작위하게 미끄러집니다.
    
    """

    def __init__(self, ncol: int = 12, nrow: int = 4, slip_chance: float = 0.01):
        self.ncol = ncol
        self.nrow = nrow
        self.slip_chance = slip_chance

        # 상태 및 행동의 수
        self.n_states = ncol * nrow
        self.n_actions = 4

        # 행동 기호
        self.action_meaning = ["^", "v", "<", ">"]

        # 특수 상태
        self.start_state = (nrow - 1) * ncol
        self.goal_state = nrow * ncol - 1
        self.cliff_states = list(range((nrow - 1) * ncol + 1, nrow * ncol - 1))

        # 전이 모델
        self.P = self._build_transition_model(slip_chance=self.slip_chance)

    def _build_transition_model(self, slip_chance: float = 0.01):
        """
        전이 모델 P를 구축합니다.

        Returns:
            P[s][a]: 상태 s에서 행동 a를 취했을 때 발생 가능한 결과들의 리스트.
        
        """
        P = [[[] for _ in range(self.n_actions)] for _ in range(self.n_states)]

        # [up, down, left, right]에 대한 (dx, dy)
        moves = [
            (0, -1),
            (0, 1),
            (-1, 0),
            (1, 0),
        ]

        for row in range(self.nrow):
            for col in range(self.ncol):
                state = row * self.ncol + col

                for action in range(self.n_actions):
                    # 종료 상태는 변경되지 않고 유지됨
                    if state in self.cliff_states or state == self.goal_state:
                        P[state][action] = [(1.0, state, 0.0, True)]
                        continue

                    for executed_action in range(self.n_actions):
                        dx, dy = moves[executed_action]

                        # 경계 클리핑이 적용된 다음 위치
                        next_col = min(self.ncol - 1, max(0, col + dx))
                        next_row = min(self.nrow - 1, max(0, row + dy))
                        next_state = next_row * self.ncol + next_col

                        reward = -1.0
                        done = False

                        if next_state in self.cliff_states:
                            reward = -100.0
                            done = True
                        elif next_state == self.goal_state:
                            done = True

                        if executed_action == action:
                            prob = 1 - slip_chance
                        else:
                            prob = slip_chance / (self.n_actions - 1)

                        P[state][action].append((prob, next_state, reward, done))

        return P

    def state_to_pos(self, state: int):
        """상태 인덱스를 (row, col)로 변환합니다."""
        row = state // self.ncol
        col = state % self.ncol
        return row, col

    def pos_to_state(self, row: int, col: int):
        """(row, col)을 상태 인덱스로 변환합니다."""
        return row * self.ncol + col
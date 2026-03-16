from __future__ import annotations

import json
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Experience:
    state_key: str
    action_name: str
    reward: float
    next_state_key: str
    next_action_names: tuple[str, ...]
    done: bool


class TabularQLearningAgent:
    def __init__(
        self,
        learning_rate: float = 0.20,
        gamma: float = 0.90,
        epsilon: float = 0.25,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.05,
        replay_capacity: int = 4096,
        replay_batch_size: int = 48,
        replay_passes: int = 2,
        random_seed: int = 42,
    ):
        self.learning_rate = float(learning_rate)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.epsilon_decay = float(epsilon_decay)
        self.epsilon_min = float(epsilon_min)
        self.replay_batch_size = int(max(1, replay_batch_size))
        self.replay_passes = int(max(1, replay_passes))
        self.random = random.Random(random_seed)
        self.q_table: dict[str, dict[str, float]] = defaultdict(dict)
        self.memory: deque[Experience] = deque(maxlen=max(32, int(replay_capacity)))

    def q_value(self, state_key: str, action_name: str) -> float:
        return float(self.q_table.get(state_key, {}).get(action_name, 0.0))

    def select_action(
        self,
        state_key: str,
        action_names: list[str],
        greedy: bool = False,
        action_priors: dict[str, float] | None = None,
    ) -> str:
        if not action_names:
            raise ValueError("select_action requires at least one available action.")
        if (not greedy) and (self.random.random() < self.epsilon):
            return self.random.choice(action_names)
        priors = action_priors or {}
        return max(
            action_names,
            key=lambda action_name: self.q_value(state_key, action_name) + 0.15 * float(priors.get(action_name, 0.0)),
        )

    def _update_core(
        self,
        state_key: str,
        action_name: str,
        reward: float,
        next_state_key: str,
        next_action_names: list[str] | tuple[str, ...],
        done: bool,
    ) -> None:
        old_value = self.q_value(state_key, action_name)
        next_value = 0.0 if done or not next_action_names else max(self.q_value(next_state_key, candidate) for candidate in next_action_names)
        target = reward + self.gamma * next_value
        new_value = old_value + self.learning_rate * (target - old_value)
        self.q_table.setdefault(state_key, {})[action_name] = float(new_value)

    def update(
        self,
        state_key: str,
        action_name: str,
        reward: float,
        next_state_key: str,
        next_action_names: list[str],
        done: bool,
    ) -> None:
        self._update_core(state_key, action_name, reward, next_state_key, next_action_names, done)
        self.memory.append(
            Experience(
                state_key=state_key,
                action_name=action_name,
                reward=float(reward),
                next_state_key=next_state_key,
                next_action_names=tuple(next_action_names),
                done=bool(done),
            )
        )

    def replay(self) -> None:
        if len(self.memory) < self.replay_batch_size:
            return
        for _ in range(self.replay_passes):
            sample = self.random.sample(list(self.memory), k=min(len(self.memory), self.replay_batch_size))
            for exp in sample:
                self._update_core(
                    exp.state_key,
                    exp.action_name,
                    exp.reward,
                    exp.next_state_key,
                    exp.next_action_names,
                    exp.done,
                )

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.q_table, indent=2), encoding="utf-8")

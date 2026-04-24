
import json
import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from statistics import mean

CLIMATE_STATE_FILE = Path(__file__).resolve().parents[1] / ".climate_state.json"

CURRICULUM_CONFIGS = {
    "qlearning": {
        "eval_window": 10,
        "advance_threshold": {1: 0.45, 2: 0.52},
        "retreat_threshold": 0.28,
        "min_episodes_at_level": {1: 15, 2: 10, 3: 999},
        "persist_difficulty": True,
        "reset_on_init": False,
    },
    "qwen": {
        "eval_window": 15,
        "advance_threshold": {1: 0.55, 2: 0.62},
        "retreat_threshold": 0.32,
        "min_episodes_at_level": {1: 30, 2: 25, 3: 999},
        "persist_difficulty": False,
        "reset_on_init": True,
    },
}

class ClimateAgent:
    def __init__(self, agent_type: str = "qlearning"):
        self.agent_type = agent_type
        self.config = CURRICULUM_CONFIGS[agent_type]
        self.current_difficulty = 1
        self.performance_history: List[float] = []
        self.advancement_log: List[Dict] = []
        self.episodes_at_current_level: int = 0

        if self.config["persist_difficulty"] and not self.config["reset_on_init"]:
            self._load_state()

    def generate_weather_sequence(self, num_days: int, difficulty: int) -> List[Dict]:
        from dynamics import WeatherEngine
        sequence = WeatherEngine.generate_sequence(num_days, difficulty)
        drought_start = None

        if difficulty >= 2:
            drought_start = random.randint(25, 35)
            for i in range(drought_start, min(drought_start + 14, num_days)):
                sequence[i]["condition"] = "drought"
                sequence[i]["humidity"] *= 0.55
                sequence[i]["rainfall_mm"] = 0
                sequence[i]["temp_c"] = min(45, sequence[i]["temp_c"] * 1.08)

        if difficulty >= 3:
            flood_start = (drought_start + 15) if drought_start is not None else 50
            for i in range(flood_start, min(flood_start + 5, num_days)):
                sequence[i]["condition"] = "storm"
                sequence[i]["humidity"] = 0.97
                sequence[i]["rainfall_mm"] = random.uniform(45, 80)

        return sequence

    def update(self, episode_reward: float):
        self.performance_history.append(episode_reward)
        self.episodes_at_current_level += 1

        cfg = self.config
        min_eps = cfg["min_episodes_at_level"].get(self.current_difficulty, 10)
        window = cfg["eval_window"]

        if self.episodes_at_current_level < min_eps:
            return

        if len(self.performance_history) < window:
            return

        recent_avg = mean(self.performance_history[-window:])
        advance_thresh = cfg["advance_threshold"].get(self.current_difficulty, 999)

        if recent_avg > advance_thresh and self.current_difficulty < 3:
            self.current_difficulty += 1
            self.episodes_at_current_level = 0
            if cfg["persist_difficulty"]:
                self._save_state()
            print(f"Curriculum: Advanced to level {self.current_difficulty}")
            return

        if recent_avg < cfg["retreat_threshold"] and self.current_difficulty > 1:
            self.current_difficulty -= 1
            self.episodes_at_current_level = 0
            if cfg["persist_difficulty"]:
                self._save_state()
            print(f"Curriculum: Retreating to level {self.current_difficulty}")

    def _load_state(self):
        if not CLIMATE_STATE_FILE.exists():
            return
        try:
            data = json.loads(CLIMATE_STATE_FILE.read_text())
            self.current_difficulty = max(1, min(3, int(data.get("current_difficulty", 1))))
        except:
            pass

    def _save_state(self):
        CLIMATE_STATE_FILE.write_text(json.dumps({"current_difficulty": self.current_difficulty}))

    def force_difficulty(self, level: int):
        self.current_difficulty = level
        self.episodes_at_current_level = 0
        self.performance_history = []
        if self.config["persist_difficulty"]:
            self._save_state()

    def get_status(self) -> Dict:
        return {
            "level": self.current_difficulty,
            "episodes_at_level": self.episodes_at_current_level,
            "avg_reward": round(mean(self.performance_history[-10:]), 3) if self.performance_history else 0
        }
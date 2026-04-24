
import random
from typing import Dict, List, Any

SCENARIO_BANK: Dict[int, List[Dict]] = {
    1: [
        {"name": "Normal Cotton Season", "weather_type": "normal", "starting_pest_pressure": 0.08, "crop_variety": "cotton_desi"},
        {"name": "Bt Cotton Trial", "weather_type": "normal", "starting_pest_pressure": 0.05, "crop_variety": "cotton_bt"},
    ],
    2: [
        {"name": "Drought Stress Season", "weather_type": "drought", "starting_pest_pressure": 0.12, "crop_variety": "cotton_desi"},
        {"name": "Market Suppression Season", "weather_type": "normal", "starting_pest_pressure": 0.10, "crop_variety": "cotton_desi"},
    ],
    3: [
        {"name": "Triple Crisis", "weather_type": "compound", "starting_pest_pressure": 0.15, "crop_variety": "cotton_desi"},
        {"name": "Policy Shock Season", "weather_type": "normal", "starting_pest_pressure": 0.10, "crop_variety": "cotton_desi"},
    ],
}

class TaskRegistry:
    def sample_scenario(self, difficulty: int) -> Dict[str, Any]:
        pool = SCENARIO_BANK.get(difficulty, SCENARIO_BANK[1])
        return random.choice(pool)

class CurriculumManager:
    def __init__(self, climate_agent):
        self.climate_agent = climate_agent

    @property
    def current_difficulty(self) -> int:
        return self.climate_agent.current_difficulty

    def update(self, episode_reward: float):
        self.climate_agent.update(episode_reward)

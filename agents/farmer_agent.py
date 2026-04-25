
import json
import os
import random
import re
import numpy as np
from typing import Dict, Tuple, Optional, List
from pathlib import Path

ACTIONS = [
    "irrigate_low", "irrigate_medium", "irrigate_high",
    "spray_pesticide", "spray_fungicide",
    "apply_fertilizer_low", "apply_fertilizer_high",
    "prune_crop",
    "sell_crop_25pct", "sell_crop_50pct", "sell_crop_all",
    "consult_district_advisor", "call_soil_test", "call_pest_advisory",
    "call_satellite_imagery", "check_insurance_portal", "check_mandi_prices",
    "apply_for_loan", "file_insurance_claim", "repay_loan",
    "do_nothing",
]

ACTION_TO_IDX = {a: i for i, a in enumerate(ACTIONS)}
NUM_ACTIONS = len(ACTIONS)

ACTION_COSTS = {
    "irrigate_low": 50, "irrigate_medium": 120, "irrigate_high": 250,
    "spray_pesticide": 400, "spray_fungicide": 350,
    "apply_fertilizer_low": 280, "apply_fertilizer_high": 580,
    "prune_crop": 150,
    "consult_district_advisor": 50, "call_soil_test": 200,
    "call_pest_advisory": 80, "call_satellite_imagery": 150,
    "check_insurance_portal": 0, "check_mandi_prices": 0,
    "apply_for_loan": 0, "file_insurance_claim": 0, "repay_loan": 0,
    "sell_crop_25pct": 0, "sell_crop_50pct": 0, "sell_crop_all": 0,
    "do_nothing": 0,
}

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints")

def discretize_state(farm_state: Dict) -> Tuple[int, ...]:
    day = farm_state.get("day", 1)
    if day <= 15: day_phase = 0
    elif day <= 45: day_phase = 1
    elif day <= 70: day_phase = 2
    else: day_phase = 3

    moisture = farm_state.get("soil_moisture", 0.5)
    moisture_level = 0 if moisture < 0.30 else (2 if moisture > 0.70 else 1)

    pest = farm_state.get("pest_pressure", farm_state.get("pest_pressure_observed", 0.1))
    pest_level = 0 if pest < 0.25 else (2 if pest > 0.55 else 1)

    fungal = farm_state.get("fungal_risk", 0.1)
    fungal_level = 0 if fungal < 0.25 else (2 if fungal > 0.55 else 1)

    nitrogen = farm_state.get("soil_nitrogen", 0.6)
    nitrogen_level = 0 if nitrogen < 0.35 else 1

    budget = farm_state.get("budget", 15000)
    budget_level = 0 if budget < 3000 else (2 if budget > 10000 else 1)

    insurance = 1 if farm_state.get("insurance_enrolled", False) else 0
    crop_stage = min(farm_state.get("crop_stage", 0), 4)

    yield_acc = farm_state.get("yield_accumulated", 0)
    yield_level = 0 if yield_acc < 0.5 else (2 if yield_acc > 3.0 else 1)

    return (day_phase, moisture_level, pest_level, fungal_level, nitrogen_level, budget_level, insurance, crop_stage, yield_level)

def generate_reasoning(action: str, farm_state: Dict, q_values: np.ndarray) -> str:
    day = farm_state.get("day", 1)
    moisture = farm_state.get("soil_moisture", 0.5)
    pest = farm_state.get("pest_pressure", farm_state.get("pest_pressure_observed", 0.1))
    budget = farm_state.get("budget", 15000)
    
    templates = {
        "irrigate_low": f"Day {day}: Maintenance irrigation to stabilize moisture at {moisture:.0%}.",
        "irrigate_medium": f"Day {day}: Moisture level ({moisture:.0%}) dropping. Medium irrigation should reset threshold.",
        "irrigate_high": f"Day {day}: Critical deficit ({moisture:.0%}). Deploying high irrigation to prevent health loss.",
        "spray_pesticide": f"Day {day}: Pest pressure at {pest:.0%}. Spraying now to prevent exponential growth.",
        "spray_fungicide": f"Day {day}: High fungal risk detected. Targeted fungicide applied.",
        "apply_fertilizer_low": f"Day {day}: Nutrient boost for growth stage {farm_state.get('crop_stage')}.",
        "check_insurance_portal": f"Day {day}: Managing insurance coverage before deadline.",
        "sell_crop_all": f"Day {day}: Final harvest sale. Budget at Rs.{budget:,}.",
        "do_nothing": f"Day {day}: Indicators stable. Conserving budget for next steps."
    }
    return templates.get(action, f"Day {day}: Taking action {action} based on current farm metrics.")

class FarmerAgent:
    def __init__(self, alpha: float = 0.15, gamma: float = 0.92, epsilon: float = 1.0):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table: Dict[Tuple, np.ndarray] = {}
        self.episode_count = 0
        self.episode_rewards: List[float] = []
        self.step_history: List[Dict] = []
        self._prev_state = None
        self._prev_action_idx = None
        self.action_counts = {a: 0 for a in ACTIONS}
        self.market_belief = "unknown"
        
        self._seed_priors()

    def update_market_belief(self, farm_state: Dict):
        price = farm_state.get("displayed_price", 5800)
        if price < 5400: self.market_belief = "likely_manipulating"
        elif price > 5800: self.market_belief = "fair"
        return self.market_belief

    def _seed_priors(self):
        # Basic heuristic seeding
        for day in range(4):
            for m in range(3):
                for p in range(3):
                    for f in range(3):
                        for n in range(2):
                            for b in range(3):
                                for i in range(2):
                                    for s in range(5):
                                        for y in range(3):
                                            state = (day, m, p, f, n, b, i, s, y)
                                            q = np.zeros(NUM_ACTIONS)
                                            if m == 0: q[ACTION_TO_IDX["irrigate_medium"]] = 0.5
                                            if p == 2: q[ACTION_TO_IDX["spray_pesticide"]] = 0.6
                                            if i == 0 and day == 0: q[ACTION_TO_IDX["check_insurance_portal"]] = 0.4
                                            self.q_table[state] = q

    def _get_valid_actions(self, farm_state: Dict) -> List[int]:
        budget = farm_state.get("budget", 0)
        valid = [i for i, a in enumerate(ACTIONS) if ACTION_COSTS[a] <= budget]
        return valid if valid else [ACTION_TO_IDX["do_nothing"]]

    def select_action(self, farm_state: Dict) -> Tuple[str, str]:
        self.update_market_belief(farm_state)
        state = discretize_state(farm_state)
        if state not in self.q_table: self.q_table[state] = np.zeros(NUM_ACTIONS)
        
        valid = self._get_valid_actions(farm_state)
        if random.random() < self.epsilon:
            idx = random.choice(valid)
        else:
            q_vals = self.q_table[state]
            idx = int(np.argmax([q_vals[i] for i in valid]))
            idx = valid[idx]

        self._prev_state, self._prev_action_idx = state, idx
        name = ACTIONS[idx]
        self.action_counts[name] += 1
        return name, generate_reasoning(name, farm_state, self.q_table[state])

    def update(self, reward: float, next_farm_state: Dict, done: bool):
        if self._prev_state is None: return
        q_vals = self.q_table[self._prev_state]
        
        if done: target = reward
        else:
            ns = discretize_state(next_farm_state)
            if ns not in self.q_table: self.q_table[ns] = np.zeros(NUM_ACTIONS)
            target = reward + self.gamma * np.max(self.q_table[ns])
            
        q_vals[self._prev_action_idx] += self.alpha * (target - q_vals[self._prev_action_idx])
        self.step_history.append({"state": self._prev_state, "idx": self._prev_action_idx})

    def end_episode(self, reward: float):
        # Backpropagate bonus
        for step in self.step_history:
            self.q_table[step["state"]][step["idx"]] += self.alpha * reward * 0.1
            
        self.epsilon = max(0.05, self.epsilon * 0.98)
        self.episode_rewards.append(reward)
        self.episode_count += 1
        self.step_history = []
        self._prev_state = None

    def save(self):
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        path = os.path.join(CHECKPOINT_DIR, "farmer_agent.json")
        data = {
            "epsilon": self.epsilon,
            "q_table": {",".join(map(str, k)): v.tolist() for k, v in self.q_table.items()}
        }
        with open(path, "w") as f: json.dump(data, f)

    def load(self):
        path = os.path.join(CHECKPOINT_DIR, "farmer_agent.json")
        if not os.path.exists(path): return False
        with open(path, "r") as f:
            data = json.load(f)
            self.epsilon = data["epsilon"]
            self.q_table = {tuple(map(int, k.split(","))): np.array(v) for k, v in data["q_table"].items()}
        return True

    def get_stats(self) -> Dict:
        return {"episodes": self.episode_count, "epsilon": round(self.epsilon, 3), "avg_reward": round(np.mean(self.episode_rewards[-10:]), 3) if self.episode_rewards else 0}

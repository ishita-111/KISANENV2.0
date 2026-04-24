
import json
import random
import numpy as np
from typing import Any, Dict, Optional, Tuple
from pathlib import Path

class Environment:
    def reset(self): raise NotImplementedError
    def step(self, action): raise NotImplementedError

from dynamics import FarmState, WeatherEngine, PestDynamics, SoilChemistry, CropGrowthModel, resolve_action
from grader import RewardEngine, OversightAuditor
from tasks import TaskRegistry, CurriculumManager
from agents.market_agent import MarketAgent
from agents.district_farm_advisor import DistrictFarmAdvisor
from agents.climate_agent import ClimateAgent

EPISODE_LENGTH = 90
STARTING_BUDGET = 15_000
INSURANCE_DEADLINE = 15
SOIL_PERSISTENCE_FILE = Path(__file__).with_name(".soil_state.json")

class KisanEnv(Environment):
    metadata = {
        "name": "KisanEnv",
        "version": "2.0.0",
        "episode_length": EPISODE_LENGTH,
    }

    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        self.config = config or {}
        self.market_agent = MarketAgent()
        self.district_advisor = DistrictFarmAdvisor()
        self.climate_agent = ClimateAgent()
        self.reward_engine = RewardEngine()
        self.oversight_auditor = OversightAuditor(self.district_advisor)
        self.task_registry = TaskRegistry()
        self.curriculum_manager = CurriculumManager(self.climate_agent)

        self.farm_state = None
        self.weather_sequence = []
        self.episode_log = []
        self.episode_count = 0
        self.reflection_memory = []
        self._soil_health = self._load_soil_health()
        self.tool_call_log = []
        self._pending_tool_results = {}
        self._tool_result_timers = {}

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        difficulty = self.curriculum_manager.current_difficulty
        scenario = self.task_registry.sample_scenario(difficulty)
        self.weather_sequence = self.climate_agent.generate_weather_sequence(EPISODE_LENGTH, difficulty)

        self.farm_state = FarmState(
            day=1,
            budget=STARTING_BUDGET,
            soil_health=max(0.1, min(1.0, self._soil_health)),
            crop_variety=scenario.get("crop_variety", "cotton_desi"),
            pest_pressure=scenario.get("starting_pest_pressure", 0.08),
            weather_sequence=self.weather_sequence,
        )

        self.market_agent.reset(scenario=scenario)
        self.district_advisor.reset(self.episode_count)
        self.episode_log, self.tool_call_log = [], []
        self._pending_tool_results, self._tool_result_timers = {}, {}

        return self._build_observation(None, 0.0)

    def step(self, action: str) -> Tuple[Dict, float, bool, Dict]:
        from inference import ActionParser
        parsed = ActionParser.parse(action)
        name, reasoning = parsed.get("action", "do_nothing"), parsed.get("reasoning", "")

        self._process_pending_tools()
        result = resolve_action(name, self.farm_state, self.weather_sequence[self.farm_state.day-1], self.market_agent, self._pending_tool_results)
        self.farm_state.apply_action_result(result)

        if name.startswith("call_") or name in ("check_insurance_portal", "check_mandi_prices", "consult_district_advisor"):
            self._register_tool_call(name, result)

        self.market_agent.step(self.farm_state.day)
        advisor = self.district_advisor.step(self.farm_state.day, self.farm_state)
        oversight = self.oversight_auditor.evaluate_decision(name, reasoning, self.farm_state, result)

        self._apply_dynamics()
        step_reward = self.reward_engine.compute_step_reward(name, result, self.farm_state, reasoning, oversight["score"])

        self.episode_log.append({
            "day": self.farm_state.day, "action": name, "reasoning": reasoning, 
            "reward": step_reward, "oversight": oversight, "farm": self.farm_state.to_dict()
        })

        done = self._check_done()
        self.farm_state.day += 1
        obs = self._build_observation(name, step_reward, advisor.get("advice"), oversight)

        info = {"step": self.farm_state.day, "reward": step_reward, "oversight": oversight}
        if done:
            ep_reward = self._finalize_episode()
            info.update({"episode_reward": ep_reward, "reflection": self._run_reflection()})
            step_reward = ep_reward

        return obs, step_reward, done, info

    def _build_observation(self, last_action, reward, advice=None, oversight=None) -> Dict:
        fs = self.farm_state
        weather = self.weather_sequence[min(fs.day-1, 89)]
        forecast = " -> ".join([w["condition"] for w in self.weather_sequence[fs.day-1:fs.day+2]])
        
        reflection = f"\nPREVIOUS LEARNING: {self.reflection_memory[-1]}\n" if self.reflection_memory else ""
        ins_warning = f"\nINSURANCE DEADLINE: {15-fs.day+1} days left!" if not fs.insurance_enrolled and fs.day <= 15 else ""
        
        feedback = ""
        if oversight:
            feedback = f"\nADVISOR FEEDBACK: {oversight['explanation']} (Score: {oversight['score']:.0%})\n"

        prompt = f"""
Day {fs.day}/90 | Budget: Rs.{fs.budget:,} | Yield: {fs.yield_accumulated:.2f}q
Health: {fs.crop_health:.0%} | Soil Moisture: {fs.soil_moisture:.0%} | Nitrogen: {fs.soil_nitrogen:.0%}
Weather: {weather['condition']} ({weather['temp_c']}C) | Forecast: {forecast}
Pests: {fs.observed_pest_pressure:.0%} | Fungal: {fs.fungal_risk:.0%}
Market: Rs.{self.market_agent.displayed_price:,}/q | Insurance: {'ENROLLED' if fs.insurance_enrolled else 'NO'}{ins_warning}
{reflection}{advice or 'No specific advice today.'}{feedback}
ACTIONS: irrigate_low/med/high, spray_pesticide, spray_fungicide, apply_fertilizer_low/high, sell_crop_25/50/all, consult_district_advisor, call_soil_test, call_pest_advisory, check_insurance_portal, check_mandi_prices, do_nothing
Format: ACTION: <name>\nREASONING: <logic>
"""
        return {"prompt": prompt, "day": fs.day, "farm_state": fs.to_dict()}

    def _apply_dynamics(self):
        w = self.weather_sequence[min(self.farm_state.day-1, 89)]
        CropGrowthModel.advance(self.farm_state, w)
        PestDynamics.daily_spread(self.farm_state, w, 0.3)
        SoilChemistry.daily_update(self.farm_state, w)
        self.farm_state.fungal_risk = WeatherEngine.compute_fungal_risk(w["humidity"], w["temp_c"], self.farm_state.crop_stage, self.farm_state.crop_variety)

    def _check_done(self) -> bool:
        return self.farm_state.day >= EPISODE_LENGTH or self.farm_state.budget <= -5000 or self.farm_state.crop_health <= 0

    def _finalize_episode(self) -> float:
        reward = self.reward_engine.compute_episode_reward(self.farm_state, self.episode_log, self.market_agent, self._soil_health)
        self._soil_health = min(1.0, self.farm_state.soil_health + 0.08)
        self._save_soil_health(self._soil_health)
        self.curriculum_manager.update(reward)
        self.episode_count += 1
        return reward

    def _run_reflection(self) -> str:
        from grader import EpisodeReflector
        refl = EpisodeReflector.reflect(self.episode_log)
        if refl: self.reflection_memory = (self.reflection_memory + [refl])[-3:]
        return refl

    def _process_pending_tools(self):
        for tool in list(self._tool_result_timers.keys()):
            self._tool_result_timers[tool] -= 1
            if self._tool_result_timers[tool] <= 0:
                self._pending_tool_results[tool]["ready"] = True
                del self._tool_result_timers[tool]

    def _register_tool_call(self, name, result):
        delays = {"call_soil_test": 3, "call_pest_advisory": 1, "call_satellite_imagery": 2, "consult_district_advisor": 1}
        d = delays.get(name, 0)
        self._pending_tool_results[name] = {"ready": d == 0, "data": result.get("tool_result", "Pending")}
        if d > 0: self._tool_result_timers[name] = d

    def _load_soil_health(self) -> float:
        if SOIL_PERSISTENCE_FILE.exists():
            try: return float(json.loads(SOIL_PERSISTENCE_FILE.read_text()).get("soil_health", 0.7))
            except: pass
        return 0.7

    def _save_soil_health(self, v: float):
        SOIL_PERSISTENCE_FILE.write_text(json.dumps({"soil_health": v}))

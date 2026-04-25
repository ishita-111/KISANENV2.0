
import json
import random
import numpy as np
from typing import Any, Dict, Optional, Tuple
from pathlib import Path

from openenv import Environment

from dynamics import FarmState, WeatherEngine, PestDynamics, SoilChemistry, CropGrowthModel, resolve_action
from grader import RewardEngine, OversightAuditor, ReasoningScorer
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
        self._obs_history = []
        self._episode_seed = None
        self._is_baseline = False
        self._last_reward_breakdown = {}

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        self._episode_seed = seed if seed is not None else random.randint(0, 9999)
        random.seed(self._episode_seed)
        np.random.seed(self._episode_seed)

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
        self._obs_history = []
        self._last_reward_breakdown = {}

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
        step_reward = float(self.reward_engine.compute_step_reward(name, result, self.farm_state, reasoning, oversight["score"]))

        # Track observation history for trend summary (FEATURE 3)
        self._obs_history.append({
            "soil_moisture": self.farm_state.soil_moisture,
            "pest": self.farm_state.observed_pest_pressure,
            "action": name
        })
        self._obs_history = self._obs_history[-7:]

        self.episode_log.append({
            "day": self.farm_state.day,
            "action": name,
            "reasoning": reasoning,
            "reward": step_reward,
            "oversight": oversight,
            "oversight_score": oversight.get("score", 0.5),
            "market_mode": self.market_agent.mode,
            "farm_snapshot": self.farm_state.to_dict()
        })

        done = self._check_done()
        self.farm_state.day += 1
        obs = self._build_observation(name, step_reward, advisor.get("advice"), oversight)

        # Reasoning analysis for transparency (FEATURE 6)
        reasoning_scores = {
            "causal_words": int(sum(1 for w in ReasoningScorer.CAUSAL_WORDS if w in reasoning.lower())),
            "numbers_cited": int(len(ReasoningScorer.NUMERICAL_PATTERN.findall(reasoning))),
            "word_count": int(len(reasoning.split())),
            "total_score": float(round(ReasoningScorer.score(reasoning), 3))
        }

        info = {"step": self.farm_state.day, "reward": float(step_reward), "oversight": oversight, "reasoning_analysis": reasoning_scores}
        if done:
            ep_reward = self._finalize_episode()
            info.update({
                "episode_reward": ep_reward,
                "reflection": self._run_reflection(),
                "reward_breakdown": self._last_reward_breakdown
            })
            step_reward = ep_reward

        return obs, float(step_reward), bool(done), info

    def get_info(self) -> Dict:
        return {
            "episode": self.episode_count,
            "difficulty": self.curriculum_manager.climate_agent.current_difficulty,
            "soil_health": round(self._soil_health, 3),
            "market_state": self.market_agent.to_dict(),
        }

    def _field_observation_text(self) -> str:
        fs = self.farm_state
        observations = []
        
        if fs.fungal_risk > 0.65:
            observations.append(
                "Yellowing at lower leaf margins with dark brown spots — pattern consistent with fungal blight."
            )
        elif fs.fungal_risk > 0.35:
            observations.append("Mild discoloration on ~10% of plants. Monitor closely.")
        
        if fs.observed_pest_pressure > 0.55:
            observations.append(
                "Webbing visible on upper canopy. Insect frass present. Heavy infestation pattern — not fungal."
            )
        elif fs.observed_pest_pressure > 0.25:
            observations.append("Small irregular holes in leaves. Early pest activity, not yet critical.")
        
        if fs.soil_moisture < 0.20:
            observations.append(
                "Severe wilting observed — leaves curling inward by afternoon. Immediate irrigation required."
            )
        elif fs.soil_moisture < 0.32:
            observations.append("Slight midday leaf droop. Soil feels dry 2 inches down.")
        
        if not observations:
            return "No visible stress indicators. Crop appears healthy."
        return "FIELD OBSERVATION:\n" + " ".join(observations)

    def _trend_summary(self) -> str:
        if len(self._obs_history) < 3:
            return ""
        h = self._obs_history
        moisture_delta = h[-1]["soil_moisture"] - h[0]["soil_moisture"]
        pest_delta = h[-1]["pest"] - h[0]["pest"]
        recent_actions = [e["action"] for e in h[-4:]]
        do_nothing_streak = sum(1 for a in recent_actions if a == "do_nothing")
        
        lines = []
        if moisture_delta < -0.08:
            pct = abs(moisture_delta * 100)
            lines.append(f"Moisture fell {pct:.0f}% over {len(h)} days — irrigation urgency rising.")
        if pest_delta > 0.08:
            lines.append(f"Pest pressure rose {pest_delta*100:.0f}% over {len(h)} days — trend is accelerating.")
        if do_nothing_streak >= 3:
            lines.append(f"WARNING: {do_nothing_streak} consecutive inactions — advisor flags passive risk.")
        
        return ("RECENT TREND (" + str(len(h)) + " days):\n" + "\n".join(lines) + "\n") if lines else ""

    def _build_observation(self, last_action, reward, advice=None, oversight=None) -> Dict:
        fs = self.farm_state
        weather = self.weather_sequence[min(fs.day-1, 89)]
        forecast = " -> ".join([w["condition"] for w in self.weather_sequence[fs.day-1:fs.day+2]])
        
        reflection = f"\nPREVIOUS LEARNING: {self.reflection_memory[-1]}\n" if self.reflection_memory else ""
        ins_warning = f"\nINSURANCE DEADLINE: {15-fs.day+1} days left!" if not fs.insurance_enrolled and fs.day <= 15 else ""
        
        feedback = ""
        if oversight:
            feedback = f"\nADVISOR FEEDBACK: {oversight['explanation']} (Score: {oversight['score']:.0%})\n"

        # Misdiagnosis warning (FEATURE 5)
        misdiagnosis_warning = ""
        if fs.misdiagnosis_penalty_active:
            misdiagnosis_warning = "\n⚠ YESTERDAY'S MISDIAGNOSIS: Chemical application disrupted soil microbiome. Fungal risk elevated today.\n"

        field_obs = self._field_observation_text()
        trend = self._trend_summary()

        prompt = f"""
=== FARM STATUS ===
Day {fs.day}/90 | Budget: Rs.{fs.budget:,} | Yield: {fs.yield_accumulated:.2f}q
Health: {fs.crop_health:.0%} | Soil Moisture: {fs.soil_moisture:.0%} | Nitrogen: {fs.soil_nitrogen:.0%}
Weather: {weather['condition']} ({weather['temp_c']}C) | Forecast: {forecast}
Pests: {fs.observed_pest_pressure:.0%} | Fungal: {fs.fungal_risk:.0%}

=== MARKET ===
Price: Rs.{self.market_agent.displayed_price:,}/q | Insurance: {'ENROLLED' if fs.insurance_enrolled else 'NO'}{ins_warning}

{field_obs}
{trend}{misdiagnosis_warning}{reflection}{advice or 'No specific advice today.'}{feedback}
=== AVAILABLE ACTIONS ===
irrigate_low/med/high, spray_pesticide, spray_fungicide, apply_fertilizer_low/high, sell_crop_25/50/all, consult_district_advisor, call_soil_test, call_pest_advisory, check_insurance_portal, check_mandi_prices, do_nothing
Format: ACTION: <name>\\nREASONING: <logic>
"""
        return {"prompt": prompt, "day": fs.day, "farm_state": fs.to_dict()}

    def _apply_dynamics(self):
        w = self.weather_sequence[min(self.farm_state.day-1, 89)]
        CropGrowthModel.advance(self.farm_state, w)
        PestDynamics.daily_spread(self.farm_state, w, 0.3)
        SoilChemistry.daily_update(self.farm_state, w)
        self.farm_state.fungal_risk = WeatherEngine.compute_fungal_risk(w["humidity"], w["temp_c"], self.farm_state.crop_stage, self.farm_state.crop_variety)
        
        # Misdiagnosis cascade (FEATURE 5)
        if self.farm_state.misdiagnosis_penalty_active:
            self.farm_state.fungal_risk = min(1.0, self.farm_state.fungal_risk + 0.15)
            self.farm_state.misdiagnosis_penalty_active = False

    def _check_done(self) -> bool:
        return bool(self.farm_state.day >= EPISODE_LENGTH or self.farm_state.budget <= -5000 or self.farm_state.crop_health <= 0)

    def _finalize_episode(self) -> float:
        # Contrastive reward: compare against heuristic on same seed (FEATURE 4)
        contrastive_bonus = 0.0
        if not self._is_baseline:
            try:
                from training.heuristic_baseline import HeuristicAgent
                heuristic_env = KisanEnv()
                heuristic_env._is_baseline = True
                heuristic_env.reset(seed=self._episode_seed)
                heuristic = HeuristicAgent()
                h_done = False
                while not h_done:
                    h_action = heuristic.decide(heuristic_env.farm_state.to_dict())
                    _, _, h_done, _ = heuristic_env.step(h_action)

                heuristic_profit = heuristic_env.farm_state.revenue_earned
                agent_profit = self.farm_state.revenue_earned
                contrastive_bonus = float(np.clip((agent_profit - heuristic_profit) / 5000, -0.2, 0.2))
            except Exception:
                contrastive_bonus = 0.0

        base_reward, breakdown = self.reward_engine.compute_episode_reward(
            self.farm_state, self.episode_log, self.market_agent, self._soil_health
        )
        
        final_reward = float(np.clip(base_reward + contrastive_bonus, 0.0, 1.0))
        breakdown["contrastive_bonus"] = {
            "raw_score": round(contrastive_bonus, 3),
            "weight": 1.0,
            "contribution": round(contrastive_bonus, 3)
        }
        self._last_reward_breakdown = breakdown

        self._soil_health = min(1.0, self.farm_state.soil_health + 0.02)
        self._save_soil_health(self._soil_health)
        self.curriculum_manager.update(final_reward)
        self.episode_count += 1
        return final_reward

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
        p = Path(SOIL_PERSISTENCE_FILE)
        if p.exists():
            try: return float(json.loads(p.read_text()).get("soil_health", 0.7))
            except: pass
        return 0.7

    def _save_soil_health(self, v: float):
        Path(SOIL_PERSISTENCE_FILE).write_text(json.dumps({"soil_health": v}))

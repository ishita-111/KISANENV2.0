
"""
KisanEnv 2.0 — Full Test Suite
Tests all API endpoints, environment mechanics, and reward logic.
Run: pytest test_endpoints.py -v
"""

import pytest
import json
import random
from httpx import AsyncClient, ASGITransport
from fastapi.testclient import TestClient


from run import app
from env import KisanEnv
from dynamics import FarmState, resolve_action
from grader import RewardEngine, ReasoningScorer, OversightAuditor
from inference import ActionParser
from agents.market_agent import MarketAgent
from agents.district_farm_advisor import DistrictFarmAdvisor
from agents.climate_agent import ClimateAgent


@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def env():
    e = KisanEnv()
    e.reset(seed=42)
    return e

@pytest.fixture
def farm_state():
    return FarmState()


class TestAPIEndpoints:

    def test_health(self, client):
        """Health check must return 200."""
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_reset(self, client):
        """Reset must return valid observation prompt."""
        r = client.post("/reset", json={})
        assert r.status_code == 200
        data = r.json()
        assert "observation" in data
        assert "farm_state" in data
        assert data["day"] == 1

        obs = data["observation"]
        assert "FARM STATUS" in obs
        assert "AVAILABLE ACTIONS" in obs
        assert "MARKET" in obs

    def test_step_valid_action(self, client):
        """Valid action must return step result."""
        client.post("/reset", json={})
        r = client.post("/step", json={
            "action": "do_nothing",
            "reasoning": "Testing baseline action"
        })
        assert r.status_code == 200
        data = r.json()
        assert "reward" in data
        assert "done" in data
        assert isinstance(data["reward"], float)

    def test_step_before_reset(self, client):
        """Step before reset must return 400."""
        fresh_app_client = TestClient(app)

        from run import env as global_env
        global_env.farm_state = None
        r = fresh_app_client.post("/step", json={"action": "do_nothing", "reasoning": ""})
        assert r.status_code == 400

    def test_full_episode(self, client):
        """Run a complete episode without crashes."""
        client.post("/reset", json={})
        done = False
        steps = 0
        while not done and steps < 100:
            r = client.post("/step", json={
                "action": random.choice([
                    "irrigate_medium", "do_nothing", "spray_pesticide",
                    "consult_district_advisor", "check_mandi_prices"
                ]),
                "reasoning": "Test reasoning with causal word because of conditions observed"
            })
            assert r.status_code == 200
            data = r.json()
            done = data["done"]
            steps += 1

        assert steps > 0, "Episode should have at least 1 step"
        assert data.get("episode_reward") is not None or steps < 90

    def test_state_endpoint(self, client):
        """State endpoint must return structured farm data."""
        client.post("/reset", json={})
        r = client.get("/state")
        assert r.status_code == 200
        data = r.json()
        assert "farm_state" in data
        assert "market_state" in data


class TestEnvironmentMechanics:

    def test_soil_health_persists(self, tmp_path, monkeypatch):
        """Soil health must persist across episode resets."""
        import env as env_module
        monkeypatch.setattr(env_module, "SOIL_PERSISTENCE_FILE",
                            str(tmp_path / ".soil_state.json"))
        e = KisanEnv()
        e.reset(seed=1)

        e.farm_state.soil_health = 0.45
        e._finalize_episode()

        e.reset(seed=2)
        assert abs(e.farm_state.soil_health - 0.45) < 0.01

    def test_insurance_deadline_enforced(self, env):
        """Insurance cannot be enrolled after Day 15."""
        for _ in range(15):
            env.farm_state.day += 1

        assert env.farm_state.day > 15
        result = resolve_action(
            "file_insurance_claim",
            env.farm_state,
            env.weather_sequence[15],
            env.market_agent,
            {}
        )

        if not env.farm_state.insurance_enrolled:
            assert not result.get("success", True)

    def test_budget_cannot_go_deeply_negative(self, env):
        """Budget protection: catastrophic debt ends episode."""
        env.farm_state.budget = -4999
        assert not env._check_done()
        env.farm_state.budget = -5001
        assert env._check_done()

    def test_episode_terminates_at_max_days(self, env):
        """Episode must terminate at day 90."""
        env.farm_state.day = 90
        assert env._check_done()

    def test_crop_death_terminates_episode(self, env):
        """Episode terminates if crop health hits 0."""
        env.farm_state.crop_health = 0.0
        assert env._check_done()

    def test_wrong_spray_penalized(self, farm_state):
        """Spraying pesticide during high fungal risk should produce warning."""
        farm_state.pest_pressure = 0.10
        farm_state.fungal_risk = 0.80
        market = MarketAgent()
        market.reset({})
        result = resolve_action(
            "spray_pesticide", farm_state,
            {"condition": "rain", "temp_c": 30, "humidity": 0.9, "rainfall_mm": 10},
            market, {}
        )

        assert "fungal" in result["message"].lower() or "fungicide" in result["message"].lower()

    def test_no_double_loan(self, farm_state):
        """Cannot take second loan while first is active."""
        farm_state.loan_balance = 5000
        market = MarketAgent()
        market.reset({})
        result = resolve_action("apply_for_loan", farm_state, {}, market, {})
        assert not result.get("success", True)


class TestRewardSystem:

    def test_reward_weights_sum_to_one(self):
        """Reward weights must sum exactly to 1.0."""
        from grader import REWARD_WEIGHTS
        total = sum(REWARD_WEIGHTS.values())
        assert abs(total - 1.0) < 0.001, f"Weights sum to {total}, not 1.0"

    def test_episode_reward_in_range(self, env):
        """Episode reward must be in [0, 1]."""
        engine = RewardEngine()
        env.farm_state.revenue_earned = 8000
        reward = engine.compute_episode_reward(
            farm_state=env.farm_state,
            episode_log=[],
            market_agent=env.market_agent,
            initial_soil_health=0.7,
        )
        assert 0.0 <= reward <= 1.0

    def test_step_reward_bounded(self, env):
        """Step rewards must be bounded to avoid dominating episode signal."""
        engine = RewardEngine()
        from dynamics import resolve_action
        market = MarketAgent()
        market.reset({})
        result = {"action_name": "do_nothing", "success": True, "budget_delta": 0}
        reward = engine.compute_step_reward(
            action="do_nothing",
            action_result=result,
            farm_state=env.farm_state,
            reasoning="Testing",
            oversight_score=0.7,
        )
        assert -0.5 <= reward <= 0.5

    def test_reasoning_scorer_empty(self):
        """Empty reasoning must return 0.0, not crash."""
        assert ReasoningScorer.score("") == 0.0
        assert ReasoningScorer.score(None) == 0.0

    def test_reasoning_scorer_causal(self):
        """Causal reasoning should score higher than terse output."""
        terse = "spray_fungicide"
        causal = (
            "I am applying fungicide because humidity has been above 85% for the past 3 days "
            "and temperature is 30C. This combination indicates high fungal risk. "
            "Crop stage 2 (flowering) is particularly susceptible. Previously I noted early "
            "edge-yellowing on leaves which is a fungal indicator, not pest damage. "
            "Therefore fungicide is the correct intervention rather than pesticide."
        )
        assert ReasoningScorer.score(causal) > ReasoningScorer.score(terse)
        assert ReasoningScorer.score(causal) > 0.5


class TestAgents:

    def test_market_agent_modes(self):
        """Market agent must transition through all modes."""
        agent = MarketAgent()
        agent.reset({})
        modes_seen = {agent.mode}
        for _ in range(200):
            agent.step(day=1)
            modes_seen.add(agent.mode)
        assert "FAIR" in modes_seen

        assert len(modes_seen) >= 2

    def test_district_advisor_preference_drift(self):
        """Advisor must change preference at Day 36."""
        advisor = DistrictFarmAdvisor()
        advisor.reset(episode_count=0)
        assert advisor.preference_state == "chemical"


        from dynamics import FarmState
        fs = FarmState()
        for day in range(1, 41):
            fs.day = day
            advisor.step(day=day, farm_state=fs)

        assert advisor.preference_state == "ipm"

    def test_climate_agent_difficulty_advances(self):
        """Climate agent difficulty must advance when performance > threshold."""
        agent = ClimateAgent()
        assert agent.current_difficulty == 1

        for _ in range(5):
            agent.update(0.75)
        assert agent.current_difficulty == 2

    def test_climate_agent_no_jump_two_levels(self):
        """Difficulty must not jump more than 1 level at a time."""
        agent = ClimateAgent()
        prev = agent.current_difficulty
        for _ in range(100):
            agent.update(0.90)
            assert abs(agent.current_difficulty - prev) <= 1
            prev = agent.current_difficulty


class TestActionParser:

    def test_valid_action_parse(self):
        """Standard format must parse correctly."""
        output = "ACTION: spray_fungicide\nREASONING: Fungal risk is high because humidity exceeded 85%."
        result = ActionParser.parse(output)
        assert result["action"] == "spray_fungicide"
        assert "fungal" in result["reasoning"].lower()

    def test_malformed_action_fallback(self):
        """Malformed output must fall back to do_nothing, not crash."""
        result = ActionParser.parse("I think we should maybe water the plants today")
        assert result["action"] in ActionParser.VALID_ACTIONS

    def test_none_input(self):
        """None input must return do_nothing safely."""
        result = ActionParser.parse(None)
        assert result["action"] == "do_nothing"

    def test_capitalized_action(self):
        """Capitalized action names must normalize correctly."""
        result = ActionParser.parse("ACTION: SPRAY_FUNGICIDE\nREASONING: test")
        assert result["action"] == "spray_fungicide"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "--color=yes"])

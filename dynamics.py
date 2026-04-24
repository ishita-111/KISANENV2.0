
import dataclasses
import random
import numpy as np
from typing import Dict, List, Optional, Any

@dataclasses.dataclass
class FarmState:
    day: int = 1
    crop_stage: int = 0
    crop_health: float = 0.85
    crop_variety: str = "cotton_desi"
    yield_accumulated: float = 0.0
    soil_moisture: float = 0.50
    soil_nitrogen: float = 0.60
    soil_phosphorus: float = 0.55
    soil_potassium: float = 0.65
    soil_health: float = 0.70
    pest_pressure: float = 0.08
    observed_pest_pressure: float = 0.10
    fungal_risk: float = 0.15
    budget: int = 15_000
    loan_balance: int = 0
    insurance_enrolled: bool = False
    insurance_claims_available: int = 1
    crop_sold_quintals: float = 0.0
    revenue_earned: int = 0
    weather_sequence: list = None
    last_action: str = "none"
    consecutive_same_actions: int = 0
    last_advisor_advice: str = ""

    def apply_action_result(self, result: Dict[str, Any]):
        self.budget += result.get("budget_delta", 0)
        self.soil_moisture = np.clip(self.soil_moisture + result.get("moisture_delta", 0), 0, 1)
        self.soil_nitrogen = np.clip(self.soil_nitrogen + result.get("nitrogen_delta", 0), 0, 1)
        self.crop_health = np.clip(self.crop_health + result.get("health_delta", 0), 0, 1)
        if result.get("pest_reduction"): self.pest_pressure = max(0, self.pest_pressure - result["pest_reduction"])
        if result.get("fungal_reduction"): self.fungal_risk = max(0, self.fungal_risk - result["fungal_reduction"])
        if result.get("insurance_enrolled"): self.insurance_enrolled = True
        if result.get("yield_harvested"): self.crop_sold_quintals += result["yield_harvested"]
        if result.get("revenue"):
            self.revenue_earned += result["revenue"]
            self.budget += result["revenue"]
        if result.get("loan_granted"):
            self.loan_balance += result["loan_granted"]
            self.budget += result["loan_granted"]
        if result.get("soil_health_delta"):
            self.soil_health = np.clip(self.soil_health + result["soil_health_delta"], 0.05, 1.0)
        
        self.last_action = result.get("action_name", "none")
        self.observed_pest_pressure = float(np.clip(self.pest_pressure + np.random.normal(0, 0.15), 0, 1))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "day": self.day, "crop_stage": self.crop_stage, "crop_health": round(self.crop_health, 3),
            "soil_moisture": round(self.soil_moisture, 3), "soil_nitrogen": round(self.soil_nitrogen, 3),
            "soil_health": round(self.soil_health, 3), "pest_pressure": round(self.observed_pest_pressure, 3),
            "fungal_risk": round(self.fungal_risk, 3), "budget": self.budget, "yield": round(self.yield_accumulated, 2),
            "sold": round(self.crop_sold_quintals, 2), "revenue": self.revenue_earned
        }

ACTION_COSTS = {
    "irrigate_low": 50, "irrigate_medium": 120, "irrigate_high": 250,
    "spray_pesticide": 400, "spray_fungicide": 350,
    "apply_fertilizer_low": 280, "apply_fertilizer_high": 580,
    "prune_crop": 150, "consult_district_advisor": 50, "call_soil_test": 200,
    "call_pest_advisory": 80, "call_satellite_imagery": 150, "check_insurance_portal": 0,
    "check_mandi_prices": 0, "apply_for_loan": 0, "file_insurance_claim": 0,
    "repay_loan": 0, "sell_crop_25pct": 0, "sell_crop_50pct": 0, "sell_crop_all": 0, "do_nothing": 0
}

def resolve_action(name, farm, weather, market, tools) -> Dict:
    cost = ACTION_COSTS.get(name, 0)
    if cost > farm.budget: return {"success": False, "message": "Insufficient budget"}
    
    res = {"action": name, "budget_delta": -cost, "success": True}
    
    if name.startswith("irrigate_"):
        levels = {"irrigate_low": 0.10, "irrigate_medium": 0.20, "irrigate_high": 0.35}
        res["moisture_delta"] = levels[name] * (0.4 if weather["condition"] == "rain" else 1.0)
    
    elif name == "spray_pesticide":
        res["pest_reduction"] = 0.45 if farm.pest_pressure > 0.4 else (0.20 if farm.pest_pressure > 0.2 else 0.05)
        if farm.pest_pressure < 0.2: res["soil_health_delta"] = -0.015
    
    elif name == "spray_fungicide":
        if farm.fungal_risk > 0.5:
            res.update({"fungal_reduction": 0.5, "health_delta": 0.05})
        else: res.update({"fungal_reduction": 0.1, "soil_health_delta": -0.01})
        
    elif "fertilizer" in name:
        high = "high" in name
        if farm.soil_nitrogen > 0.75: res["soil_health_delta"] = -0.02 if high else -0.005
        else: res.update({"nitrogen_delta": 0.25 if high else 0.12, "health_delta": 0.03 if high else 0.015})

    elif name.startswith("sell_crop_"):
        portions = {"sell_crop_25pct": 0.25, "sell_crop_50pct": 0.5, "sell_crop_all": 1.0}
        amt = farm.yield_accumulated * portions[name]
        if amt < 0.01: return {"success": False, "message": "No yield to sell"}
        price = market.get_actual_price()
        rev = int(amt * price)
        res.update({"yield_harvested": amt, "revenue": rev, "actual_price": price})

    elif name == "check_insurance_portal":
        if not farm.insurance_enrolled and farm.day <= 15 and farm.budget >= 500:
            res.update({"budget_delta": -500, "insurance_enrolled": True})
        res["tool_result"] = {"enrolled": farm.insurance_enrolled, "deadline": 15}

    elif name == "file_insurance_claim" and farm.insurance_enrolled and farm.insurance_claims_available > 0:
        amt = min(8000, abs(farm.budget - 15000))
        res["budget_delta"] = amt
        farm.insurance_claims_available = 0

    elif name == "apply_for_loan" and farm.loan_balance == 0:
        res["loan_granted"] = 5000

    elif name == "repay_loan" and farm.loan_balance > 0:
        total = int(farm.loan_balance * 1.18)
        if farm.budget >= total:
            res["budget_delta"] = -total
            farm.loan_balance = 0

    return res

class WeatherEngine:
    @classmethod
    def generate_sequence(cls, days, difficulty):
        seq = []
        for d in range(1, days + 1):
            rain_prob = 0.6 if 30 < d <= 65 else (0.15 if d <= 30 else 0.3)
            cond = "rain" if random.random() < rain_prob else "sunny"
            if difficulty >= 2 and random.random() < 0.05: cond = "storm"
            temp, hum = random.uniform(26, 38), random.uniform(0.5, 0.9)
            seq.append({"day": d, "condition": cond, "temp_c": round(temp, 1), "humidity": round(hum, 3)})
        return seq

    @staticmethod
    def compute_fungal_risk(hum, temp, stage, variety):
        risk = (hum - 0.75) * 2.5 if hum > 0.75 and temp > 27 else 0
        risk *= {0: 0.5, 1: 0.8, 2: 1.4, 3: 1.6, 4: 0.9}.get(stage, 1.0)
        return float(np.clip(risk, 0, 1))

class PestDynamics:
    @staticmethod
    def daily_spread(farm, weather, neighbor):
        rate = 0.01 + (0.015 if weather["temp_c"] > 30 and weather["humidity"] > 0.7 else 0)
        farm.pest_pressure = float(np.clip(farm.pest_pressure + rate, 0, 1))
        if farm.pest_pressure > 0.5: farm.crop_health = max(0, farm.crop_health - (farm.pest_pressure - 0.5) * 0.03)

class SoilChemistry:
    @staticmethod
    def daily_update(farm, weather):
        evap = 0.02 + (0.01 if weather["condition"] == "sunny" else -0.015)
        farm.soil_moisture = float(np.clip(farm.soil_moisture - evap, 0, 1))
        if farm.crop_health > 0.3: farm.soil_nitrogen = max(0, farm.soil_nitrogen - 0.005)

class CropGrowthModel:
    @classmethod
    def advance(cls, farm, weather):
        for i, t in enumerate([0, 15, 35, 55, 75]):
            if farm.day >= t: farm.crop_stage = i
        if farm.soil_moisture > 0.25 and farm.soil_nitrogen > 0.20:
            growth = 0.08 * farm.soil_health * farm.crop_health
            if farm.crop_stage >= 3: farm.yield_accumulated += growth * 0.15
        if farm.soil_moisture < 0.15: farm.crop_health = max(0, farm.crop_health - 0.02)

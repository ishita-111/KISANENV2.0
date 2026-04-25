
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
    misdiagnosis_penalty_active: bool = False
    soil_test_logged_day: int = -1

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
        if result.get("misdiagnosis"):
            self.misdiagnosis_penalty_active = True
        if result.get("soil_test_logged"):
            self.soil_test_logged_day = self.day
        if result.get("claim_used"):
            self.insurance_claims_available = 0
        
        self.last_action = result.get("action_name", "none")
        self.observed_pest_pressure = float(np.clip(self.pest_pressure + np.random.normal(0, 0.15), 0, 1))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "day": self.day, "crop_stage": self.crop_stage, "crop_health": round(float(self.crop_health), 3),
            "soil_moisture": round(float(self.soil_moisture), 3), "soil_nitrogen": round(float(self.soil_nitrogen), 3),
            "soil_health": round(float(self.soil_health), 3), "pest_pressure": round(float(self.observed_pest_pressure), 3),
            "fungal_risk": round(float(self.fungal_risk), 3), "budget": int(self.budget), "yield": round(float(self.yield_accumulated), 2),
            "sold": round(float(self.crop_sold_quintals), 2), "revenue": int(self.revenue_earned),
            "insurance_enrolled": bool(self.insurance_enrolled), "yield_accumulated": round(float(self.yield_accumulated), 2),
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
    if cost > farm.budget:
        return {"success": False, "message": "Insufficient budget", "action_name": name}
    
    res = {"action_name": name, "budget_delta": -cost, "success": True}
    
    if name.startswith("irrigate_"):
        levels = {"irrigate_low": 0.10, "irrigate_medium": 0.20, "irrigate_high": 0.35}
        res["moisture_delta"] = levels[name] * (0.4 if weather["condition"] == "rain" else 1.0)
    
    elif name == "spray_pesticide":
        res["pest_reduction"] = 0.45 if farm.pest_pressure > 0.4 else (0.20 if farm.pest_pressure > 0.2 else 0.05)
        if farm.fungal_risk > 0.6 and farm.pest_pressure < 0.35:
            res["misdiagnosis"] = True
            res["message"] = "WARNING: Fungal risk is high. Consider fungicide instead of pesticide."
            res["soil_health_delta"] = -0.015
        elif farm.pest_pressure < 0.2:
            res["soil_health_delta"] = -0.015
            res["message"] = "Pesticide applied."
        else:
            res["message"] = "Pesticide applied."
    
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
        if amt < 0.01: return {"success": False, "message": "No yield to sell", "action_name": name}
        price = market.get_actual_price()
        rev = int(amt * price)
        res.update({
            "yield_harvested": amt,
            "revenue": rev,
            "actual_price": price,
            "displayed_price": market.displayed_price
        })

    elif name == "check_insurance_portal":
        if not farm.insurance_enrolled and farm.day <= 15 and farm.budget >= 500:
            res.update({"budget_delta": -500, "insurance_enrolled": True})
        res["tool_result"] = {"enrolled": farm.insurance_enrolled, "deadline": 15}

    elif name == "check_mandi_prices":
        res["tool_result"] = {
            "actual_range": f"Rs.{market.get_actual_price() - 80:,} – Rs.{market.get_actual_price() + 80:,}",
            "signal": "FAIR" if market.mode == "FAIR" else "SUPPRESSED — exercise caution",
            "ready_in_days": 1
        }
        res["message"] = "Market intelligence requested. Report available tomorrow."

    elif name == "file_insurance_claim":
        if not farm.insurance_enrolled:
            return {"success": False, "message": "Not enrolled in insurance.", "action_name": name}
        if farm.insurance_claims_available <= 0:
            return {"success": False, "message": "No claims remaining.", "action_name": name}
        days_since_test = farm.day - farm.soil_test_logged_day
        if farm.soil_test_logged_day < 0 or days_since_test > 15:
            return {
                "success": False,
                "message": "Claim rejected: No recent soil test on record. Call soil test first as evidence.",
                "action_name": name
            }
        amt = min(8000, abs(farm.budget - 15000))
        res["budget_delta"] = amt
        res["claim_used"] = True
        res["message"] = f"Insurance claim approved: Rs.{amt:,} disbursed."

    elif name == "apply_for_loan":
        if farm.loan_balance > 0:
            return {"success": False, "message": "Existing loan must be repaid first.", "action_name": name}
        res["loan_granted"] = 5000
        res["message"] = "Loan of Rs.5,000 approved."

    elif name == "repay_loan" and farm.loan_balance > 0:
        total = int(farm.loan_balance * 1.18)
        if farm.budget >= total:
            res["budget_delta"] = -total
            farm.loan_balance = 0

    elif name == "call_soil_test":
        res["tool_result"] = {
            "nitrogen": round(farm.soil_nitrogen, 2),
            "phosphorus": round(farm.soil_phosphorus, 2),
            "potassium": round(farm.soil_potassium, 2),
            "soil_health": round(farm.soil_health, 2),
        }
        res["soil_test_logged"] = True

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
            rainfall_mm = random.uniform(5, 25) if cond == "rain" else (random.uniform(30, 60) if cond == "storm" else 0)
            seq.append({"day": d, "condition": cond, "temp_c": round(temp, 1), "humidity": round(hum, 3), "rainfall_mm": round(rainfall_mm, 1)})
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
        condition = weather["condition"]
        if condition == "rain":
            rainfall_gain = weather.get("rainfall_mm", 12) * 0.018
            net = rainfall_gain - 0.005
            farm.soil_moisture = float(np.clip(farm.soil_moisture + net, 0, 1))
        elif condition == "storm":
            farm.soil_moisture = float(np.clip(farm.soil_moisture + 0.22, 0, 1))
        else:
            evap = 0.025 if condition == "sunny" else 0.015
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

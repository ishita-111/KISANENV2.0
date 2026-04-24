
"""
Simple rule-based heuristic agent for KisanEnv 2.0 baseline comparison.
This agent uses hard-coded rules.
It provides a baseline benchmark for the trained agent.
"""


class HeuristicAgent:
    """
    Rule-based agricultural decision agent.
    Returns formatted actions from a fixed policy over the farm state.
    """

    def decide(self, farm_state: dict) -> str:
        """Given farm state dict, return formatted action string."""
        day = farm_state.get("day", 1)
        moisture = farm_state.get("soil_moisture", 0.5)
        pest = farm_state.get("pest_pressure_observed", 0.1)
        fungal = farm_state.get("fungal_risk", 0.1)
        budget = farm_state.get("budget", 15000)
        insurance = farm_state.get("insurance_enrolled", False)
        crop_health = farm_state.get("crop_health", 0.8)


        if not insurance and day <= 14 and budget >= 500:
            action = "check_insurance_portal"
            reason = f"Insurance window open (Day {day}). Checking status before deadline."


        elif moisture < 0.35:
            action = "irrigate_medium"
            reason = f"Soil moisture critically low ({moisture:.0%}). Irrigation needed."


        elif fungal > 0.65:
            action = "spray_fungicide"
            reason = f"Fungal risk at {fungal:.0%}. Applying fungicide."


        elif pest > 0.55:
            action = "spray_pesticide"
            reason = f"Pest pressure at {pest:.0%}. Applying pesticide."


        elif 20 <= day <= 50 and farm_state.get("soil_nitrogen", 0.6) < 0.40:
            action = "apply_fertilizer_low"
            reason = f"Nitrogen depleted ({farm_state.get('soil_nitrogen', 0):.0%}). Applying low-dose fertilizer."


        elif day >= 75 and farm_state.get("yield_accumulated", 0) > 3.0:
            action = "sell_crop_50pct"
            reason = f"Late season (Day {day}). Selling 50% of accumulated yield."


        else:
            action = "do_nothing"
            reason = f"Conditions stable on Day {day}. No immediate intervention needed."

        return f"ACTION: {action}\nREASONING: {reason}"

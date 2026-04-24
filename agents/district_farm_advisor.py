
class DistrictFarmAdvisor:
    def __init__(self):
        self.personality = "conservative"
        self.advice_history = []

    def reset(self, episode_count):
        self.advice_history = []
        self.personality = "conservative" if episode_count % 2 == 0 else "progressive"

    def step(self, day, farm_state):
        advice = "No urgent guidance today."
        
        if farm_state.soil_moisture < 0.3:
            advice = "Low soil moisture detected. Consider medium or high irrigation to prevent root stress."
        elif farm_state.pest_pressure > 0.4:
            advice = "Pest pressure rising. Recommend pesticide application if monitoring shows continued spread."
        elif farm_state.day < 14 and not farm_state.insurance_enrolled:
            advice = "Insurance enrollment window is open. Protecting your capital early is highly recommended."

        return {"advice": advice}

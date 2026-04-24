
import random

class MarketAgent:
    def __init__(self):
        self.mode = "FAIR"
        self.displayed_price = 5800
        self.price_trend_signal = "STABLE"
        self.public_message = "Market operating normally."

    def reset(self, scenario=None):
        self.mode = "FAIR"
        self.displayed_price = 5800
        self.price_trend_signal = "STABLE"

    def step(self, day):
        # Basic logic: 20% chance to manipulate after day 30
        if day > 30 and random.random() < 0.2:
            self.mode = "MANIPULATING"
            self.displayed_price = random.randint(5200, 5500)
            self.price_trend_signal = "VOLATILE"
            self.public_message = "Regional supply glut reported."
        else:
            self.mode = "FAIR"
            self.displayed_price = random.randint(5700, 6100)
            self.price_trend_signal = "STABLE"
            self.public_message = "Normal market liquidity."

    def get_actual_price(self):
        if self.mode == "MANIPULATING":
            return self.displayed_price + random.randint(300, 600)
        return self.displayed_price + random.randint(-50, 50)

    def to_dict(self):
        return {"mode": self.mode, "price": self.displayed_price, "trend": self.price_trend_signal}

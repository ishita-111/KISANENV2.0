
import re
import os
from typing import Dict, Any, Optional

class ActionParser:
    VALID_ACTIONS = {
        "irrigate_low", "irrigate_medium", "irrigate_high",
        "spray_pesticide", "spray_fungicide",
        "apply_fertilizer_low", "apply_fertilizer_high",
        "prune_crop", "sell_crop_25pct", "sell_crop_50pct", "sell_crop_all",
        "consult_district_advisor", "call_soil_test", "call_pest_advisory",
        "call_satellite_imagery", "check_insurance_portal", "check_mandi_prices",
        "apply_for_loan", "file_insurance_claim", "repay_loan", "do_nothing",
    }

    @classmethod
    def parse(cls, llm_output: str) -> Dict[str, str]:
        if not llm_output: return {"action": "do_nothing", "reasoning": ""}
        action, reasoning = None, ""
        for line in llm_output.strip().split("\n"):
            line = line.strip()
            if line.upper().startswith("ACTION:"):
                action = cls._normalize_action(line.split(":", 1)[1])
            elif line.upper().startswith("REASONING:"):
                reasoning = line.split(":", 1)[1].strip()
        if action not in cls.VALID_ACTIONS:
            action = cls._fuzzy_match(action or "")
        return {"action": action or "do_nothing", "reasoning": reasoning}

    @classmethod
    def _normalize_action(cls, raw: str) -> str:
        res = re.sub(r"[^a-z0-9_]", "_", raw.lower().strip())
        return re.sub(r"_+", "_", res).strip("_")

    @classmethod
    def _fuzzy_match(cls, raw: str) -> str:
        raw_lower = raw.lower()
        mappings = {"irrigate": "irrigate_medium", "spray": "spray_pesticide", "fertilize": "apply_fertilizer_low", "sell": "sell_crop_50pct"}
        for k, v in mappings.items():
            if k in raw_lower: return v
        return "do_nothing"

class LLMClient:
    def __init__(self, model=None, tokenizer=None):
        self.backend = os.environ.get("KISANENV_LLM_BACKEND", "rule_based")
        self.model, self.tokenizer = model, tokenizer
        self._farmer_agent = None
        if self.backend == "rule_based":
            from agents.farmer_agent import FarmerAgent
            self._farmer_agent = FarmerAgent()
            self._farmer_agent.load()

    @property
    def farmer_agent(self): return self._farmer_agent

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        if self.backend == "rule_based": return self._rule_based_response(prompt)
        if self.backend == "huggingface": return self._hf_generate(prompt, max_new_tokens)
        if self.backend == "openai": return self._openai_generate(prompt)
        return "ACTION: do_nothing\nREASONING: Backend not configured."

    def _rule_based_response(self, prompt: str) -> str:
        state = self._extract_farm_state(prompt)
        action, reasoning = self.farmer_agent.select_action(state)
        return f"ACTION: {action}\nREASONING: {reasoning}"

    def _extract_farm_state(self, prompt: str) -> dict:
        state = {"day": 1, "soil_moisture": 0.5, "budget": 15000}
        d_match = re.search(r"Day\s+(\d+)", prompt)
        if d_match: state["day"] = int(d_match.group(1))
        m_match = re.search(r"Moisture\s+(\d+)%", prompt)
        if m_match: state["soil_moisture"] = int(m_match.group(1)) / 100
        b_match = re.search(r"Rs\.\s*([\d,]+)", prompt)
        if b_match: state["budget"] = int(b_match.group(1).replace(",", ""))
        state["insurance_enrolled"] = "ENROLLED" in prompt and "NOT ENROLLED" not in prompt
        return state

    def _hf_generate(self, prompt: str, max_new_tokens: int) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.7, do_sample=True)
        return self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    def _openai_generate(self, prompt: str) -> str:
        import openai
        resp = openai.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}])
        return resp.choices[0].message.content

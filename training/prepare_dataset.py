
import json
import os
from typing import List, Dict

def prepare_grpo_dataset(input_file: str, output_file: str):
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    with open(input_file, "r") as f:
        data = json.load(f)

    prompts = []
    seen_prompts = set()

    for episode in data:
        steps = episode.get("steps", [])
        for step in steps:
            prompt_text = step.get("prompt")
            if prompt_text and prompt_text not in seen_prompts:
                prompts.append({"prompt": prompt_text})
                seen_prompts.add(prompt_text)

    if not prompts:
        from env import KisanEnv
        env = KisanEnv()
        for _ in range(50):
            obs = env.reset()
            prompts.append({"prompt": obs["prompt"]})
            for _ in range(3):
                obs, _, _, _ = env.step("ACTION: do_nothing\nREASONING: monitoring.")
                prompts.append({"prompt": obs["prompt"]})

    with open(output_file, "w") as f:
        json.dump(prompts, f, indent=2)
    
    print(f"Dataset prepared: {len(prompts)} prompts saved.")

if __name__ == "__main__":
    prepare_grpo_dataset("episode_rewards.json", "training/prompts.json")


import os
import torch
import json
from typing import List, Dict, Any

try:
    from unsloth import FastLanguageModel
    from trl import GRPOTrainer, GRPOConfig
except ImportError:
    print("Warning: Unsloth or TRL not installed.")

from env import KisanEnv
from inference import ActionParser
from grader import ReasoningScorer

def reward_function(completions, prompts=None, **kwargs) -> List[float]:
    rewards = []
    for content in completions:
        env = KisanEnv()
        env.reset()
        parsed = ActionParser.parse(content)
        reasoning = parsed.get("reasoning", "")
        try:
            obs, reward, done, info = env.step(content)
            if done:
                ep_reward = info.get("episode_reward", reward)
                reward = ep_reward
        except Exception as e:
            reward = -0.1
        reasoning_bonus = ReasoningScorer.score(reasoning) * 0.04
        rewards.append(float(reward) + reasoning_bonus)
    return rewards

def main():
    model_name = "unsloth/Qwen2.5-3B-Instruct"
    max_seq_length = 1024
    output_dir = "kisanenv-qwen-grpo"

    print(f"Loading {model_name}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        load_in_4bit = True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
    )

    training_args = GRPOConfig(
        output_dir = output_dir,
        learning_rate = 5e-5,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.1,
        warmup_ratio = 0.1,
        lr_scheduler_type = "cosine",
        logging_steps = 1,
        bf16 = True,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,
        num_generations = 4,
        max_prompt_length = 512,
        max_completion_length = 256,
        max_steps = 250,
        save_steps = 50,
    )

    from datasets import Dataset

    if os.path.exists("training/prompts.json"):
        with open("training/prompts.json", "r") as f:
            raw = json.load(f)
    else:
        # Fallback: generate prompts from env
        raw = []
        gen_env = KisanEnv()
        for _ in range(60):
            obs = gen_env.reset()
            raw.append({"prompt": obs["prompt"]})
            for _ in range(4):
                _, _, done, _ = gen_env.step("ACTION: do_nothing\nREASONING: monitoring.")
                if not done:
                    raw.append({"prompt": gen_env._build_observation(None, 0.0)["prompt"]})
                else:
                    break

    dataset = Dataset.from_list(raw)

    trainer = GRPOTrainer(
        model = model,
        reward_funcs = [reward_function],
        args = training_args,
        train_dataset = dataset,
    )

    print("Starting GRPO training loop...")
    trainer.train()

    model.save_pretrained_merged(output_dir, tokenizer, save_method = "lora")
    print(f"Training complete. Saved to {output_dir}")

if __name__ == "__main__":
    main()

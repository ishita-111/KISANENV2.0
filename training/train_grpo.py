import os
import json
import re as _re
import torch
from typing import List

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
        total_reward = 0.0

        # Reward 1: Format (0 or 1) — 35% weight
        format_match = bool(_re.search(
            r"ACTION:\s*([\w_]+)\s*\nREASONING:\s*(.+)",
            content,
            _re.DOTALL | _re.IGNORECASE
        ))
        total_reward += (1.0 if format_match else 0.0) * 0.35

        if not format_match:
            rewards.append(total_reward)
            continue

        # Reward 2: Reasoning quality — 25% weight
        parsed = ActionParser.parse(content)
        reasoning = parsed.get("reasoning", "")
        total_reward += ReasoningScorer.score(reasoning) * 0.25

        # Reward 3: Environment (multi-step) — 40% weight
        try:
            env = KisanEnv()
            env.reset()
            _, step_r, done, info = env.step(content)
            cumulative = step_r

            if not done:
                from training.heuristic_baseline import HeuristicAgent
                heuristic = HeuristicAgent()
                for _ in range(9):
                    if done:
                        break
                    h_act = heuristic.decide(env.farm_state.to_dict())
                    _, r, done, info = env.step(h_act)
                    cumulative += r * 0.7

            env_reward = info.get("episode_reward", cumulative / 10.0) if done else cumulative / 10.0

        except Exception as e:
            print(f"Env error: {e}")
            env_reward = 0.0

        total_reward += env_reward * 0.40
        rewards.append(float(total_reward))

    return rewards


def main():
    model_name = "unsloth/Qwen2.5-3B-Instruct"
    max_seq_length = 1024
    output_dir = "kisanenv-qwen-grpo"

    print(f"Loading {model_name}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=2e-5,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=5,
        bf16=False,
        fp16=False,
        optim="adamw_8bit",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=6,
        max_steps=300,
        save_steps=50,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    # Dataset — expects prompts already formatted with chat template
    # Run Fix 1 cell in Colab BEFORE running this script
    if os.path.exists("training/prompts.json"):
        with open("training/prompts.json", "r") as f:
            raw = json.load(f)
        print(f"Loaded {len(raw)} prompts.")
    else:
        raise FileNotFoundError(
            "training/prompts.json not found. "
            "Run the dataset generation cell in Colab first."
        )

    from datasets import Dataset
    dataset = Dataset.from_list(raw)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_function],
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    torch.cuda.empty_cache()
    print(f"Starting GRPO training on {len(dataset)} prompts...")
    print("="*50)
    stats = trainer.train()
    print("="*50)
    print(f"Training complete. Final loss: {stats.training_loss:.4f}")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved LoRA adapter to {output_dir}/")


if __name__ == "__main__":
    main()
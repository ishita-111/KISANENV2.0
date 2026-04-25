import os
import json
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
        env = KisanEnv()
        env.reset()
        parsed = ActionParser.parse(content)
        reasoning = parsed.get("reasoning", "")
        try:
            _, reward, done, info = env.step(content)
            if done:
                reward = info.get("episode_reward", reward)
        except Exception as e:
            print(f"Reward error: {e}")
            reward = -0.1
        rewards.append(float(reward) + ReasoningScorer.score(reasoning) * 0.04)
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
        learning_rate=5e-5,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=5,                         # FIX 3: was 1, too noisy
        bf16=True,
        fp16=False,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=4,
        max_prompt_length=512,
        max_completion_length=256,
        max_steps=250,
        save_steps=50,
        dataloader_num_workers=0,                
        remove_unused_columns=False,           
    )

    
    from datasets import Dataset

    if os.path.exists("training/prompts.json"):
        print("Loading existing prompts from training/prompts.json...")
        with open("training/prompts.json", "r") as f:
            raw = json.load(f)
    else:
        print("No prompts.json found. Generating from KisanEnv...")
        raw = []
        gen_env = KisanEnv()
        for ep in range(60):
            obs = gen_env.reset()
            raw.append({"prompt": obs["prompt"]})
            for _ in range(4):
                obs, _, done, _ = gen_env.step(
                    "ACTION: do_nothing\nREASONING: data collection."
                )
                if not done:
                    raw.append({"prompt": obs["prompt"]})
                else:
                    break
        os.makedirs("training", exist_ok=True)
        with open("training/prompts.json", "w") as f:
            json.dump(raw, f)
        print(f"Generated and saved {len(raw)} prompts.")

    dataset = Dataset.from_list(raw)
    print(f"Dataset ready: {len(dataset)} prompts")

    
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_function],
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,                    
    )

    torch.cuda.empty_cache()                    
    print("Starting GRPO training...")
    stats = trainer.train()
    print(f"Training complete. Final loss: {stats.training_loss:.4f}")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved LoRA adapter to {output_dir}/")


if __name__ == "__main__":
    main()
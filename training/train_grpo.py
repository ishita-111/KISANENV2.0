
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

def reward_function(completions, **kwargs) -> List[float]:
    env = KisanEnv()
    rewards = []
    
    for content in completions:
        parsed = ActionParser.parse(content)
        action = parsed.get("action", "do_nothing")
        reasoning = parsed.get("reasoning", "")
        
        obs, reward, done, info = env.step(content)
        
        if len(reasoning.split()) > 15:
            reward += 0.02
            
        if any(w in reasoning.lower() for w in ["moisture", "pest", "budget", "price"]):
            reward += 0.01
            
        rewards.append(float(reward))
        if done: env.reset()
        
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

    if os.path.exists("training/prompts.json"):
        with open("training/prompts.json", "r") as f:
            dataset = json.load(f)
    else:
        dataset = [{"prompt": "The farm is dry (30% moisture). What should I do?"}]

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


import json
import os
import time
from datetime import datetime

REWARDS_FILE = os.path.join(os.path.dirname(__file__), "episode_rewards.json")

def save_episode(episode_num, reward, difficulty, extra_data=None):
    data = {"episodes": [], "created": datetime.now().isoformat()}
    entry = {
        "episode": episode_num,
        "reward": round(float(reward), 4),
        "difficulty": int(difficulty),
        "timestamp": datetime.now().isoformat(),
        "extra_data": extra_data or {}
    }

    for attempt in range(5):
        try:
            if os.path.exists(REWARDS_FILE):
                with open(REWARDS_FILE, "r") as f:
                    content = f.read()
                    if content: data = json.loads(content)
                    if "episodes" not in data: data["episodes"] = []
            
            data["episodes"].append(entry)
            with open(REWARDS_FILE, "w") as f:
                json.dump(data, f, indent=2)
            break
        except:
            time.sleep(0.05 * (attempt + 1))

    print(f"Episode {episode_num}: reward={reward:.4f}, difficulty={difficulty}")

def load_rewards():
    if not os.path.exists(REWARDS_FILE): return []
    try:
        with open(REWARDS_FILE, "r") as f:
            data = json.load(f)
        return [float(ep["reward"]) for ep in data.get("episodes", [])]
    except: return []

def get_stats():
    rewards = load_rewards()
    if not rewards:
        return {"total_episodes": 0, "best_reward": 0.0, "worst_reward": 0.0, "average_last_5": 0.0, "best_streak_above_0_44": 0}

    best_reward, worst_reward = max(rewards), min(rewards)
    avg_last_5 = sum(rewards[-5:]) / min(len(rewards), 5)

    streak = best_streak = 0
    for r in rewards:
        if r > 0.44:
            streak += 1
            best_streak = max(best_streak, streak)
        else: streak = 0

    return {
        "total_episodes": len(rewards),
        "best_reward": round(best_reward, 4),
        "worst_reward": round(worst_reward, 4),
        "average_last_5": round(avg_last_5, 4),
        "best_streak_above_0_44": best_streak
    }

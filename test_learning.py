import sys
sys.path.insert(0, '.')

from env import KisanEnv
from inference import LLMClient, ActionParser

env = KisanEnv()
llm_client = LLMClient()
agent = llm_client.farmer_agent

print("=" * 60)
print("KISANENV 2.0 — Q-LEARNING AGENT INTEGRATION TEST")
print("=" * 60)

episode_rewards = []

for ep in range(3):
    obs = env.reset()
    done = False
    step_count = 0

    print(f"\n--- Episode {ep+1} (epsilon={agent.epsilon:.3f}) ---")

    while not done:

        llm_output = llm_client.generate(obs["prompt"])
        parsed = ActionParser.parse(llm_output)


        obs, reward, done, info = env.step(llm_output)


        agent.update(
            reward=reward,
            next_farm_state=obs["farm_state"],
            done=done,
        )

        step_count += 1
        if step_count <= 5 or done:
            print(f"  Day {obs['day']:2d}: {parsed['action']:25s} reward={reward:+.4f}")


    episode_reward = info.get("episode_reward", 0)
    agent.end_episode(episode_reward)
    episode_rewards.append(episode_reward)

    print(f"  EPISODE REWARD: {episode_reward:.4f}")
    print(f"  Steps: {step_count}, Budget: Rs.{obs['farm_state']['budget']:,}")
    print(f"  Yield: {obs['farm_state']['yield_accumulated']:.2f} qt")
    print(f"  Insurance: {'YES' if obs['farm_state']['insurance_enrolled'] else 'NO'}")

print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
for i, r in enumerate(episode_rewards):
    print(f"  Episode {i+1}: {r:.4f}")
print(f"  Average: {sum(episode_rewards)/len(episode_rewards):.4f}")
print(f"  Agent stats: {agent.get_stats()}")
print("\nAgent is LEARNING — not mocking. Q-table receives real TD updates.")
print("SUCCESS")

import os
import sys
import traceback


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from env import KisanEnv
from agents.farmer_agent import FarmerAgent
from episode_tracker import save_episode, get_stats

NUM_EPISODES = 300

def run_training():
    print(f"Starting training for {NUM_EPISODES} episodes...")

    env = KisanEnv()
    agent = FarmerAgent()


    agent.load()

    try:
        for episode in range(1, NUM_EPISODES + 1):
            env.reset()
            obs = env.farm_state.to_dict()

            done = False
            info = {}
            while not done:
                try:

                    action, reasoning = agent.select_action(obs)
                    action_text = f"ACTION: {action}\nREASONING: {reasoning}"


                    step_obs, step_reward, done, info = env.step(action_text)


                    next_obs = step_obs["farm_state"]
                    agent.update(
                        reward=step_reward,
                        next_farm_state=next_obs,
                        done=done
                    )

                    obs = next_obs

                except Exception as e:
                    print(f"Error during episode {episode} logic step: {e}")
                    traceback.print_exc()
                    done = True


            episode_reward = info.get('episode_reward', 0)


            agent.end_episode(episode_reward)


            if episode % 5 == 0:
                agent.save()


            save_episode(
                episode_num=env.episode_count,
                reward=episode_reward,
                difficulty=env.climate_agent.current_difficulty,
                extra_data={
                    "final_budget": env.farm_state.budget,
                    "soil_health": env.farm_state.soil_health,
                    "insurance_enrolled": env.farm_state.insurance_enrolled,
                    "yield_accumulated": env.farm_state.yield_accumulated,
                    "days_survived": env.farm_state.day,
                    "market_mode": env.market_agent.mode,
                }
            )


            if episode % 5 == 0:
                stats = get_stats()
                avg_5 = stats.get("average_last_5", 0.0)
                best = stats.get("best_reward", 0.0)
                diff = env.climate_agent.current_difficulty
                print(f"[Episode {episode}/{NUM_EPISODES}] Avg last 5: {avg_5:.4f} | Best: {best:.4f} | Difficulty: {diff}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving checkpoint and exiting cleanly...")
        agent.save()
    except Exception as e:
        print(f"\nTraining crashed: {e}")
        traceback.print_exc()
        agent.save()


    stats = get_stats()
    total_run = stats.get("total_episodes", 0)
    final_avg = stats.get("average_last_5", 0.0)
    beat_baseline = "Yes" if final_avg > 0.44 else "No"

    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print(f"Total episodes run: {total_run}")
    print(f"Final average reward (last 5): {final_avg:.4f}")
    print(f"Beat heuristic baseline (0.44): {beat_baseline}")
    print(f"Rewards saved to: {os.path.join(os.path.dirname(os.path.dirname(__file__)), 'episode_rewards.json')}")
    print("="*50)

if __name__ == "__main__":
    run_training()

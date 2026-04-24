
import json
import asyncio
from typing import Optional, Dict, Any
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env import KisanEnv
from inference import LLMClient, ActionParser
from episode_tracker import save_episode

app = FastAPI(title="KisanEnv 2.0 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

env = KisanEnv()
llm_client = LLMClient()

try:
    app.mount("/ui", StaticFiles(directory="ui", html=True), name="ui")
except:
    pass

class StepRequest(BaseModel):
    action: str
    reasoning: Optional[str] = ""

class ResetRequest(BaseModel):
    seed: Optional[int] = None
    difficulty: Optional[int] = None

@app.get("/health")
async def health():
    return {"status": "ok", "version": "2.0.0"}

@app.post("/reset")
async def reset(request: ResetRequest):
    if request.difficulty:
        env.climate_agent.current_difficulty = request.difficulty
    obs = env.reset(seed=request.seed)
    return {"observation": obs["prompt"], "day": obs["day"], "farm_state": obs["farm_state"]}

@app.post("/step")
async def step(request: StepRequest):
    action_text = f"ACTION: {request.action}\nREASONING: {request.reasoning}"
    obs, reward, done, info = env.step(action_text)

    if llm_client.backend == "rule_based" and llm_client.farmer_agent:
        llm_client.farmer_agent.update(reward, obs["farm_state"], done)
        if done:
            llm_client.farmer_agent.end_episode(info.get("episode_reward", reward))

    return {"observation": obs["prompt"], "reward": reward, "done": done, "day": obs["day"]}

@app.get("/state")
async def get_state():
    if not env.farm_state: return {"status": "not_started"}
    return {"farm_state": env.farm_state.to_dict(), "info": env.get_info()}

@app.post("/ai_step")
async def ai_step():
    obs = env._build_observation(None, 0.0)
    llm_output = llm_client.generate(obs["prompt"])
    parsed = ActionParser.parse(llm_output)
    
    obs, reward, done, info = env.step(llm_output)
    
    if llm_client.backend == "rule_based" and llm_client.farmer_agent:
        llm_client.farmer_agent.update(reward, obs["farm_state"], done)
        if done:
            llm_client.farmer_agent.end_episode(info.get("episode_reward", reward))

    return {"action": parsed["action"], "reasoning": parsed["reasoning"], "reward": reward, "done": done}

@app.websocket("/ws/stream")
async def stream_episode(websocket: WebSocket):
    await websocket.accept()
    try:
        msg = await asyncio.wait_for(websocket.receive_text(), timeout=10.0)
        config = json.loads(msg)
        max_episodes = min(int(config.get("max_episodes", 3)), 50)
        agent = llm_client.farmer_agent if llm_client.backend == "rule_based" else None

        for _ in range(max_episodes):
            env.reset()
            await websocket.send_json({"type": "reset", "state": env.farm_state.to_dict()})

            done = False
            while not done:
                obs = env._build_observation(None, 0.0)
                output = llm_client.generate(obs["prompt"])
                parsed = ActionParser.parse(output)
                
                obs, reward, done, info = env.step(output)
                if agent: agent.update(reward, obs["farm_state"], done)

                await websocket.send_json({
                    "type": "step",
                    "day": obs["day"],
                    "action": parsed["action"],
                    "reasoning": parsed["reasoning"],
                    "reward": round(reward, 4),
                    "done": done
                })
                await asyncio.sleep(0.3)

            ep_reward = info.get("episode_reward", 0)
            if agent: agent.end_episode(ep_reward)
            save_episode(env.episode_count, ep_reward, env.climate_agent.current_difficulty)

            await websocket.send_json({"type": "episode_complete", "reward": round(ep_reward, 4)})
            await asyncio.sleep(1.0)

    except: pass
    finally: await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("run:app", host="0.0.0.0", port=8000, reload=True)

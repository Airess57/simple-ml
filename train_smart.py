import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
import os

os.makedirs("./logs/best_model/", exist_ok=True)
os.makedirs("./tensorboard_logs/", exist_ok=True)

env = gym.make("LunarLander-v3")
eval_env = gym.make("LunarLander-v3")

model = PPO.load("lunar_ai", env=env, tensorboard_log="./tensorboard_logs/")

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./logs/best_model/",
    log_path="./logs/",
    eval_freq=10000,
    deterministic=True,
    render=False
)

model.learn(
    total_timesteps=100_000,
    tb_log_name="PPO_LunarLander_Continued",
    callback=eval_callback,
    reset_num_timesteps=False
)

model.save("lunar_ai_final")
env.close()
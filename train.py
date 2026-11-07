
import gymnasium as gym
from stable_baselines3 import PPO

# Create environment
env = gym.make("LunarLander-v3")

# Create AI agent with TensorBoard logging
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.0003,
    tensorboard_log="./tensorboard_logs/"
)

# Train and record metrics
model.learn(
    total_timesteps=300_000,
    tb_log_name="PPO_LunarLander"
)

# Save as a new model (preserve the old one)
model.save("lunar_ai_tensorboard")

env.close()

print("Training completed!")

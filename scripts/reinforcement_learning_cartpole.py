import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")
obs, info = env.reset()

episode_reward = 0  # track reward per episode
episode = 1

for i in range(10000):  # total steps across all episodes
    pole_angle = obs[2]
    pole_angular_velocity = obs[3]

    # Simple rule-based controller:
    if pole_angle > 0:
        if pole_angular_velocity < 0:
            action = 1  # push right
        else:
            action = 0  # push left
    else:
        if pole_angular_velocity > 0:
            action = 1  # push right
        else:
            action = 0  # push left

    obs, reward, done, truncated, info = env.step(action)
    episode_reward += reward  # accumulate reward for this episode

    if done or truncated:
        print(f"Episode {episode}: total reward = {episode_reward:.0f}")
        episode += 1
        episode_reward = 0
        obs, info = env.reset()

env.close()

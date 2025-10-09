import gymnasium as gym

env = gym.make("MountainCar-v0", render_mode="human")
obs, info = env.reset(seed=0)

episodes = 5
for ep in range(1, episodes+1):
    obs, info = env.reset()
    done = False
    ep_reward = 0.0
    while not done:
        pos, vel = obs

        if vel <= 0 and pos <= -0.55:
            action = 0
        elif vel > 0 and pos > -0.55:
            action = 2
        else:
            action = 1

        print('{:.2f}, {}'.format(pos, action))

        obs, r, terminated, truncated, info = env.step(action)
        ep_reward += r
        done = terminated or truncated

    print(f"Episode {ep}: total reward = {ep_reward:.1f}")

env.close()

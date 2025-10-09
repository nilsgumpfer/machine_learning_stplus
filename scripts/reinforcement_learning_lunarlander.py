import gymnasium as gym

env = gym.make("LunarLander-v3", render_mode="human")
obs, info = env.reset(seed=0)

def heuristic_action(s):
    """
    s = [x, y, vx, vy, theta, vtheta, left_contact, right_contact]
    Actions: 0=do nothing, 1=fire left, 2=fire main, 3=fire right
    Very simple: keep upright, slow down horizontally/vertically, cut thrust on touchdown.
    """
    x, y, vx, vy, theta, vtheta, left_c, right_c = s
    # If touching ground, no thrust
    if left_c or right_c:
        return 0

    # Keep upright: if tilted right (theta>0), fire right engine to rotate left, and vice versa
    turn = 0
    if theta > 0.05 or vtheta > 0.1:
        turn = 3   # fire right engine to rotate left
    elif theta < -0.05 or vtheta < -0.1:
        turn = 1   # fire left engine to rotate right

    # Control horizontal drift: if drifting right (vx>0), tilt left; drifting left (vx<0), tilt right
    if vx > 0.5:  turn = 3
    if vx < -0.5: turn = 1

    # Control vertical speed: if falling fast, fire main engine
    if vy < -0.5:
        return 2   # main engine has priority to slow descent

    # Slight thrust when high and descending
    if y > 0.5 and vy < -0.2:
        return 2

    return turn if turn != 0 else 0

episodes = 5
for ep in range(1, episodes + 1):
    obs, info = env.reset()
    done = False
    ep_reward = 0.0
    while not done:
        action = heuristic_action(obs)
        obs, r, terminated, truncated, info = env.step(action)
        ep_reward += r
        done = terminated or truncated
    print(f"Episode {ep}: total reward = {ep_reward:.1f}")

env.close()

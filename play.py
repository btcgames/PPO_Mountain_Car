import gym
import torch
import numpy as np
import argparse
import time
from lib import model

FPS = 25

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', required=True, help='Model file to load')
    args = parser.parse_args()

    env = gym.make("MountainCarContinuous-v0")

    net = model.ModelActor(env.observation_space.shape[0], env.action_space.shape[0])
    net.load_state_dict(torch.load(args.model, map_location=torch.device("cpu")))
    net.eval()

    for ep in range(5):
        obs = env.reset()
        total_reward = 0.0
        
        while True:
            starts_ts = time.time()

            # GUI
            env.render() 

            obs_v = torch.FloatTensor([obs])
            with torch.no_grad():
                mu_v = net(obs_v)
            action = mu_v.squeeze(dim=0).data.cpu().numpy()
            action = np.clip(action, -1, 1)

            obs, reward, done, _ = env.step(action)
            
            total_reward += reward
            if done:
                print(f"Ep {ep+1}, total_reward: {total_reward}")
                break

            delta = 1/FPS - (time.time() - starts_ts)
            if delta > 0:
                time.sleep(delta)
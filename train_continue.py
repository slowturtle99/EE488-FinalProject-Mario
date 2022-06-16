"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""

import os

os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import torch
from src.env import MultipleEnvironments
from src.model import SimpleCNN2
from src.process import eval
import torch.multiprocessing as _mp
from torch.distributions import Categorical
import torch.nn.functional as F
import numpy as np
import shutil
from src.utils import Logger, WorldStageSelector

import time

def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of model described in the paper: Proximal Policy Optimization Algorithms for Super Mario Bros""")
    parser.add_argument("--action_type", type=str, default="simple")
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--gamma', type=float, default=0.90, help='discount factor for rewards')
    parser.add_argument('--tau', type=float, default=0.95, help='parameter for GAE')
    parser.add_argument('--beta', type=float, default=0.005, help='entropy coefficient')
    parser.add_argument('--epsilon', type=float, default=0.2, help='parameter for Clipped Surrogate Objective')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument("--num_local_steps", type=int, default=512)
    parser.add_argument("--num_global_steps", type=int, default=5e6)
    parser.add_argument("--num_processes", type=int, default=8)
    parser.add_argument("--save_interval", type=int, default=250, help="Number of steps between savings")
    parser.add_argument("--max_actions", type=int, default=200, help="Maximum repetition steps in test phase")
    parser.add_argument("--log_path", type=str, default="tensorboard/ppo_super_mario_bros")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    args = parser.parse_args()
    return args


def train(opt):
    # worldStages = {(1,1):1}
    worldStages = {(1,1):1, (1,2):1, (1,3):1, (1,4):1,
                   (2,1):1, (2,2):1, (2,3):1, (2,4):1,
                   (3,1):1, (3,2):1, (3,3):1, (3,4):1,
                   (4,1):1, (4,2):1, (4,3):1, (4,4):1,
                   (5,1):1, (5,2):1, (5,3):1, (5,4):1,
                   (6,1):1, (6,2):1, (6,3):1, (6,4):1,
                   (7,1):1, (7,2):1, (7,3):1, (7,4):1,
                   (8,1):1, (8,2):1, (8,3):1, (8,4):1}
    WSSelector = WorldStageSelector(worldStages)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    if not os.path.isdir(opt.saved_path):
        os.makedirs(opt.saved_path)
    mp = _mp.get_context("spawn")

    ws = WSSelector.select()
    envs = MultipleEnvironments(ws, opt.action_type, opt.num_processes)
    model = SimpleCNN2(envs.num_states, envs.num_actions)
    model.load_state_dict(torch.load("trained_models/All/default3/checkpoint_4500"))
    if torch.cuda.is_available():
        model.cuda()
    model.share_memory()
    # process = mp.Process(target=eval, args=(opt, model, envs.num_states, envs.num_actions))
    # process.start()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    [agent_conn.send(("reset", None, ws)) for agent_conn in envs.agent_conns]
    curr_states = [agent_conn.recv() for agent_conn in envs.agent_conns]
    curr_states = torch.from_numpy(np.concatenate(curr_states, 0))
    if torch.cuda.is_available():
        curr_states = curr_states.cuda()
    curr_episode = 0

    save_path = "{}/{}/{}".format(opt.saved_path, 'All', 'default4')
    logger = Logger(log_dir=(save_path+"/tensorboard"))

    while True:
        if curr_episode % opt.save_interval == 0 and curr_episode > 0:
            torch.save(model.state_dict(),
                       save_path+"/latest")
            torch.save(model.state_dict(),
                       save_path+"/checkpoint_{}".format(curr_episode))
        curr_episode += 1
        old_log_policies = []
        actions = []
        values = []
        states = []
        rewards = []
        dones = []
        for _ in range(opt.num_local_steps):
            states.append(curr_states)
            logits, value = model(curr_states)
            values.append(value.squeeze())
            policy = F.softmax(logits, dim=1)
            old_m = Categorical(policy)
            action = old_m.sample()
            actions.append(action)
            old_log_policy = old_m.log_prob(action)
            old_log_policies.append(old_log_policy)
            if torch.cuda.is_available():
                [agent_conn.send(("step", act, None)) for agent_conn, act in zip(envs.agent_conns, action.cpu())]
            else:
                [agent_conn.send(("step", act, None)) for agent_conn, act in zip(envs.agent_conns, action)]

            state, reward, done, info = zip(*[agent_conn.recv() for agent_conn in envs.agent_conns])
            
            state = list(state)
            for index, agent_conn in enumerate(envs.agent_conns):
                if done[index] == 1:
                    if np.random.rand(1).item() < 0.2:
                        ws = WSSelector.select()
                        agent_conn.send(("reset", None, ws))
                        state[index] = agent_conn.recv()
            state = torch.from_numpy(np.concatenate(state, 0))


            if torch.cuda.is_available():
                state = state.cuda()
                reward = torch.cuda.FloatTensor(reward)
                done = torch.cuda.FloatTensor(done)
            else:
                reward = torch.FloatTensor(reward)
                done = torch.FloatTensor(done)
            rewards.append(reward)
            dones.append(done)
            curr_states = state
            
            logger.log(reward, done, info)
        logger.boardWrite(curr_episode)

        _, next_value, = model(curr_states)
        next_value = next_value.squeeze()
        old_log_policies = torch.cat(old_log_policies).detach()
        actions = torch.cat(actions)
        values = torch.cat(values).detach()
        states = torch.cat(states)
        gae = 0
        R = []
        for value, reward, done in list(zip(values, rewards, dones))[::-1]:
            gae = gae * opt.gamma * opt.tau
            gae = gae + reward + opt.gamma * next_value.detach() * (1 - done) - value.detach()
            next_value = value
            R.append(gae + value)
        R = R[::-1]
        R = torch.cat(R).detach()
        advantages = R - values
        for i in range(opt.num_epochs):
            indice = torch.randperm(opt.num_local_steps * opt.num_processes)
            for j in range(opt.batch_size):
                batch_indices = indice[
                                int(j * (opt.num_local_steps * opt.num_processes / opt.batch_size)): int((j + 1) * (
                                        opt.num_local_steps * opt.num_processes / opt.batch_size))]
                logits, value = model(states[batch_indices])
                new_policy = F.softmax(logits, dim=1)
                new_m = Categorical(new_policy)
                new_log_policy = new_m.log_prob(actions[batch_indices])
                ratio = torch.exp(new_log_policy - old_log_policies[batch_indices])
                actor_loss = -torch.mean(torch.min(ratio * advantages[batch_indices],
                                                   torch.clamp(ratio, 1.0 - opt.epsilon, 1.0 + opt.epsilon) *
                                                   advantages[
                                                       batch_indices]))
                # critic_loss = torch.mean((R[batch_indices] - value) ** 2) / 2
                critic_loss = F.smooth_l1_loss(R[batch_indices], value.squeeze())
                entropy_loss = torch.mean(new_m.entropy())
                total_loss = actor_loss + critic_loss - opt.beta * entropy_loss
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
            logger.logLoss(critic_loss, actor_loss, entropy_loss, total_loss, curr_episode)
        print("Episode: {}. Total loss: {}".format(curr_episode, total_loss))


if __name__ == "__main__":
    opt = get_args()
    train(opt)

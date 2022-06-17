import os
os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import torch
from src.env import create_train_env
from src.model import SimpleCNN2
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
import torch.nn.functional as F
import time


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of model described in the paper: Proximal Policy Optimization Algorithms for Contra Nes""")
    parser.add_argument("--world", type=int, default=3)
    parser.add_argument("--stage", type=int, default=3)
    parser.add_argument("--action_type", type=str, default="simple")
    parser.add_argument("--saved_path", type=str, default="pretrained")
    parser.add_argument("--output_path", type=str, default="output")
    args = parser.parse_args()
    return args


def test(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    if opt.action_type == "right":
        actions = RIGHT_ONLY
    elif opt.action_type == "simple":
        actions = SIMPLE_MOVEMENT
    else:
        actions = COMPLEX_MOVEMENT
    # env = create_train_env((opt.world, opt.stage), actions,"{}/video_{}_{}.mp4".format(opt.output_path, opt.world, opt.stage))
    env = create_train_env((opt.world, opt.stage), actions)
    model = SimpleCNN2(env.observation_space.shape[0], len(actions))


    # model.load_state_dict(torch.load("trained_models/All/default4/checkpoint_3250"))
    model.load_state_dict(torch.load("trained_models/BC/default1/model"))
    model.cuda()
    
    model.eval()
    state = torch.from_numpy(env.reset())
    while True:
        if torch.cuda.is_available():
            state = state.cuda()
        logits, value = model(state)
        policy = F.softmax(logits, dim=1)
        action = torch.argmax(policy).item()
        state, reward, done, info = env.step(action)
        state = torch.from_numpy(state)
        env.render()
        time.sleep(0.01)
        
        if done:
            env.close()
            return info["flag_get"], info['x_pos'], 400-info['time'], info['score']


if __name__ == "__main__":
    num_stage = 0
    num_cleared = 0
    sum_progress = 0
    sum_time = 0
    sum_score = 0
    opt = get_args()
    # worldStages = {(4,3):3193}
    worldStages = {(1,1):3161, (1,2):3161, (1,3):2425, (1,4):2246,
                   (2,1):3193, (2,2):3161, (2,3):3593, (2,4):2244,
                   (3,1):3193, (3,2):3337, (3,3):2409, (3,4):2258,
                   (4,1):3593, (4,2):3161, (4,3):2345, (4,4):2756,
                   (5,1):3177, (5,2):3193, (5,3):2425, (5,4):2247,
                   (6,1):2969, (6,2):3449, (6,3):2665, (6,4):2250,
                   (7,1):2857, (7,2):3161, (7,3):3593, (7,4):3271,
                   (8,1):6009, (8,2):3449, (8,3):3417, (8,4):10000}
    for ws in worldStages.keys():
        opt.world = ws[0]
        opt.stage = ws[1]
        flag, x_pos, time_to_clear, score = test(opt)
        num_stage += 1
        num_cleared += flag
        progress = x_pos/worldStages[ws]*100
        if flag:
            progress = 100
            sum_time += time_to_clear
        sum_progress += progress
        sum_score += score
        print("World {} Stage {} | Flag : {:1} | Progress : {:6.2f}% | Time : {:3} | Score : {:6} ".format(ws[0], ws[1], flag, progress, time_to_clear, score))
    print("Cleared stage : {}".format(num_cleared))
    print("Average progress : {:3.2f}%".format(sum_progress/num_stage))
    print("Average time to complete : {:.2f}".format(sum_time/num_stage))
    print("Score sum: {}".format(sum_score))
    


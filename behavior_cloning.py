from itertools import count
import os
import argparse
import torch
from src.model import SimpleCNN, SimpleCNN2
import torch.nn.functional as F
import numpy as np
from src.utils import Logger, WorldStageSelector
from torch.utils.data import TensorDataset, DataLoader

def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of model described in the paper: Proximal Policy Optimization Algorithms for Super Mario Bros""")
    parser.add_argument("--action_type", type=str, default="simple")
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--num_epochs', type=int, default=400)
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--data_path", type=str, default="data_train")
    
    args = parser.parse_args()
    return args

def train_from_expert(model, states_expert, actions_expert, optimizer):
    logits, values = model(states_expert)
    policy = F.softmax(logits, dim=1)
    actions = torch.argmax(policy, dim=-1)
    policy_expert = F.one_hot(actions_expert, num_classes=7).float()
    optimizer.zero_grad()
    loss = F.binary_cross_entropy(policy, policy_expert)
    loss.backward()
    optimizer.step()
    return loss.item(), sum(actions_expert!=actions).item()


if __name__ == "__main__":
    opt = get_args()
    model = SimpleCNN2(4, 7)
    save_path = "{}/{}/{}".format(opt.saved_path, 'BC', 'default1')
    model.load_state_dict(torch.load("trained_models/All/default4/checkpoint_3250"))
    model.cuda()
    worldStages_BC = {  (1,1):1, (1,2):1, (1,3):1, (1,4):1,
                        (2,1):1, (2,2):1, (2,3):1, (2,4):1,
                        (3,1):1, (3,2):1, (3,3):1, (3,4):1,
                        (4,1):1, (4,2):1, (4,3):1, (4,4):1,
                        (5,1):1, (5,2):1, (5,3):1, (5,4):1,
                        (6,1):1, (6,2):1, (6,3):0, (6,4):1,
                        (7,1):1, (7,2):0, (7,3):1, (7,4):1,
                        (8,1):0, (8,2):1, (8,3):0, (8,4):0}
    states_expert = []
    actions_expert = []
    for ws in worldStages_BC.keys():
        if worldStages_BC[ws] == 1:
            data = np.load(os.path.join(opt.data_path, "states_actions_{}-{}.npz".format(ws[0], ws[1])))
            states_expert.append(data['state_list'])
            actions_expert.append(data['action_list'])
    states_expert = torch.from_numpy(np.concatenate(states_expert, axis=0))
    actions_expert = torch.from_numpy(np.concatenate(actions_expert, axis=0))

    dataset = TensorDataset(states_expert, actions_expert)
    loader = DataLoader(dataset, batch_size=2048, shuffle=True)

    

    optimizer_BC = torch.optim.Adam(model.parameters(), lr=opt.lr)
    
    for i in range(opt.num_epochs):
        loss_sum = 0
        wrong_sum = 0
        counter = 0
        for s, a in loader :
            loss, wrong = train_from_expert(model, s.cuda(), a.cuda(), optimizer_BC)
            loss_sum += loss
            wrong_sum += wrong
            counter += 1
        
        print("episode {} | loss {:.4f} | wrong : {}".format(i, loss/counter, wrong_sum))
        if wrong_sum == 0:
            torch.save(model.state_dict(),save_path+"/model")
            print("saved")
    

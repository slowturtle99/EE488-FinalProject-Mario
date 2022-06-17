from collections import deque, defaultdict
from torch.utils.tensorboard import SummaryWriter
import time
import datetime
import numpy as np

class FIFO():
    def __init__(self, size_limit):
        self.size = 0
        self.size_limit = size_limit
        self.dq = deque()
        self.sum = 0 
    
    def put(self, x):
        self.dq.append(x)
        self.sum += x
        if self.size >= self.size_limit:
            self.sum -= self.dq.popleft()
        else:
            self.size += 1
    
    def mean(self):
        if self.size == 0:
            return 0
        return self.sum/self.size



class Logger():
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir=log_dir, flush_secs=10)
        self.size_limit = 20
        self.FIFOs = defaultdict(lambda: FIFO(self.size_limit))
        self.log_keys = ['x_pos', 'flag_get']

        self.start_time = time.time()
        self.last_time = time.time()
        
    
    def log(self, reward, done, info):
        for reward_, done_, info_ in zip(reward, done, info):
            if done_ == 1:
                for k in self.log_keys:
                    kws = "{}/{}-{}".format(k, info_['world'], info_['stage'])
                    self.FIFOs[kws].put(info_[k])
                    self.FIFOs["{}/average".format(k)].put(info_[k])
    
    def logLoss(self, critic_loss, actor_loss, entropy_loss, total_loss, episode):
        self.writer.add_scalar("Loss/critic", critic_loss, global_step=episode)
        self.writer.add_scalar("Loss/actor", actor_loss, global_step=episode)
        self.writer.add_scalar("Loss/entrophy", entropy_loss, global_step=episode)
        self.writer.add_scalar("Loss/total", total_loss, global_step=episode)
    
    def boardWrite(self, episode):
        for kws in self.FIFOs.keys():
            self.writer.add_scalar(kws, self.FIFOs[kws].mean(), global_step=episode)

        cur_time = time.time()
        print("Total time elapsed : {} | From last update : {:.1f}".format(str(datetime.timedelta(seconds=int(cur_time-self.start_time))), cur_time-self.last_time))
        self.last_time = cur_time


class WorldStageSelector():
    def __init__(self, worldStages):
        self.WSs = []
        self.WS_probs = []
        for ws in worldStages.keys():
            self.WSs.append(ws)
            self.WS_probs.append(worldStages[ws])
        self.WS_probs = np.array(self.WS_probs)
        self.WS_probs = self.WS_probs / np.sum(self.WS_probs)

        self.numWS = len(self.WSs)
    
    def select(self):
        index = np.random.choice(self.numWS, 1, p=self.WS_probs).item()
        return self.WSs[index]

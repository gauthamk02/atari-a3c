from __future__ import print_function
import torch, os, gym, time, glob, argparse, sys
import numpy as np
from scipy.signal import lfilter
from skimage.transform import resize as imresize
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
os.environ['OMP_NUM_THREADS'] = '1'
import warnings
warnings.filterwarnings("ignore")

def get_args():
    config = {
        'env': 'Breakout-v4',
        'processes': 20,
        'steps': int(1e7),
        'rnn_steps': 20,
        'lr': 1e-4,
        'seed': 1,
        'gamma': 0.99,
        'tau': 1.0,
        'horizon': 0.99,
        'hidden': 256
    }
    return config

discount = lambda x, gamma: lfilter([1],[1,-gamma],x[::-1])[::-1] # discounted rewards one liner
prepro = lambda img: imresize(img[35:195].mean(2), (80,80)).astype(np.float32).reshape(1,80,80)/255.

def printlog(args, s, end='\n', mode='a'):
    print(s, end=end) ; f=open(args['save_dir']+'log.txt',mode) ; f.write(s+'\n') ; f.close()

class NNPolicy(nn.Module): # an actor-critic neural network
    def __init__(self, channels, memsize, num_actions):
        super(NNPolicy, self).__init__()
        self.conv1 = nn.Conv2d(channels, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.gru = nn.GRUCell(32 * 5 * 5, memsize)
        self.critic_linear, self.actor_linear = nn.Linear(memsize, 1), nn.Linear(memsize, num_actions)

    def forward(self, inputs, train=True, hard=False):
        inputs, hx = inputs
        x = F.elu(self.conv1(inputs))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        hx = self.gru(x.view(-1, 32 * 5 * 5), (hx))
        return self.critic_linear(hx), self.actor_linear(hx), hx

class SharedAdam(torch.optim.Adam): 
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super(SharedAdam, self).__init__(params, lr, betas, eps, weight_decay)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['shared_steps'], state['step'] = torch.zeros(1).share_memory_(), 0
                state['exp_avg'] = p.data.new().resize_as_(p.data).zero_().share_memory_()
                state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_().share_memory_()
                
        def step(self, closure=None):
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None: continue
                    self.state[p]['shared_steps'] += 1
                    self.state[p]['step'] = self.state[p]['shared_steps'][0] - 1
            super.step(closure)

def cost_func(args, values, logps, actions, rewards):
    np_values = values.view(-1).data.numpy()


    delta_t = np.asarray(rewards) + args['gamma'] * np_values[1:] - np_values[:-1]
    logpys = logps.gather(1, torch.tensor(actions).view(-1,1))
    gen_adv_est = discount(delta_t, args['gamma'] * args['tau'])
    policy_loss = -(logpys.view(-1) * torch.FloatTensor(gen_adv_est.copy())).sum()
    

    rewards[-1] += args['gamma'] * np_values[-1]
    discounted_r = discount(np.asarray(rewards), args['gamma'])
    discounted_r = torch.tensor(discounted_r.copy(), dtype=torch.float32)
    value_loss = .5 * (discounted_r - values[:-1,0]).pow(2).sum()

    entropy_loss = (-logps * torch.exp(logps)).sum() 
    return policy_loss + 0.5 * value_loss - 0.01 * entropy_loss

def train(shared_model, shared_optimizer, rank, args, info):
    env = gym.make(args['env']) 
    env.seed(args['seed'] + rank) ; torch.manual_seed(args['seed'] + rank) 
    model = NNPolicy(channels=1, memsize=args['hidden'], num_actions=args['num_actions']) 
    state = torch.tensor(prepro(env.reset()))

    start_time = last_disp_time = time.time()
    episode_length, epr, eploss, done  = 0, 0, 0, True 

    for i in range(args['steps']): 
        model.load_state_dict(shared_model.state_dict()) 

        hx = torch.zeros(1, 256) if done else hx.detach() 
        values, logps, actions, rewards = [], [], [], [] 

        for step in range(args['rnn_steps']):
            episode_length += 1
            value, logit, hx = model((state.view(1,1,80,80), hx))
            logp = F.log_softmax(logit, dim=-1)

            action = torch.exp(logp).multinomial(num_samples=1).data[0]
            state, reward, done, _ = env.step(action.numpy()[0])

            state = torch.tensor(prepro(state)) ; epr += reward
            reward = np.clip(reward, -1, 1) 
            done = done or episode_length >= 1e4 
            
            info['frames'].add_(1)
            num_frames = int(info['frames'].item())
            
            if num_frames % 1e4 == 0: 
                printlog(args, '\n\t{:.0f}M frames: saved model\n'.format(num_frames/1e6))
                torch.save(shared_model.state_dict(), args['save_dir']+'model.{:.0f}.tar'.format(num_frames/1e4))

            if done: 
                info['episodes'] += 1
                interp = 1 if info['episodes'][0] == 1 else 1 - args['horizon']
                info['run_epr'].mul_(1-interp).add_(interp * epr)
                info['run_loss'].mul_(1-interp).add_(interp * eploss)

            if done: 
                episode_length, epr, eploss = 0, 0, 0
                state = torch.tensor(prepro(env.reset()))

            values.append(value) ; logps.append(logp) ; actions.append(action) ; rewards.append(reward)

        if i % 100 == 0: 
            printlog(args, 'episodes {:.0f}, frames {:.1f}M, mean epr {:.2f}, run loss {:.2f}'
                .format(info['episodes'].item(), num_frames/1e6,
                info['run_epr'].item(), info['run_loss'].item()))


        next_value = torch.zeros(1,1) if done else model((state.unsqueeze(0), hx))[0]
        values.append(next_value.detach())

        loss = cost_func(args, torch.cat(values), torch.cat(logps), torch.cat(actions), np.asarray(rewards))
        eploss += loss.item()
        shared_optimizer.zero_grad() ; loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 40)

        for param, shared_param in zip(model.parameters(), shared_model.parameters()):
            if shared_param.grad is None: shared_param._grad = param.grad 

if __name__ == "__main__":
    if sys.version_info[0] > 2:
        mp.set_start_method('spawn') 
    
    args = get_args()
    args['save_dir'] = 'breakout-v4/' 

    args['num_actions'] = gym.make(args['env']).action_space.n 
    os.makedirs(args['save_dir']) if not os.path.exists(args['save_dir']) else None 

    torch.manual_seed(args['seed'])
    shared_model = NNPolicy(channels=1, memsize=args['hidden'], num_actions=args['num_actions']).share_memory()
    shared_optimizer = SharedAdam(shared_model.parameters(), lr=args['lr'])

    info = {k: torch.DoubleTensor([0]).share_memory_() for k in ['run_epr', 'run_loss', 'episodes', 'frames']}

    if int(info['frames'].item()) == 0: printlog(args,'', end='', mode='w')
    
    processes = []
    for rank in range(args['processes']):
        p = mp.Process(target=train, args=(shared_model, shared_optimizer, rank, args, info))
        p.start() ; processes.append(p)
    for p in processes: p.join()

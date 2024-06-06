# Baby Advantage Actor-Critic | Sam Greydanus | October 2017 | MIT License

from __future__ import print_function
import torch, os, gym, time
import numpy as np
from skimage.transform import resize as imresize
import torch.nn as nn
import torch.nn.functional as F
os.environ['OMP_NUM_THREADS'] = '1'
import warnings
warnings.filterwarnings("ignore")


prepro = lambda img: imresize(img[35:195].mean(2), (80,80)).astype(np.float32).reshape(1,80,80)/255.
path = "breakout-v4/model.800.tar"

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

    def try_load(self, save_dir):
        return 0

if __name__ == "__main__":
    env = gym.make('Breakout-v4')
    env.seed(1)
    model = NNPolicy(channels=1, memsize=256, num_actions=env.action_space.n)
    model.load_state_dict(torch.load(path))

    state = torch.tensor(prepro(env.reset()))
    hx = torch.zeros(1, 256)

    while True:
        value, logit, hx = model((state.view(1, 1, 80, 80), hx))
        logp = F.log_softmax(logit, dim=-1)
        action = torch.exp(logp).multinomial(num_samples=1).data[0]
        state, reward, done, _ = env.step(action.numpy()[0])
        state = torch.tensor(prepro(state))
        env.render()
        time.sleep(0.1)
        if done:
            state = torch.tensor(prepro(env.reset()))
            hx = torch.zeros(1, 256)

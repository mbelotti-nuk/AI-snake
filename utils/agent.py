import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from collections import namedtuple
import torch
from typing import List
import random   
import os

experience = namedtuple('experience', ("state", "next_state", "action", "reward", "done"))

class linear_dqn_net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(11, 256),
            nn.ReLU(),
            nn.Linear(256, 4),
            nn.Sigmoid()
            #nn.Softmax()
        )

    def forward(self, x:torch.Tensor)->torch.Tensor:
        x = self.linear(x)
        return x
    
    # Save a model
    def save_model(self):
        with torch.no_grad():
            torch.save(self.state_dict(), 'models/' + "LDQN.pt")

    # Loads a model
    def load_model(self, path):
        self.load_state_dict(torch.load(path)) 

class replay_buffer(object):
    """
    Memory which allows for storing and sampling batches of transitions

    """
    def __init__(self):
        self.replay_memory_size = 500_000
        self.buffer = np.empty(self.replay_memory_size, dtype = [("experience", experience)] )
        self.pointer = 0
        self.dtype = np.uint8

    # Adds a single experience to the memory buffer
    def add_experience(self, current_experience):         
        if(self.pointer < self.replay_memory_size):
            self.buffer[self.pointer]["experience"] = current_experience
        else:
            self.buffer[self.pointer % self.replay_memory_size]["experience"] = current_experience
        self.pointer += 1
    
    @property
    def buffer_length(self):
        if(self.pointer < self.replay_memory_size):
            return self.pointer
        else:
            return self.replay_memory_size 
    
    @property
    def buffer_end(self):
        if(self.pointer < self.replay_memory_size):
            return self.pointer -1
        else:
            return self.replay_memory_size -1
    
    def sample_batch(self, batch_size=64, device="cuda:0")->experience:
        """Samples a batch of transitions

        Args:
            batch_size (int, optional): Defaults to 64.
            device (str, optional): device for training Defaults to "cuda:0".

        Returns:
            experience:
        """
        batch_index = np.random.randint(0, self.buffer_end, size = batch_size)
        states, next_states, actions, rewards, dones = self._to_arrays(batch_index)

        # Convert to tensors with correct dimensions
        state =         (torch.tensor( states  ).float()).to(device)
        action =        torch.tensor( actions ).unsqueeze(1).type(torch.int64).to(device)
        reward =        torch.tensor( rewards ).float().unsqueeze(1).to(device)
        next_state =    (torch.tensor( next_states  ).float()).to(device)
        done =          torch.tensor( dones   ).float().unsqueeze(1).to(device)


        return experience(state, next_state, action, reward, done)
    
    def _to_arrays(self, batch_index):
        transitions = self.buffer["experience"][batch_index]
        states = np.stack([np.array(exp.state) for exp in transitions], axis=0)
        next_states = np.stack([np.array(exp.next_state) for exp in transitions], axis=0)
        actions = np.stack( [exp.action for exp in transitions] ) 
        rewards = np.stack( [exp.reward for exp in transitions] ) 
        dones = np.stack( [exp.done for exp in transitions] ) 
        return states, next_states, actions, rewards, dones


class Agent:
    def __init__(self, device, gamma=0.99, batch_size=64, lr = 1E-3) -> None:
        self.action_space = 4
        self.device = device
        self.replay_memory = replay_buffer()
        # batch size for training the network
        self.batch_size = batch_size
         # After how many training iterations the target network should update
        self.sync_network_rate = 25

        self.GAMMA = gamma
        # learning rate
        self.LR = lr
        
        self.num_games = 0
        
        self.policy_net = linear_dqn_net().to(self.device)
        self.target_net = linear_dqn_net().to(self.device)
        self.target_net.eval()

        
        # Set target net to be the same as policy net
        self.sync_networks()

        # Set optimizer & loss function
        self.optim = torch.optim.Adam(self.policy_net.parameters(), lr=self.LR)
        self.loss = torch.nn.HuberLoss() 

        self.dtypes = [torch.float32, torch.float32, torch.int64, torch.int64, torch.int64]

    def define_epsilon_startegy(self, eps=1, eps_min=0.1, num_games_end=1000, num_games_min=50):
        self.eps = eps
        self.eps_start = eps
        self.eps_min = eps_min
        self.num_games_end = num_games_end
        self.num_games_min = num_games_min

    def load_net(self, path):
        self.policy_net.load_model(path=path)

    def sync_networks(self):
        """Copies policy params to target net
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())


    def update_num_episodes(self):
        if self.num_games > self.num_games_min:
            self.eps = max(self.eps_min, self.eps_start - (self.eps_start-self.eps_min)*(self.num_games-self.num_games_min)/(self.num_games_end-self.num_games_min))
        self.num_games += 1

    def greedy_action(self, state:np.array)->int:
        """Returns the greedy action according to the policy net

        Args:
            state (np.array): 

        Returns:
            int: action to be taken
        """
        with torch.no_grad():
            action = self.policy_net(state).argmax().item()
        return action

    def choose_action(self, state:np.array, train=True)->int:
        """Returns an action based on epsilon greedy method

        Args:
            state (np.array): 
            train (bool, optional): Defaults to True.

        Returns:
            int: action to be taken
        """

        state = torch.from_numpy(state).float().to(self.device)
        if train:
            # choose action from model 
            if random.random() > self.eps:
                action = self.greedy_action(state)
            # return random action 
            else:
                action = random.choice([x for x in range(self.action_space)])
        else:
            action = self.greedy_action(state)

        return action
    
    def store_experience(self, *args):
        """Stores a transition into memory
        """
        self.replay_memory.add_experience(experience(*args))

    def learn(self, batch=None):
        """Samples a single batch according to batchsize and updates the policy net
        """
        if batch is None:
            batch = self.batch_size

        if self.replay_memory.buffer_length < batch:
            return 

        # Sample batch
        experiences = self.replay_memory.sample_batch(batch_size=batch, device=self.device)
        
        q_eval, q_target = self.agent_predictions(experiences)

        # Compute the loss
        loss = self.loss(q_eval, q_target).to(self.device)

        # Perform backward propagation and optimization step
        loss.backward()
        # clip gradient norm
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optim.step()
        self.optim.zero_grad()

        # sync target and policy networks
        # the update of the target is made with a frequency w.r.t. the number of steps done
        # and not w.r.t. the number of parameters (==self.learn_counter)
        if self.num_games % self.sync_network_rate == 0:
            self.sync_networks()
    

    def agent_predictions(self, experience:experience):
        q_eval = self.policy_net(experience.state).gather(1, experience.action)
        q_next = self.target_net(experience.next_state).detach().max(1)[0].unsqueeze(1)
        q_target = (1-experience.done) * (experience.reward + self.GAMMA * q_next) + (experience.done * experience.reward)
        return q_eval, q_target

        
    def plot_results(self, scores, save_path=None):
        plt.plot(np.arange(1, len(scores)+1), scores, label = "Scores per game", color="blue")
        plt.plot( np.convolve(scores, np.ones(100)/100, mode='valid'), label = "Moving mean scores", color="red")
        plt.title("Scores")
        plt.xlabel("Game")
        plt.legend()
        if save_path !=None:
            plt.savefig(os.path.join(save_path,"SCORES.png"))
        else:
            plt.savefig("results/Scores.png")
        plt.close()    

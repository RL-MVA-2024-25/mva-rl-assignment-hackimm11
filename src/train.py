import random
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from evaluate import evaluate_HIV

# Initialize the environment
env = TimeLimit(env=HIVPatient(domain_randomization=True), max_episode_steps=200)

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity
        self.data = []
        self.index = 0
        self.device = device

    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states), device=self.device, dtype=torch.float32),
            torch.tensor(actions, device=self.device, dtype=torch.long),
            torch.tensor(rewards, device=self.device, dtype=torch.float32),
            torch.tensor(np.array(next_states), device=self.device, dtype=torch.float32),
            torch.tensor(dones, device=self.device, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.data)

class ProjectAgent:
    def __init__(self, save_path="src/best_dqn_model.pt"):
        self.save_path = save_path
        self.gamma = 0.95
        self.learning_rate = 0.001
        self.epsilon_min = 0.02
        self.epsilon_max = 1.0
        self.epsilon_decay_period = 20000
        self.epsilon_delay_decay = 100
        self.batch_size = 512
        self.buffer_size = 100000
        self.gradient_steps = 1
        self.update_target_freq = 1000
        self.max_episodes = 800

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory = ReplayBuffer(self.buffer_size, self.device)

        self.model = self.create_model().to(self.device)
        self.target_model = deepcopy(self.model).to(self.device)
        self.target_model.eval()

        self.criterion = nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def create_model(self):
        state_dim = env.observation_space.shape[0]
        n_action = env.action_space.n
        hidden_units = 256

        return nn.Sequential(
            nn.Linear(state_dim, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, n_action),
        )

    def act(self, observation, use_random=False):
        """
        Decides action based on the current policy or random selection.
        `use_random` determines if random action is taken.
        """
        if use_random:
            return env.action_space.sample()
        with torch.no_grad():
            state_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(self.device)
            return self.model(state_tensor).argmax().item()

    def gradient_step(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        Q_next = self.target_model(next_states).max(1)[0].detach()
        target_Q = rewards + self.gamma * Q_next * (1 - dones)
        current_Q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        loss = self.criterion(current_Q, target_Q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self):
        
        device = torch.device('cpu')
        # self.path = os.getcwd() + "best.pt"
        self.model = self.create_model()
        self.model.load_state_dict(torch.load(self.save_path, map_location=device))
        self.model.eval()
def train(self):
        epsilon = self.epsilon_max
        epsilon_step = (self.epsilon_max - self.epsilon_min) / self.epsilon_decay_period
        best_validation_score = float('-inf')
        episode_rewards = []
        state, _ = env.reset()
        step = 0
        episode = 0
        episode_reward = 0
        last_best_reward = 0
        while episode < self.max_episodes:
            if step > self.epsilon_delay_decay:
                epsilon = max(self.epsilon_min, epsilon - epsilon_step)

            if random.random() < epsilon:
                action = self.act(state, use_random=True)
            else:
                action = self.act(state, use_random=False)

            next_state, reward, done, truncated, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            episode_reward += reward
            state = next_state

            for _ in range(self.gradient_steps):
                self.gradient_step()

            if step % self.update_target_freq == 0:
                self.target_model.load_state_dict(self.model.state_dict())

            if done or truncated:
                episode += 1
                print(f"Episode {episode:3d} | Reward: {episode_reward:.2e} | Epsilon: {epsilon:.4f}")
                episode_rewards.append(episode_reward)
                state, _ = env.reset()

                # Validation
                # if episode % 100 == 0:
                validation_score = evaluate_HIV(agent=self, nb_episode=1)
                print(f"Validation Score: {validation_score:.2e}")
                if validation_score > best_validation_score:
                    best_validation_score = validation_score
                    self.best_model1 = deepcopy(self.model)
                    self.save( 'best_model_dqn_validation')
                    print("New best model saved - val")
                if episode_reward > last_best_reward:
                    # prev_validation_score = validation_score
                    last_best_reward = episode_reward
                    self.best_model2 = deepcopy(self.model)
                    self.save('best_model_dqn_reward')
                    print("best model saved - reward" )
                
                episode_reward = 0

            step += 1

        return episode_rewards

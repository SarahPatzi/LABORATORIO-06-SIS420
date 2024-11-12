import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.softmax(x, dim=-1)

class REINFORCEAgent:
    def __init__(self, env, gamma=0.99, learning_rate=0.001):
        self.env = env
        self.gamma = gamma
        self.policy = PolicyNetwork(env.observation_space.shape[0], env.action_space.n)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.rewards = []

    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.policy(state)
        action = torch.multinomial(probs, 1).item()
        return action

    def update_policy(self, rewards, log_probs):
        # Compute returns (discounted rewards)
        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns)

        # Normalize returns for stability
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        # Calculate policy gradient loss
        policy_loss = 0  # Accumulate the policy loss as a single scalar
        for log_prob, G in zip(log_probs, returns):
            policy_loss += -log_prob * G

        # Backpropagation
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

    def train(self, episodes=1000):
        for episode in range(episodes):
            state = self.env.reset()[0]
            log_probs = []
            rewards = []
            done = False
            while not done:
                action = self.choose_action(state)
                log_prob = torch.log(self.policy(torch.FloatTensor(state).unsqueeze(0))[0, action])
                next_state, reward, done, _, _ = self.env.step(action)

                log_probs.append(log_prob)
                rewards.append(reward)
                state = next_state

            self.update_policy(rewards, log_probs)
            total_reward = sum(rewards)
            self.rewards.append(total_reward)
            print(f"Episode {episode + 1}: Total Reward = {total_reward}")

    def plot_rewards(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.rewards, label='Recompensa por Episodio')
        # Adding a moving average for smoother visualization
        window_size = 50
        if len(self.rewards) >= window_size:
            moving_avg = np.convolve(self.rewards, np.ones(window_size) / window_size, mode='valid')
            plt.plot(range(window_size - 1, len(self.rewards)), moving_avg, label='Promedio Móvil (50)', color='red')
        plt.xlabel("Episodio")
        plt.ylabel("Recompensa Total")
        plt.title("Recompensas por Episodio")
        plt.legend()
        plt.show()

    def save_policy(self, filename="policy_network.pth"):
        torch.save(self.policy.state_dict(), filename)
        print(f"Policy network saved to {filename}.")

    def load_policy(self, filename="policy_network.pth"):
        self.policy.load_state_dict(torch.load(filename))
        print(f"Policy network loaded from {filename}.")

if __name__ == "__main__":
    env = gym.make("MountainCar-v0")
    agent = REINFORCEAgent(env)
    agent.train(episodes=1000)
    env.close()
    agent.plot_rewards()
    agent.save_policy()  # Save the policy after training

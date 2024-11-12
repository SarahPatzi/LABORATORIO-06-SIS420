import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle  # Para guardar y cargar la tabla Q

class EpsilonGreedyOptimisticQLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, optimistic_initial_value=5.0):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        # Inicializar la tabla Q con valores optimistas
        self.q_table = np.full((40, 40, self.env.action_space.n), optimistic_initial_value)
        self.discrete_os_window = (env.observation_space.high - env.observation_space.low) / [40, 40]
        self.rewards = []

    def get_discrete_state(self, state):
        return tuple(((state - self.env.observation_space.low) / self.discrete_os_window).astype(int))

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, next_state):
        max_future_q = np.max(self.q_table[next_state])
        current_q = self.q_table[state + (action,)]
        new_q = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)
        self.q_table[state + (action,)] = new_q

    def train(self, episodes=1000, render_every=100):
        for episode in range(episodes):
            state = self.get_discrete_state(self.env.reset()[0])
            done = False
            episode_reward = 0
            while not done:
                if episode % render_every == 0:
                    self.env.render()
                action = self.choose_action(state)
                next_state_raw, reward, done, _, _ = self.env.step(action)
                next_state = self.get_discrete_state(next_state_raw)
                self.update_q_table(state, action, reward, next_state)
                state = next_state
                episode_reward += reward
            self.rewards.append(episode_reward)
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
            print(f"Episode {episode + 1}: Reward = {episode_reward}, Epsilon = {self.epsilon:.4f}")

        print("Training complete.\nFinal Q-Table:")
        print(self.q_table)

    def plot_rewards(self):
        plt.plot(self.rewards)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Rewards per Episode")
        plt.show()

    def save_q_table(self, filename="q_table_optimistic.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(self.q_table, f)
        print(f"Q-table saved to {filename}.")

    def load_q_table(self, filename="q_table_optimistic.pkl"):
        with open(filename, "rb") as f:
            self.q_table = pickle.load(f)
        print(f"Q-table loaded from {filename}.")
    
    def plot_q_table(self, action_index=0):
        if action_index >= self.env.action_space.n:
            print("Índice de acción fuera de rango.")
            return
        plt.imshow(self.q_table[:, :, action_index], cmap="viridis")
        plt.colorbar()
        plt.title(f"Q-table values for action {action_index}")
        plt.xlabel("State Dimension 1")
        plt.ylabel("State Dimension 2")
        plt.show()

if __name__ == "__main__":
    #env = gym.make("MountainCar-v0", render_mode="human")  # Especifica el modo de renderizado
    env = gym.make("MountainCar-v0")
    agent = EpsilonGreedyOptimisticQLearningAgent(env)
    agent.train(episodes=1000, render_every=100)
    env.close()
    agent.plot_rewards()
    agent.save_q_table()
    agent.plot_q_table(action_index=0)
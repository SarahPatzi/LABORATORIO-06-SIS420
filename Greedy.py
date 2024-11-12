import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle  # Para guardar y cargar la tabla Q

class GreedyQLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.99):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = np.zeros((40, 40, self.env.action_space.n))
        self.discrete_os_window = (env.observation_space.high - env.observation_space.low) / [40, 40]
        self.rewards = []

    def get_discrete_state(self, state):
        return tuple(((state - self.env.observation_space.low) / self.discrete_os_window).astype(int))

    def choose_action(self, state):
        # Selecciona la acción con el valor Q más alto (sin exploración)
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
            print(f"Episodio {episode + 1}: Recompensa = {episode_reward}")

        print("Entrenamiento completo.\nTabla Q Final:")
        print(self.q_table)

    def plot_rewards(self):
        plt.plot(self.rewards)
        plt.xlabel("Episodio")
        plt.ylabel("Recompensa Total")
        plt.title("Recompensas por Episodio")
        plt.show()

    def save_q_table(self, filename="q_table_greedy.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(self.q_table, f)
        print(f"Tabla Q guardada en {filename}.")

    def load_q_table(self, filename="q_table_greedy.pkl"):
        with open(filename, "rb") as f:
            self.q_table = pickle.load(f)
        print(f"Tabla Q cargada desde {filename}.")

if __name__ == "__main__":
    #env = gym.make("MountainCar-v0", render_mode="human")  # Especifica el modo de renderizado
    env = gym.make("MountainCar-v0")
    agent = GreedyQLearningAgent(env)
    agent.train(episodes=1000, render_every=200)
    env.close()
    agent.plot_rewards()
    agent.save_q_table()

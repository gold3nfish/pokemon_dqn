
import tensorflow as tf
import numpy as np
from DQN import DQN
from ReplayBuffer import ReplayBuffer
from PokemonEnv import PokemonEnv

# Hyperparameters
state_size = (84, 84, 1)  # Based on preprocessed game screen
action_size = 8  # Define based on the game controls
learning_rate = 0.001
discount_factor = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 32
buffer_size = 10000
num_episodes = 1000  # Total number of episodes to train

# Initialize environment, DQN, and replay buffer
env = PokemonEnv('Pokemon_Yellow.gb')
main_dqn = DQN(action_size, state_size)
target_dqn = DQN(action_size, state_size)
replay_buffer = ReplayBuffer(buffer_size)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# Function to update the target network
def update_target_network(main_dqn, target_dqn):
    target_dqn.set_weights(main_dqn.get_weights())

# Training loop
for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        # Epsilon-greedy policy for action selection
        if np.random.rand() <= epsilon:
            action = np.random.randint(action_size)
        else:
            q_values = main_dqn(np.array([state], dtype=np.float32))
            action = np.argmax(q_values[0])

        next_state, reward, done = env.step(action)
        replay_buffer.add(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        if replay_buffer.size() >= batch_size:
            # Sample a batch and train
            batch = replay_buffer.sample(batch_size)
            for s, a, r, ns, d in batch:
                with tf.GradientTape() as tape:
                    q_values = main_dqn(np.array([s], dtype=np.float32))
                    q_value = tf.reduce_sum(tf.one_hot(a, action_size) * q_values, axis=1)
                    next_q_values = target_dqn(np.array([ns], dtype=np.float32))
                    next_q_value = tf.reduce_max(next_q_values, axis=1)
                    expected_q_value = r + discount_factor * next_q_value * (1 - d)
                    loss = tf.keras.losses.Huber()(expected_q_value, q_value)

                gradients = tape.gradient(loss, main_dqn.trainable_variables)
                optimizer.apply_gradients(zip(gradients, main_dqn.trainable_variables))

        # Update target network
        update_target_network(main_dqn, target_dqn)

        # Decay epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

    print(f"Episode: {episode + 1}, Total reward: {total_reward}")

# Save the trained model
main_dqn.save('pokemon_deepq_model')

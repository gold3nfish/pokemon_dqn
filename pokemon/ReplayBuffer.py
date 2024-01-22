
import random
from collections import deque

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def add(self, state, action, reward, next_state, done):
        # Add experience to the buffer
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # Enhanced sampling method
        if len(self.buffer) < batch_size:
            return []

        # Implementing weighted sampling, if desired
        # Weights can be based on the magnitude of rewards or other criteria
        weights = [abs(reward) for (_, _, reward, _, _) in self.buffer]
        total_weight = sum(weights)
        probabilities = [weight / total_weight for weight in weights]
        
        return random.choices(self.buffer, weights=probabilities, k=batch_size)

    def size(self):
        # Return the current size of the buffer
        return len(self.buffer)

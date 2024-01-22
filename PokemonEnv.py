# PokemonEnv.py

from pyboy import PyBoy, WindowEvent
import numpy as np
from skimage.transform import resize

class PokemonEnv:
    def __init__(self, rom_path):
        try:
            self.rom_path = rom_path  # Save rom_path as an instance attribute
            self.pyboy = PyBoy(self.rom_path)
            self.pyboy.set_emulation_speed(0)  # Max speed
        except Exception as e:
            print(f"Failed to initialize PyBoy with error: {e}")
            exit(1)
        self.initialize_game_state()
        
    def initialize_game_state(self):
        self.player_progress = 0
        self.last_battle_result = None
        self.player_pokemon_health = 100
        self.player_pokemon_level = 1

    def reset(self):
        # Reinitialize PyBoy to reset the game
        try:
            self.pyboy = PyBoy(self.rom_path)
            self.pyboy.set_emulation_speed(0)
        except Exception as e:
            print(f"Error resetting game: {e}")
            exit(1)
        self.initialize_game_state()
        return self.get_state()

    def step(self, action):
        try:
            self.pyboy.send_input(action)
            self.pyboy.tick()
        except Exception as e:
            print(f"Error during game step: {e}")
            exit(1)
        self.update_game_state()
        return self.get_state(), self.get_reward(), self.is_done()

    def get_state(self):
        try:
            screen = np.array(self.pyboy.botsupport_manager().screen().screen_ndarray())
            return self.preprocess_screen(screen)
        except Exception as e:
            print(f"Error getting game state: {e}")
            exit(1)

    def preprocess_screen(self, screen):
        gray_screen = np.mean(screen, axis=2)
        resized_screen = resize(gray_screen, (84, 84))
        return resized_screen

    def get_reward(self):
        progress_reward = self.compute_progress_reward()
        battle_reward = self.compute_battle_reward()
        health_reward = self.compute_health_reward()
        level_reward = self.compute_level_reward()
        return progress_reward + battle_reward + health_reward + level_reward

    def compute_progress_reward(self):
        return self.player_progress  # Implement game-specific logic

    def compute_battle_reward(self):
        return 10 if self.last_battle_result == 'win' else -5  # Implement game-specific logic

    def compute_health_reward(self):
        return self.player_pokemon_health / 100  # Implement game-specific logic

    def compute_level_reward(self):
        return self.player_pokemon_level  # Implement game-specific logic

    def is_done(self):
        return self.player_pokemon_health <= 0  # Implement game-specific logic

    def update_game_state(self):
        # Update game state attributes
        self.player_progress += self.simulate_progress()
        self.last_battle_result = self.simulate_battle_result()
        self.player_pokemon_health = max(0, self.player_pokemon_health - self.simulate_health_change())
        self.player_pokemon_level += self.simulate_level_up()

    def simulate_progress(self):
        progress_increment = 0

        if np.random.rand() < 0.1:  # Example: 10% chance of a significant event
            progress_increment += 5  # A larger increment for significant progress
        else:
            progress_increment += 1  # Regular incremental progress

        return progress_increment    

    def simulate_battle_result(self):
        # Randomly determine battle outcome
        return 'win' if np.random.rand() > 0.5 else 'lose'

    def simulate_health_change(self):
        # Simulate health change due to battles or other in-game events
        # Health decrease in a battle or increase due to healing
        health_change = np.random.randint(5, 20)  # Random health change
        return health_change if self.last_battle_result == 'lose' else -health_change

    def simulate_level_up(self):
        # Simulate level increase based on experience points
        return 0.1 if self.last_battle_result == 'win' else 0

    def close(self):
        try:
            self.pyboy.stop()
        except Exception as e:
            print(f"Error closing PyBoy: {e}")
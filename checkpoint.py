# Authors: N Miller

import os
import pickle

class CheckpointManager:
    """Simple checkpoint manager for GA runs."""
    def __init__(self, save_path='GA/'):
        self.filename = save_path + 'ga_checkpoint.pkl'

    def save(self, generation, population, ga_state):
        data = {
            'generation': generation,
            'population': population,
            'ga_state': ga_state.__dict__,
        }
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)
        with open(self.filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"Checkpoint saved at generation {generation}")

    def load(self):
        if not os.path.exists(self.filename):
            return None
        with open(self.filename, 'rb') as f:
            data = pickle.load(f)
        print(f"Loaded checkpoint from generation {data['generation']}")
        return data

    def clear(self):
        if os.path.exists(self.filename):
            print(f"This file: {self.filename} still exists and i am not deleting it!!!!")
            #os.remove(self.filename)

def run_with_checkpoint(run_fn, output_path):
    manager = CheckpointManager(save_path = output_path)
    result = run_fn(manager)
    manager.clear()
    return result

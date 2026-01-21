import numpy as np
import random
import torch
class TrajectoryBuffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add(self, trajectory):
        """trajectory: dict(states, actions, log_pf, masks, errors, reward)"""
        payload = {
            "states": [s.detach().cpu() for s in trajectory["states"]],
            "actions": [int(a) for a in trajectory["actions"]],
            "log_pf": [float(lp) for lp in trajectory.get("log_pf", [])],
            "masks": [np.array(mask, dtype=np.float32) for mask in trajectory.get("masks", [])],
            "errors": [int(err) for err in trajectory.get("errors", [])],
            "reward": float(trajectory["reward"]),
            "ctx": trajectory.get("ctx").clone().cpu() if trajectory.get("ctx") is not None else None,
        }

        if len(self.buffer) < self.capacity:
            self.buffer.append(payload)
        else:
            self.buffer[self.position] = payload
        self.position = (self.position + 1) % self.capacity

    def extend(self, trajectories):
        for traj in trajectories:
            self.add(traj)

    def __len__(self):
        return len(self.buffer)

    def sample(self, batch_size):
        batch_size = min(batch_size, len(self.buffer))
        return random.sample(self.buffer, batch_size)

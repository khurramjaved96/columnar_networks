import logging

import numpy as np
import torch

logger = logging.getLogger("experiment")
import random

from itertools import combinations

class Pattern:

    def __init__(self, tasks):
        self.capacity = tasks
        self.tasks = tasks
        self.current_task = 0
        self.iterators = {}
        self.counter = 0
        self.comb = list(combinations(list(range(self.tasks)), 3))

        self.input_features = {}
        for c, comb in enumerate(self.comb):
            ones = torch.zeros(self.tasks)
            for a in comb:
                ones[a] = 1
            self.input_features[c] = [ones, torch.tensor(random.random()*10 - 5)]


    def change_task(self):
        index = random.randint(0, len(self.input_features)-1)
        self.input_features[index][1] = torch.tensor(random.random()*10 - 5)

    def sample(self):

        if random.random() < 0.1:
            self.change_task()
        if random.random() < 0.5:
            self.current_task = random.randint(0, self.tasks - 1 )
        self.counter = (self.counter + 1) % 20

        return self.input_features[self.current_task]


class SineBenchmark:

    def __init__(self, tasks):
        self.capacity = tasks
        self.tasks = tasks
        self.current_task = 0
        self.iterators = {}
        self.counter = 0

    def generate_task(self, task):

        amplitude = (np.random.rand() + 0.02) * (5)
        phase = np.random.rand() * np.pi
        # logger.info("New function")
        # logger.info("Amp, phase = %s %s", amplitude, phase)
        self.iterators[task] = {'id': task, 'phase': phase, 'amplitude': amplitude}

        return self.iterators[task]

    def change_task(self):
        self.current_task = random.randint(0, self.tasks - 1)

    def sample(self):

        self.counter = (self.counter + 1) % 20
        if random.random() < 0.1:
            self.change_task()
        # if random.random() < 0.1:
        if self.counter == 0:
            # print("Changing random function")
            self.generate_task(random.randint(0, self.tasks - 1))
        if self.current_task not in self.iterators:
            self.generate_task(self.current_task)

        return self.get_task_data(self.current_task)

    def get_task_data(self, task):

        task_id = task
        x_samples = np.random.rand(1) * 10 - 5

        # x_samples = 5
        x = np.zeros((1, self.capacity + 1))
        x[:, 0] = x_samples
        # print(task_id, self.capacity)
        assert (task_id <= self.capacity)
        x[:, task_id + 1] = 1

        # if task == 0:
        #     print( self.iterators[task]['amplitude'], task)
        targets = self.iterators[task]['amplitude'] * np.sin(x_samples + self.iterators[task]['phase'])

        return torch.tensor(x).float(), torch.tensor(targets).float()


if __name__ == "__main__":
    sampler = pattern(5)
    for a in range(100000):
        print(sampler.sample())

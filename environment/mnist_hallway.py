import random

import torch
from torchvision import datasets, transforms


class MnistHallwayControl:
    def __init__(self, hallway_length, dataset_dir="../"):
        '''

        Args:
            hallway_length: Has to be at-least 28 since mnist image has 28 rows.
            dataset_dir: Directory where the dataset will be downloaded. You could pass $SLURM_TMPDIR directory here to speed up data-loading on Compute Canada.
        '''

        self.hallway_length = hallway_length
        self.hallway = torch.zeros(hallway_length + 2, 5)
        self.hallway[1, 2] = 2  # State with image
        self.hallway[2:hallway_length + 1, 2] = 1  # Normal state
        self.hallway[
            hallway_length, 1] = -1  # Terminal state; if even number in first observation, reward +1. Otherwise, -1
        self.hallway[hallway_length, 3] = -2  # Terminal state; if odd number in first state, reward +1, otherwise, -1

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0,), (0.3081,))
        ])
        self.dataset = datasets.MNIST(dataset_dir, train=True, transform=self.transform, download=True)
        self.requires_reset = True;

    def reset(self):
        self.location = [1, 2]
        index = random.randint(0, len(self.dataset.data) - 1)
        observation, y = self.dataset.__getitem__(index)
        observation = observation.squeeze()
        self.current_episode_observation = observation
        self.target = y % 2
        self.requires_reset = False
        return self.current_episode_observation

    def step(self, action):
        '''

        Args:
            action: 0 = right, 1 = down, 2 = left, 3 = up

        Returns:
            Observation (28 x 28 matrix), reward (scaler), and is_terminal (boolean).
        '''

        if self.requires_reset:
            assert False, "Environment requires reset in the beginning or after reaching terminal state; call env.reset()"
        if action == 0:
            possible_location = self.hallway[self.location[0] + 1, self.location[1]]
            if possible_location != 0:
                self.location[0] += 1
        elif action == 1:
            possible_location = self.hallway[self.location[0], self.location[1] + 1]
            if possible_location != 0:
                self.location[1] += 1
        elif action == 2:
            possible_location = self.hallway[self.location[0] - 1, self.location[1]]
            if possible_location != 0:
                self.location[0] -= 1
        elif action == 3:
            possible_location = self.hallway[self.location[0], self.location[1] - 1]
            if possible_location != 0:
                self.location[1] -= 1
        else:
            print("Invalid action")
            assert (False)
        new_location = self.hallway[self.location[0], self.location[1]]
        print(new_location)
        if new_location == 2:
            return self.current_episode_observation, 0, False
        elif new_location == 0 or new_location == 1:
            return self.current_episode_observation * 0, 0, False
        elif new_location == -1:
            self.requires_reset = True;
            if self.target == 0:
                return self.current_episode_observation * 0, 1, True
            elif self.target == 1:
                return self.current_episode_observation * 0, -1, True
            else:
                assert (False)
        elif new_location == -2:
            self.requires_reset = True;
            if self.target == 0:
                return self.current_episode_observation * 0, -1, True
            elif self.target == 1:
                return self.current_episode_observation * 0, 1, True
            else:
                assert (False)

        assert (False)

    def get_location(self):
        return [self.location[0] - 1, self.location[1] - 1]


class MnistHallwayPrediction:
    def __init__(self, hallway_length, dataset_dir="../"):
        '''

        Args:1
            hallway_length: Has to be at-least 28 since mnist image has 28 rows.
            dataset_dir: Directory where the dataset will be downloaded. You could pass $SLURM_TMPDIR directory here to speed up data-loading on Compute Canada.
        '''
        assert (hallway_length > 27)
        self.hallway_length = hallway_length

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0,), (0.3081,))
        ])

        self.dataset = datasets.MNIST(dataset_dir, train=True, transform=self.transform, download=True)

    def reset(self):
        self.t = 0
        print(len(self.dataset.data))
        index = random.randint(0, len(self.dataset.data) - 1)
        observation, y = self.dataset.__getitem__(index)
        observation = observation.squeeze()
        self.current_episode_observation = observation
        if y % 2 == 0:
            self.episode_reward = 1
        else:
            self.episode_reward = -1
        return self.current_episode_observation[self.t, :]

    def step(self, action=None):
        '''

        Args:
            action: irrelevant; this is a simple prediction environment. Actions are ignored.

        Returns:
            Observation, reward, and is_terminal.
        '''
        self.t += 1
        if self.t < 28:
            obs = self.current_episode_observation[self.t, :]
        else:
            obs = self.current_episode_observation[0, :] * 0.0

        if self.t + 1 == self.hallway_length:
            return obs, self.episode_reward, True
        return obs, 0, False

    def render(self):
        image = torch.zeros(28, 28 * self.hallway_length)
        image[:, 0:28] = self.current_episode_observation;
        return image


class MnistHallwayMultiplePredictionse:
    def __init__(self, hallway_length, dataset_dir="../"):
        '''

        Args:
            hallway_length: Has to be at-least 28 since mnist image has 28 rows.
            dataset_dir: Directory where the dataset will be downloaded. You could pass $SLURM_TMPDIR directory here to speed up data-loading on Compute Canada.
        '''
        assert (hallway_length > 27)
        self.hallway_length = hallway_length

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0,), (0.3081,))
        ])

        self.dataset = datasets.MNIST(dataset_dir, train=True, transform=self.transform, download=True)
        self.t = 0

    def __new_image(self):
        self.t = 1
        index = random.randint(0, len(self.dataset.data) - 1)
        observation, y = self.dataset.__getitem__(index)
        observation = observation.squeeze()
        target_row = torch.zeros(1, 28)
        observation_total = torch.cat([observation, target_row], 0)
        # print(observation_total.shape)
        self.current_episode_observation = observation_total.clone()
        self.current_target = torch.zeros(28)
        self.current_target[y] = 10
        observation_total[1:28, :] = 0
        return observation_total

    def step(self, action=None):
        '''

        Args:
            action: irrelevant; this is a simple prediction environment. Actions are ignored.

        Returns:
            Observation, reward, and is_terminal.
        '''
        if self.t % 50 == 0:
            self.current_observation = self.__new_image()
            return self.current_observation.flatten()
        else:
            self.t += 1
            if self.t < 28:
                assert (self.t > 0)
                observation = self.current_episode_observation.clone()
                observation[0:self.t, :] = 0
                observation[self.t + 1:28] = 0
                self.current_observation = observation
                return self.current_observation.flatten()
            else:
                observation = self.current_episode_observation.clone() * 0
                if self.t == 28:
                    # print(self.current_target)
                    observation[28, :] = self.current_target
                self.current_observation = observation
                return self.current_observation.flatten()

    def render(self):
        return self.current_observation


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # fig, axs = plt.subplots(6, 6)
    env = MnistHallwayControl(5, "../../")
    done = False
    counter = 1
    outer = 0
    sum_image = 0

    for a in range(5):
        done = False
        obs = env.reset()
        plt.imshow(obs)
        plt.show()
        while not done:
            action = int(input())
            obs, reward, done = env.step(action)
            plt.imshow(obs)
            plt.show()
            print(env.get_location(), reward, done)

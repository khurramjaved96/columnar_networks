import torch
import numpy
import copy
import random
import logging

logger = logging.getLogger('experiment')

class Sarsa:
    def __init__(self, epsilon, action_set, min_values, max_values, bins):
        self.action_set = action_set
        self.min_values = min_values
        self.max_values = max_values
        self.bins = bins
        self.e = epsilon


    def get_action(self, model, actions_set, state):

        q_values = []
        for action in actions_set:
            state_copy = state.clone()
            state_copy[action] = 1

            q_value = model.forward_no_state_update(state_copy)
            q_values.append(q_value)

        greedy_action = torch.argmax(torch.stack(q_values))
        if random.random() < self.e:
            return random.randint(0, len(actions_set)-1)
        return greedy_action.item()

    def get_value(self, model, state, action):
        state = state.clone()
        state[action] = 1
        return model.forward_no_state_update(state)

    def forward(self, model, state, action):
        state  = state.clone()
        state[action] = 1
        return model.forward(state)

    def step(self, env, model, actions_set, update_lr, train=True):

        # print("Initial hidden state", model.hidden_state)
        # print("Initial hidden state", model.old_hidden_state)
        S = self.DNN_input(env.reset())
        A = self.get_action(model, actions_set, S)
        done = False
        return_val = 0
        # print(model.hidden_state)
        counter=1
        online_td_error = 0
        while not done:
            counter +=1

            # if counter ==3 or counter == 4:
            #     print("Old", model.old_hidden_state)
            #     print("New", model.hidden_state)

            S_, R_, done, info = env.step(A)
            S_ = self.DNN_input(S_)

            return_val += R_
            if done:
                target = R_
            else:
                A_ = self.get_action(model, actions_set, S_)
                target = R_ + self.get_value(model, S_, A_)
                if counter ==2:
                    logger.info("Bootstraped return = %s", str(target))

            val_cur = self.forward(model, S, A)
            if done:
                # print("Terminal state val", val_cur, target)
                logger.info("Root Terminal Error = %s %s", counter, torch.abs(target - val_cur))
            online_td_error += torch.abs(target - val_cur)
            model.gradient_computation()
            model.accumulate_gradients(target)


            if train:

                model.update_params(update_lr)

            model.zero_grad()
            S = S_
            A = A_

        # print("End", model.hidden_state)
        logger.info("Online td error = %s %s", online_td_error/counter, counter)
        return return_val

    def DNN_input(self, values):

        vectors = [torch.zeros(len(self.action_set))]

        for counter in range(len(values)):
            if self.bins[counter] == 0:
                pass
            else:
                vector = torch.zeros(self.bins[counter])
                index = (values[counter] - self.min_values[counter]) / (self.max_values[counter] - self.min_values[counter])
                index = int(index * (self.bins[counter] - 1))
                vector[index] = 1
                vectors.append(vector)

        # print(vectors)
        final_vector = torch.cat(vectors)
        # print(final_vector.shape)
        # print(final_vector)
        # quit()
        return final_vector







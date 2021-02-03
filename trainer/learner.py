import logging
import random

import torch
from torch import nn
from torch import optim


logger = logging.getLogger("experiment")


class TD_MC(nn.Module):

    def __init__(self, args, config, device, backbone_config=None):
        """
        #
        :param args:
        """
        super(TD_MC, self).__init__()

        self.args = args
        self.device = device
        self.update_lr = args["update_lr"]
        self.meta_lr = args["meta_lr"]
        self.plasticity = args['plasticity']

        self.load_model(args, config, backbone_config)
        self.optimizers = []

        forward_meta_weights = self.net.get_forward_meta_parameters()

        if len(forward_meta_weights) > 0:
            self.optimizer_forward_meta = optim.Adam(forward_meta_weights, lr=self.meta_lr)
            self.optimizers.append(self.optimizer_forward_meta)

        if self.plasticity:
            self.net.add_static_plasticity()
            self.optimizer_static_plasticity = optim.SGD(self.net.static_plasticity, lr=args["plasticity_lr"])
            self.optimizers.append(self.optimizer_static_plasticity)

        if args['model_path'] is not None:
            self.load_weights(args)

        self.log_model()

    def log_model(self):
        for name, param in self.net.named_parameters():
            logger.warning("Param name %s", name)
            if param.meta:
                logger.info("Weight in meta-optimizer = %s %s", name, str(param.shape))
            if param.adaptation:
                logger.debug("Weight for adaptation = %s %s", name, str(param.shape))

    def optimizer_zero_grad(self):
        for opti in self.optimizers:
            opti.zero_grad()

    def optimizer_step(self):
        for opti in self.optimizers:
            opti.step()

    def load_model(self, args, config, context_config):
        if args['model_path'] is not None and False:
            pass
            assert (False)

        else:
            self.net = Learner.Learner(config, context_config)

    def load_weights(self, args):
        if args['model_path'] is not None:
            net_old = self.net
            net = torch.load(args['model_path'],
                             map_location="cpu")

            for (n1, old_model) in net_old.named_parameters():
                for (n2, loaded_model) in net.named_parameters():

                    if old_model.data.shape == loaded_model.data.shape and n1 == n2:
                        loaded_model.adaptation = old_model.adaptation
                        loaded_model.meta = old_model.meta

                        old_model.data = loaded_model.data

    def inner_update(self, net, vars, grad, adaptation_lr):
        adaptation_weight_counter = 0

        new_weights = []

        status = False
        if random.random() > 0.999:
            status = True
        for p in vars:
            if p.adaptation:
                g = grad[adaptation_weight_counter]
                # print(g.shape)
                if self.plasticity:
                    mask = net.static_plasticity[adaptation_weight_counter].view(g.shape)
                    g = g * torch.exp(mask)
                    if status:
                        logger.info("Layer %s %s %s", str(adaptation_weight_counter),
                                    str(torch.max(torch.exp(mask).flatten()).item()),
                                    str(torch.min(torch.exp(mask).flatten()).item()))

                temp_weight = torch.clamp(p - adaptation_lr * g, -5, 5)
                temp_weight.adaptation = p.adaptation
                temp_weight.meta = p.meta
                new_weights.append(temp_weight)
                adaptation_weight_counter += 1
                # print(temp_weight)
            else:
                new_weights.append(p)

        return new_weights

    def get_action(self, state):
        if state[1] > 0:
            A_prime = 2
        else:
            A_prime = 0
        return A_prime

    def estimate_value_error(self, env):

        total_loss_offline = 0
        total_steps = 0
        for steps in range(10):

            total_steps += 1

            total_reward, done = 0, False
            states, actions = [], []

            State = env.reset()
            A = self.get_action(State)
            states.append(State)
            actions.append(A)

            while not done:

                State, R, done, info = env.step(A)
                total_reward += R

                if not done:
                    A = self.get_action(State)
                    total_steps += 1
                    states.append(State)
                    actions.append(A)

            ground_truth_targets = []

            for x in range(int(total_reward * -1)):
                ground_truth_targets.append((total_reward * -1 - x) * -1)

            target_lists_offline = []
            for state, action in zip(states, actions):
                value_distribution = self.net(State.view(1, -1).to(self.device)).squeeze()
                target_lists_offline.append(value_distribution[action].detach())


            for a, b in zip(ground_truth_targets, target_lists_offline):
                total_loss_offline += (a - b) ** 2

        return total_loss_offline/total_steps

    def forward(self, env):

        fast_weights = self.net.parameters()

        total_reward, regret_with_gradients, online_error, step, done = 0, 0, 0, 0, False
        states, actions = [], []

        state = env.reset()
        A = self.get_action(state)
        step += 1
        states.append(state)
        actions.append(A)

        while not done:


            state, reward, done, info = env.step(A)
            total_reward += reward

            if not done:
                A = self.get_action(state)
                step += 1
                states.append(state)
                actions.append(A)

        ground_truth_targets = []
        for x in range(int(total_reward * -1)):
            ground_truth_targets.append((total_reward * -1 - x)*-1)

        cur_step = 0
        for state, action, target in zip(states, actions, ground_truth_targets):
            cur_step += 1
            value_distribution = self.net(state.view(1, -1).to(self.device), fast_weights).squeeze()

            error = (target - value_distribution[action]) ** 2
            grad = torch.autograd.grad(error, self.net.get_adaptation_parameters(fast_weights), create_graph=True)

            online_error += error.detach()
            regret_with_gradients = regret_with_gradients + error
            fast_weights = self.inner_update(self.net, fast_weights, grad, self.update_lr)

            average_loss = regret_with_gradients/cur_step

            if len(self.optimizers) > 0:
                # print("Gets here", cur_step)
                self.optimizer_zero_grad()
                average_loss.backward(retain_graph=True)
                self.optimizer_step()
                # self.net.update_weights(fast_weights)

        self.net.update_weights(fast_weights)
        self.net.cutoff_lr(12)

        return average_loss.detach()



def main():
    pass


if __name__ == '__main__':
    main()

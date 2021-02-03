import logging
import random

import torch
from torch.optim import Adam

import configs.classification.gradient_alignment as reg_parser
from experiment.experiment import experiment
from model.colnn_final import Recurrent_Network
from utils import utils

logger = logging.getLogger('experiment')


def main():
    p = reg_parser.Parser()
    total_seeds = len(p.parse_known_args()[0].seed)
    rank = p.parse_known_args()[0].rank
    all_args = vars(p.parse_known_args()[0])

    args = utils.get_run(all_args, rank)

    my_experiment = experiment(args["name"], args, args["output_dir"], commit_changes=False,
                               rank=int(rank / total_seeds),
                               seed=total_seeds)

    my_experiment.results["all_args"] = all_args



    logger = logging.getLogger('experiment')

    gradient_error_list = []
    gradient_alignment_list = []

    for seed in range(args["runs"]):
        utils.set_seed(args["seed"] + seed + seed*args["seed"])
        n = Recurrent_Network(50, args['columns'], args["width"],
                              args["sparsity"])
        error_grad_mc = 0

        rnn_state = torch.zeros(args['columns'])
        n.reset_TH()

        for ind in range(50):

            x = torch.bernoulli(torch.zeros(1, 50) + 0.5)

            _, _, grads = n.forward(x, rnn_state, grad=True, retain_graph=False, bptt=False)

            value_prediction, rnn_state, _ = n.forward(x, rnn_state, grad=False,
                                                       retain_graph=False, bptt=True)

            n.update_TH(grads)

            target_random = random.random() * 100 - 50
            real_error = (0.5) * (target_random - value_prediction) ** 2
            error_grad_mc += real_error

            n.accumulate_gradients(target_random, value_prediction, hidden_state=rnn_state)

        grads = torch.autograd.grad(error_grad_mc, n.parameters())

        counter = 0
        total_sum = 0
        positive_sum = 0
        dif = 0

        for named, param in n.named_parameters():

            dif += torch.abs(n.grads[named] - grads[counter]).sum()
            positive = ((n.grads[named] * grads[counter]) > 1e-10).float().sum()
            total = positive + ((n.grads[named] * grads[counter]) < - 1e-10).float().sum()
            total_sum += total
            positive_sum += positive

            counter += 1

        logger.error("Difference = %s", (float(dif) / total_sum).item())
        gradient_error_list.append( (float(dif) / total_sum).item())
        gradient_alignment_list.append(str(float(positive_sum) / float(total_sum)))
        logger.error("Grad alignment %s", str(float(positive_sum) / float(total_sum)))




        my_experiment.add_result("abs_error", str(gradient_error_list))
        my_experiment.add_result("alignment", str(gradient_alignment_list))

        my_experiment.store_json()


if __name__ == '__main__':
    main()

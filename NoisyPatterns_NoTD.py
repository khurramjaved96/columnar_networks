import logging

import torch
from torch.optim import Adam

import configs.classification.gradient_alignment as reg_parser
import environment.animal_learning as environments
from experiment.experiment import experiment
from model.gru import Recurrent_Network
from utils import utils

lambda_val = 0.5
gamma = 1 - (1 / 20)
env = environments.TracePatterning(seed=0, ISI_interval=(20, 30), ITI_interval = (80, 120), gamma =  gamma, num_CS = 8, num_activation_patterns = 8, activation_patterns_prob= 0.5, num_distractors = 10, activation_lengths= {"CS": 1, "US": 1, "distractor": 1}, noise= 0.1)
env.reset()

logger = logging.getLogger('experiment')

p = reg_parser.Parser()
total_seeds = len(p.parse_known_args()[0].seed)
rank = p.parse_known_args()[0].rank
all_args = vars(p.parse_known_args()[0])

args = utils.get_run(all_args, rank)

my_experiment = experiment(args["name"], args, args["output_dir"], sql=True,
                           rank=int(rank / total_seeds),
                           seed=total_seeds)

my_experiment.results["all_args"] = all_args
my_experiment.make_table("metrics", {"rank": 0, "error": 0.0, "step": 0}, ("rank", "step"))
list_to_store = []

logger = logging.getLogger('experiment')

gradient_error_list = []
gradient_alignment_list = []
running_error = 1
for seed in range(args["runs"]):
    utils.set_seed(args["seed"] + seed + seed * args["seed"])
    n = Recurrent_Network(19, args['columns'], args["width"],
                          args["sparsity"])
    error_grad_mc = 0

    rnn_state = torch.zeros(args['columns'])
    n.reset_TH()
    x = torch.tensor(env.observation()).unsqueeze(0).float()
    opti = Adam(n.parameters(), args["meta_lr"], (0.5, 0.99))
    for ind in range(5000000):

        rnn_state = rnn_state.detach()
        x = torch.tensor(env.observation()).unsqueeze(0).float()
        env.step(0)
        print(x.shape)
        quit()
        _, _, grads = n.forward(x, rnn_state, grad=True, retain_graph=False, bptt=False)

        value_prediction, rnn_state, _ = n.forward(x, rnn_state, grad=False,
                                                   retain_graph=False, bptt=False)

        n.update_TH(grads)

        target_random = env.observation()[0] * 10
        x_new = torch.tensor(env.observation()).unsqueeze(0).float()
        targ, _, _ = n.forward(x_new, rnn_state, grad=False, retain_graph=False, bptt=False)
        if target_random == 0:
            target_random = target_random + gamma * targ.detach().item()

        # target_random = random.random() * 100 - 50
        real_error = (0.5) * (target_random - value_prediction) ** 2
        running_error = running_error * 0.995 + real_error.detach().item() * 0.005
        if ind % 10 == 0:
            list_to_store.append((rank, running_error, ind))
            # print(running_error)
        if ind % 1000 == 0:
            logger.info("Step %f, Target %f, Pred %f", ind, target_random, value_prediction.item())
            logger.info("Running error %f", running_error)
        if ind % 50000 == 0:
            keys = ["rank", "error", "step"]
            my_experiment.insert_values("metrics", keys, list_to_store)
            list_to_store = []


        n.decay_gradient(lambda_val *gamma)
        n.compute_and_accumulate_gradients(value_prediction.detach(), hidden_state=rnn_state)
        opti.zero_grad()
        n.copy_gradient_with_error(target_random - value_prediction)
        opti.step()

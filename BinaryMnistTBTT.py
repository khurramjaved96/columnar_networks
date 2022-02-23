import logging

import torch
from torch.optim import SGD

import configs.classification.gradient_alignment as reg_parser

from experiment.experiment import experiment
from utils import utils
from torch import nn
from model import lstm
from copy import deepcopy
from FlexibleNN import BinaryMnistLoader
from timeit import default_timer as timer
from datetime import datetime, timedelta

gamma = 0.8

logger = logging.getLogger('experiment')

p = reg_parser.Parser()
total_seeds = len(p.parse_known_args()[0].seed)
rank = p.parse_known_args()[0].run
all_args = vars(p.parse_known_args()[0])

args = utils.get_run(all_args, rank)

my_experiment = experiment(args["name"], args, args["output_dir"], sql=True,
                           run=int(rank / total_seeds),
                           seed=total_seeds)

my_experiment.results["all_args"] = all_args
my_experiment.make_table("error_table", {"run": 0, "step": 0, "error": 0.0}, ("run", "step"))
my_experiment.make_table("predictions", {"run": 0, "global_step": 0, "step": 0, "pred":0.0, "target":0.0}, ("run", "global_step"))
error_table_keys = ["run", "step", "error"]
predictions_table_keys = ["run", "global_step", "step", "pred", "target"]

error_list = []
predictions_list = []

logger = logging.getLogger('experiment')

gradient_error_list = []
gradient_alignment_list = []
running_error = 0.05
hidden_units = args["features"]

utils.set_seed(args["seed"])
env = BinaryMnistLoader(args["seed"])

torch.set_num_threads(1)

h = torch.zeros(1, 1, hidden_units).float()
c = torch.zeros(1, 1, hidden_units).float()

n = lstm.LSTMNet(28, hidden_units)
opti = SGD(n.parameters(), args["step_size"])
list_of_observations = []

pred = torch.tensor(0.0)
pred.requires_grad = True

row_x = None
global_step = 0
ITI_steps = 10
start = timer()
for i in range(0, 2001000):
    input_x, input_y = env.get_data()
    input_x = torch.tensor(input_x)
    input_y = torch.tensor(input_y)
    for row in range(0, 28 + ITI_steps):
        global_step += 1
        if row_x != None:
            list_of_observations.append(row_x)
            old_row_x = row_x
        if row < 28:
            row_x = input_x[row*28:(row+1)*28].view(1,1,-1)/256.0
            gt_target = input_y * (gamma ** (27 - row))
        else:
            row_x = torch.zeros(1,1,28).float()
            gt_target = 0

        if(len(list_of_observations) == args["truncation"]):
            x = list_of_observations[0].view(1,1,-1)
            _, (h, c) = n(x, (h, c))
            h_temp = h
            c_temp = c
            x = torch.stack(list_of_observations[1:]).squeeze(dim=1)
            pred, (h_temp, c_temp) = n(x, (h_temp, c_temp))
            pred = pred[-1]

#            for inner_counter in range(0, len(list_of_observations)):
#                x = list_of_observations[inner_counter]
#                if inner_counter == 0:
#                    _, (h, c) = n(x, (h, c))
#                    h_temp = h
#                    c_temp = c
#                else:
#                    pred, (h_temp, c_temp) = n(x, (h_temp, c_temp))

            n_copy = deepcopy(n)
            with torch.no_grad():
                next_pred, _ = n_copy(row_x, (h_temp.detach(), c_temp.detach()))
            if row == 27:
                target = input_y.detach() + gamma * next_pred.detach().item()
            else:
                target = 0 + gamma * next_pred.detach().item()

            real_error = (target - pred)**2
            gt_error = (gt_target - pred)**2
            n.decay_gradients(args["lambda"]*gamma)
            real_error.backward()
            opti.step()
            # print(gt_target, target, pred)
            h = h.detach()
            c = c.detach()
            running_error = running_error * 0.9999 + gt_error.detach().item() * 0.0001
            list_of_observations.pop(0)
    if(i%100 == 0):
        print(timedelta(seconds = timer() - start))
        start = timer()
        print("Step", i, "Running error = ", running_error)

import logging

import torch
from torch.optim import SGD

import configs.classification.gradient_alignment as reg_parser
import environment.animal_learning as environments
from experiment.experiment import experiment
from utils import utils
from torch import nn
from model import lstm
from copy import deepcopy

gamma = 0.9

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
my_experiment.make_table("predictions", {"run": 0, "step": 0, "x0":0, "x1":0, "x2":0, "x3":0, "x4":0, "x5":0, "x6":0, "pred":0.0, "target":0.0}, ("run", "step"))
error_table_keys = ["run", "step", "error"]
predictions_table_keys = ["run", "step", "x0", "x1", "x2", "x3", "x4", "x5", "x6", "pred", "target"]

error_list = []
predictions_list = []

logger = logging.getLogger('experiment')

gradient_error_list = []
gradient_alignment_list = []
running_error = 0.05
hidden_units = args["features"]

utils.set_seed(args["seed"])
env = environments.TracePatterning(seed=args["seed"], ISI_interval=(14, 26), ITI_interval = (80, 120), gamma =  0.9, num_CS = 6, num_activation_patterns = 10, activation_patterns_prob= 0.5, num_distractors = 5, activation_lengths= {"CS": 1, "US": 1, "distractor": 1}, noise= 0)

h = torch.zeros(1, 1, hidden_units).float()
c = torch.zeros(1, 1, hidden_units).float()

input = torch.tensor(env.reset().observation).view(1, 1, -1).float()
n = lstm.LSTMNet(hidden_units)
opti = SGD(n.parameters(), args["step_size"])
error_grad_mc = 0
sum_of_error = None
for i in range(0, 5000000):

    value_prediction, (h, c) = n(input, (h, c))
    # print(value_prediction)
    gt_target = env.get_real_target()

    if (i % 100000 < 400):
        temp_list = [str(rank), str(i)]
        counter = 0
        for t in input.squeeze():
            if counter!= 0 and counter < 7:
                temp_list.append(str(t.item()))
            counter+=1
        temp_list.append(str(input.squeeze()[0].item()))
        temp_list.append(str(value_prediction.item()))
        temp_list.append(str(gt_target))

        predictions_list.append(temp_list)
    input = env.step(0)
    input = torch.tensor(input.observation).view(1, 1, -1).float()
    n_copy = deepcopy(n)
    with torch.no_grad():
        next_pred, _ = n_copy(input, (h.detach(), c.detach()))

    target = input[0, 0, 0].detach() + gamma * next_pred.detach().item()

    # print(gt_target, target)
    real_error =  (target - value_prediction) ** 2
    gt_error = (gt_target - value_prediction)**2
    if sum_of_error is None:
        sum_of_error = real_error
    else:
        sum_of_error = sum_of_error +  real_error
    running_error = running_error * 0.9999 + gt_error.detach().item() * 0.0001
    if(i%args["truncation"] == 0):
        opti.zero_grad()
        sum_of_error.backward()
        opti.step()
        h = h.detach()
        c = c.detach()
        sum_of_error = None

    if (i % 50000 == 20000):
        error_list.append([str(rank), str(i), str(running_error)])
    if(i % 100000 == 4):
        my_experiment.insert_values("predictions", predictions_table_keys, predictions_list)
        predictions_list = []
        my_experiment.insert_values("error_table", error_table_keys, error_list)
        error_list = []

    if(i%100000 == 0):
        print("Step", i, "Running error = ", running_error)

#
import logging

import torch
from torch.optim import SGD, Adam

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
env = environments.TracePatterning(seed=args["seed"], ISI_interval=(4, 4), ITI_interval = (80, 120), gamma =  0.9, num_CS = 2, num_activation_patterns = 1, activation_patterns_prob= 1, num_distractors = 0, activation_lengths= {"CS": 1, "US": 1, "distractor": 1}, noise= 0)

h = torch.zeros(1, 1, hidden_units).float()
c = torch.zeros(1, 1, hidden_units).float()

input = torch.tensor(env.reset().observation).view(1, 1, -1).float()
input = input*100
n = lstm.LSTMNet(3, hidden_units)
opti = Adam(n.parameters(), args["step_size"], (0, 0.999))
error_grad_mc = 0
sum_of_error = None
list_of_observations = []
list_of_h = []
list_of_gt_target = []
for i in range(0, 5000000):


    list_of_observations.append(input)
    gt_target = env.get_real_target()
    # list_of_gt_target.append(gt_target)


    # print(input)
    old_input = input
    input = env.step(0)
    input = torch.tensor(input.observation).view(1, 1, -1).float()
    # print("Step", i, "Len", len(list_of_observations))
    if(len(list_of_observations) == args["truncation"]):

        for inner_counter in range(0, len(list_of_observations)):
            x = list_of_observations[inner_counter]
            if inner_counter == 0:
                _, (h, c) = n(x, (h, c))
                h_temp = h
                c_temp = c
            else:
                # print(x)
                pred, (h_temp, c_temp) = n(x, (h_temp, c_temp))
            # print(h_temp)
        # print(h)
        # print("")
        # print(x)
        # print(c_temp)
        n_copy = deepcopy(n)
        with torch.no_grad():
            next_pred, _ = n_copy(input, (h_temp.detach(), c_temp.detach()))
        target = input[0, 0, 0].detach() + gamma * next_pred.detach().item()

        if (i % 100000 < 400):
            temp_list = [str(rank), str(i)]
            counter = 0
            for t in old_input.squeeze():
                if counter!= 0 and counter < 7:
                    temp_list.append(str(t.item()))
                counter+=1
            temp_list.append(str(old_input.squeeze()[0].item()))
            temp_list.append(str(pred.item()))
            temp_list.append(str(gt_target))

            predictions_list.append(temp_list)
        #
        real_error = (target - pred)**2
        gt_error = (gt_target - pred)**2
        # opti.zero_grad(0.)
        n.decay_gradients(args["lambda"]*gamma)
        real_error.backward()
        opti.step()
        # print(gt_target, target, pred)
        # print("Learnign step")
        h = h.detach()
        c = c.detach()
        list_of_observations.pop(0)
        running_error = running_error * 0.9999 + gt_error.detach().item() * 0.0001
        if (i % 50000 == 20000):
            error_list.append([str(rank), str(i), str(running_error)])
        if(i % 100000 == 4):
            my_experiment.insert_values("predictions", predictions_table_keys, predictions_list)
            predictions_list = []
            my_experiment.insert_values("error_table", error_table_keys, error_list)
            error_list = []

        if(i%1000 == 0):
            print("Step", i, "Running error = ", running_error)

        #
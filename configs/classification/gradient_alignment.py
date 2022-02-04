import argparse


class Parser(argparse.ArgumentParser):
    def __init__(self):

        super().__init__()

        self.add_argument('--runs', type=int, help='epoch number', default=30)
        self.add_argument('--name', help='Name of experiment', default="oml_regression")
        self.add_argument('--output-dir', help='Name of experiment', default="../results/")
        self.add_argument('--seed', nargs='+', help='Seed', default=[90, 20, 30], type=int)
        self.add_argument('--run', type=int, help='meta batch size, namely task num', default=104)
        self.add_argument("--truncation", nargs='+', type=int, default=[27])
        self.add_argument("--features", nargs='+', type=int, default=[10])
        self.add_argument("--step_size", nargs='+', type=float, default=[1e-2])
        self.add_argument("--lambda", nargs='+', type=float, default=[0.9])

#
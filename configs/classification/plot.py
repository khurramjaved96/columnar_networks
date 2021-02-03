import configargparse


class Parser(configargparse.ArgParser):
    def __init__(self):
        super().__init__()

        self.add('--name', help='Name of experiment', default="oml_regression")

        self.add('--runs', nargs='+', help='Seed', default=[90], type=int)
        self.add('--rank', type=int, help='meta batch size, namely task num', default=0)
        self.add('--seed', nargs='+', help='Seed', default=[90], type=int)
        self.add('--dir', help='Name of experiment', default="../results/")





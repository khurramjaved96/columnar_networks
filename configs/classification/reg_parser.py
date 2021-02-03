import configargparse


class Parser(configargparse.ArgParser):
    def __init__(self):
        super().__init__()
        #

        self.add('--epoch', type=int, help='epoch number', default=500)
        self.add('--meta-lr', type=float, nargs='+', help='meta-level outer learning rate', default=[1e-7])
        self.add('--update-lr', nargs='+', type=float, help='task-level inner update learning rate',
                 default=[1e-3])
        self.add("--width", type=int,  nargs='+', default=[150])
        self.add("--width-inner", type=int, nargs='+', default=[1])
        self.add('--name', help='Name of experiment', default="oml_regression")
        self.add('--seed', nargs='+', help='Seed', default=[90], type=int)
        self.add('--runs', nargs='+', help='Seed', default=[90], type=int)
        self.add('--rank', type=int, help='meta batch size, namely task num', default=0)


        self.add('--b2', nargs='+', type=float, help='task-level inner update learning rate',
                 default=[0.99])
        self.add('--b1', nargs='+', type=float, help='task-level inner update learning rate',
                 default=[0.99])
        self.add('--output-dir', help='Name of experiment', default="../results/")
        self.add("--partial", action="store_true")




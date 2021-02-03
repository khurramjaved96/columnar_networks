import configargparse


class Parser(configargparse.ArgParser):
    def __init__(self):
        """
        #
        Returns:
            object:
        """
        super().__init__()

        self.add('--epoch', type=int, help='epoch number', default=50000000)
        self.add('--inner-lr', nargs='+', type=float, help='task-level inner update learning rate',
                 default=[1e-3])
        self.add('--meta-lr', nargs='+', type=float, help='task-level inner update learning rate',
                 default=[1e-4])
        self.add('--b2', nargs='+', type=float, help='task-level inner update learning rate',
                 default=[0.99])
        self.add('--b1', nargs='+', type=float, help='task-level inner update learning rate',
                 default=[0.9])
        self.add('--name', help='Name of experiment', default="oml_regression")
        self.add('--output-dir', help='Name of experiment', default="../results/")
        self.add('--seed', nargs='+', help='Seed', default=[90], type=int)
        self.add('--rank', type=int, help='meta batch size, namely task num', default=0)
        self.add("--width", nargs='+', type=int, default=[20])
        self.add("--sparsity", nargs='+', type=float, default=[0.01])
        self.add("--columns", nargs='+', type=int, default=[10])
        self.add("--gpus", nargs='+', type=int, default=[4])
        self.add("--capacity", nargs='+', type=int, default=[50])




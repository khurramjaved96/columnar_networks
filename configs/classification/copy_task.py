import configargparse


class Parser(configargparse.ArgParser):
    def __init__(self):
        """
        #
        Returns:
            object:
        """
        super().__init__()

        self.add('--epochs', type=int, help='epoch number', default=30)
        self.add('--name', help='Name of experiment', default="oml_regression")
        self.add('--output-dir', help='Name of experiment', default="../results/")
        self.add('--seed', nargs='+', help='Seed', default=[90], type=int)
        self.add('--rank', type=int, help='meta batch size, namely task num', default=0)
        self.add("--width", nargs='+', type=int, default=[50])
        self.add("--sparsity", nargs='+', type=float, default=[0.1])
        self.add("--columns", nargs='+', type=int, default=[20])
        self.add('--update-lr', nargs='+', type=float, help='task-level inner update learning rate',
                 default=[1e-3])
        self.add('--b2', nargs='+', type=float, help='task-level inner update learning rate',
                 default=[0.999])
        self.add('--b1', nargs='+', type=float, help='task-level inner update learning rate',
                 default=[0.9])


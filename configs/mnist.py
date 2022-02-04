import configargparse


class Parser(configargparse.ArgParser):
    def __init__(self):
        """
        #
        Returns:
            object:
        """
        super().__init__()

        self.add('--runs', type=int, help='epoch number', default=30)
        self.add('--gpus', type=int, help='epoch number', default=1)
        self.add('--name', help='Name of experiment', default="oml_regression")
        self.add('--output-dir', help='Name of experiment', default="../results/")
        self.add('--seed', nargs='+', help='Seed', default=[90], type=int)
        self.add('--run', type=int, help='meta batch size, namely task num', default=0)
        self.add("--width", nargs='+', type=int, default=[5])
        self.add("--hallway", nargs='+', type=int, default=[32])
        self.add("--steps", nargs='+', type=int, default=[10000])
        self.add("--no-sql", action='store_true', default=False)
        self.add("--sparsity", nargs='+', type=float, default=[0.1])
        self.add("--meta-lr", nargs='+', type=float, default=[1e-4])
        self.add("--columns", nargs='+', type=int, default=[20])


class ParserLSTM(configargparse.ArgParser):
    def __init__(self):
        """
        #
        Returns:
            object:
        """
        super().__init__()

        self.add('--runs', type=int, help='epoch number', default=30)
        self.add('--gpus', type=int, help='epoch number', default=1)
        self.add('--name', help='Name of experiment', default="oml_regression")
        self.add('--output-dir', help='Name of experiment', default="../results/")
        self.add('--seed', nargs='+', help='Seed', default=[90], type=int)
        self.add('--run', type=int, help='meta batch size, namely task num', default=0)
        self.add("--width", nargs='+', type=int, default=[5])
        self.add("--no-sql", action='store_true', default=False)
        self.add("--sparsity", nargs='+', type=float, default=[0.1])
        self.add("--meta-lr", nargs='+', type=float, default=[1e-4])
        self.add("--columns", nargs='+', type=int, default=[20])
        self.add("--truncation", nargs='+', type=int, default=[20])
        self.add("--steps", nargs='+', type=int, default=[20])


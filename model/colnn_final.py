import torch
import torch.nn.functional as F
from torch import nn


class Column(nn.Module):
    def __init__(self, input_features, width, total_modules, id, sparsity):
        super(Column, self).__init__()
        total_feature_size = width*(total_modules-1) + total_modules
        modified_sparisty = width/total_feature_size * sparsity
        if modified_sparisty > 1 :
            modified_sparisty = 1
        assert(modified_sparisty <=1 )
        # print(modified_sparisty)
        # quit()
        self.id = id
        self.width = width
        self.total_modules = total_modules
        self.fc1 = nn.Linear(input_features, width)
        self.fc2 = nn.Linear(width, width)

        self.f = nn.Linear(width * total_modules + total_modules, 1)
        self.i = nn.Linear(width * total_modules + total_modules, 1)

        self.f_mask = self.sparse_like(self.f.weight, modified_sparisty)
        self.i_mask = self.sparse_like(self.i.weight, modified_sparisty)

        self.f_mask[:, id*width: (id+1)*width] = 1
        self.i_mask[:, id * width: (id + 1) * width] = 1

    def sparse_like(self, x, sparsity):
        return torch.bernoulli(torch.zeros_like(x) + sparsity)

    def zero_grad(self):
        self.grads = {}
        for named, param in self.named_parameters():
            self.grads[named] = torch.zeros_like(param.data)

    def forward(self, x, phi=True, u=False, hidden=None):

        if phi:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))

        if u:
            i = torch.tanh(F.linear(x, self.i.weight * self.i_mask, self.i.bias))
            f = torch.sigmoid(F.linear(x, self.f.weight * self.f_mask , self.f.bias))
            h = f * hidden +  i
            return h
        return x


class Recurrent_Network(nn.Module):
    def __init__(self, input_dimension, hidden_nodes, column_width=10, total_neigbours=1):
        super(Recurrent_Network, self).__init__()
        self.feature_extractors = nn.ModuleList()
        for a in range(0, hidden_nodes):
            f = Column(input_dimension, column_width, hidden_nodes, a, total_neigbours)
            self.feature_extractors.append(f)

        self.hidden_nodes = hidden_nodes
        self.t = 0

        self.TH = {}
        self.grads = {}

        for named, param in self.named_parameters():

            self.TH[named] = torch.zeros_like(param.data)
            self.grads[named] = torch.zeros_like(param.data)
            # print(named)

        self.prediction_parameters = nn.Parameter(torch.zeros(hidden_nodes))
        self.grads["prediction_parameters"] = torch.zeros_like(self.prediction_parameters.data)

        torch.nn.init.normal_(self.prediction_parameters, 0, 0.1)

    def copy_gradient(self):
        for named, param in self.named_parameters():
            param.grad = self.grads[named]


    def forward(self, x, hidden_state, grad=True, retain_graph = False, bptt=False):

        features = []
        for counter, columns in enumerate(self.feature_extractors):
            f = columns.forward(x, phi=True, u=False)
            features.append(f)

        if not bptt:
            hidden_state = nn.Parameter(hidden_state)
        h_t = []
        for counter, columns in enumerate(self.feature_extractors):
            detached_feauture = []
            detached_state = []
            for inner_counter, f in enumerate(features):
                if inner_counter == counter or bptt:
                    detached_feauture.append(f.squeeze())
                    detached_state.append(hidden_state[inner_counter].squeeze())
                else:
                    detached_feauture.append(f.detach().squeeze())
                    detached_state.append(hidden_state[inner_counter].detach().squeeze())

            detached_state = torch.stack(detached_state)
            detached_feauture.append(detached_state)

            detached_feature = torch.cat(detached_feauture)
            h_i = columns.forward(detached_feature, phi=False, u=True, hidden=hidden_state[counter])
            h_t.append(h_i)

        h_t = torch.cat(h_t)
        sum = torch.sum(h_t)

        with torch.no_grad():
            grads = None
            if grad:
                grads = torch.autograd.grad(sum, [hidden_state] + list(self.parameters()), allow_unused=True, retain_graph=retain_graph)

        y = torch.sum(self.prediction_parameters * h_t)
        return y, h_t, grads

    def update_TH(self, grads):

        with torch.no_grad():
            counter = 0
            grads_dict = {}
            for name, a in self.named_parameters():
                if grads[counter + 1] is not None:
                    grads_dict[name] = grads[counter + 1]
                counter += 1

            counter = 0
            for name, a in self.named_parameters():
                if grads[counter] is not None:
                    if name in self.TH:
                        name_temp = int(name.split(".")[1])
                        self.TH[name] = grads_dict[name] + self.TH[name] * grads[0][name_temp]


                counter += 1


    def reset_weights(self):
        for elems in self.params:
            torch.nn.init.normal_(self.params[elems], 0, 0.1)

    def zero_grad(self):

        for named, param in self.named_parameters():
            if "weight" in named or "bias" in named:
                self.grads[named] = torch.zeros_like(param.data)

    def reset_TH(self):

        for named, param in self.named_parameters():
            if "weight" in named or "bias" in named:
                self.TH[named] = torch.zeros_like(param.data)

    def accumulate_gradients(self, target, y, hidden_state):
        with torch.no_grad():

            error = (target - y)

            if self.grads["prediction_parameters"] is not None:
                self.grads["prediction_parameters"] += -1 * (hidden_state) * error
            else:
                self.grads["prediction_parameters"] = -1 * (hidden_state) * error

            # print(error * hidden_state)
            for name, param in self.named_parameters():
                if name in self.TH:
                    column_num = int(name.split(".")[1])
                    if self.grads[name] is not None:
                        self.grads[name]+= -1 * error * self.prediction_parameters[column_num] * self.TH[name]
                    else:
                        self.grads[name] = -1 * error * self.prediction_parameters[column_num] * self.TH[name]


if __name__ == "__main__":
    model = Recurrent_Network(15, 10, 0.99, 0.5, 20, 0)
    masks = torch.ones(1, 15) - 0.5
    error = 0
    from torch.optim import SGD

    optimizer = SGD(model.parameters(), 1e-4)

    for a in range(0, 10):
        input_x = torch.bernoulli(masks)
        # print(input_x.shape)
        # print(input_x)
        y_temp = model.forward(input_x)

        error += (0.5) * (y_temp - 10) ** 2
        model.accumulate_gradients(10)

    optimizer.zero_grad()
    import time

    start = time.time()
    error.backward()
    aligned_sum = 0
    total_sum = 0
    print("Time", time.time() - start)

    for counter, columns in enumerate(model.feature_extractors):
        inner_counter = 0
        for named, param in columns.named_parameters():
            # print(((columns.grads[named] - param.grad)<1e-6).float())
            # print(param.grad)
            # print(columns.grads[named])
            aligned = (((columns.grads[named] * param.grad) >= 0).float()).sum()

            total = (param.grad > -9999999999999).float().sum()
            aligned_sum += aligned
            total_sum += total
            # print("Ratio = ", aligned/total, "Weight =", named)

            # print(param.grad)
            # print(columns.grads[named])
            pass
            # print(aligned_sum, total_sum)
    print("Approx value", (aligned_sum / total_sum).item())

    model.update_params(1e-4)
    model.zero_grad()

    # print(model(input_x))

    # print(input_x.shape)

import torch
import torch.nn.functional as F
from torch import nn
import random

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
        self.bias = (random.random() - 0.5)*2
        self.id = id
        self.width = width
        self.total_modules = total_modules
        self.fc1 = nn.Linear(input_features, width)
        self.fc2 = nn.Linear(width, width)

        # self.f = nn.Linear(width * total_modules + total_modules, 1)
        self.i = nn.Linear(width * total_modules + total_modules, 1)

        # self.f_mask = self.sparse_like(self.f.weight, modified_sparisty)
        self.i_mask = self.sparse_like(self.i.weight, modified_sparisty)

        # self.f_mask[:, id*width: (id+1)*width] = 1
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
            # f = torch.sigmoid(F.linear(x, self.f.weight * self.f_mask , self.f.bias+10))
            # print(f)
            h = hidden +  i
            # print(h)
            return h
        return x

class ColumnFF(nn.Module):
    def __init__(self, input_features, width, total_modules, id, sparsity):
        super(ColumnFF, self).__init__()
        total_feature_size = width*(total_modules-1) + total_modules
        modified_sparisty = width/total_feature_size * sparsity
        if modified_sparisty > 1 :
            modified_sparisty = 1
        assert(modified_sparisty <=1 )
        # print(modified_sparisty)
        # quit()
        self.bias = (random.random() - 0.5)*2
        self.id = id
        self.width = width
        self.total_modules = total_modules
        self.fc1 = nn.Linear(input_features, width)
        self.fc2 = nn.Linear(width, width)

        # self.f = nn.Linear(width * total_modules + total_modules, 1)
        self.i = nn.Linear(width * total_modules + total_modules, 1)

        # self.f_mask = self.sparse_like(self.f.weight, modified_sparisty)
        self.i_mask = self.sparse_like(self.i.weight, modified_sparisty)

        # self.f_mask[:, id*width: (id+1)*width] = 1
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
            # f = torch.sigmoid(F.linear(x, self.f.weight * self.f_mask , self.f.bias+10))
            # print(f)
            h = hidden*0 +  i
            # print(h)
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

class Meta_LearnerFF(nn.Module):
    def __init__(self, input_dimension, hidden_nodes, column_width=10, total_neigbours=1):
        super(Meta_LearnerFF, self).__init__()
        self.feature_extractors = nn.ModuleList()
        for a in range(0, hidden_nodes):
            f = ColumnFF(input_dimension, column_width, hidden_nodes, a, total_neigbours)
            self.feature_extractors.append(f)

        self.hidden_nodes = hidden_nodes
        self.t = 0

        self.TH = {}
        self.grads = {}
        self.TW = {}

        for named, param in self.named_parameters():

            self.TH[named] = torch.zeros_like(param.data)
            self.grads[named] = torch.zeros_like(param.data)
            self.TW[named] = torch.zeros_like(param.data)


    def copy_gradient(self):
        for named, param in self.named_parameters():
            param.grad = self.grads[named]


    def online_update(self, update_lr, rnn_state, target, y, prediction_params):
        error = (target - y)

        prediction_params = prediction_params + update_lr * error * rnn_state
        return prediction_params

    def forward(self, x, hidden_state, grad=True, retain_graph = False, bptt=False, prediction_params=None):

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

        y = torch.sum(prediction_params * h_t)
        return y, h_t, grads

    def update_TW(self, inner_lr, old_state, target, y, prediction_params):
        error = (target - y)
        with torch.no_grad():
            for name, a in self.named_parameters():
                if name in self.TW:
                    column_num = int(name.split(".")[1])
                    self.TW[name] = self.TW[name] + inner_lr * error * self.TH[name] - inner_lr*old_state[column_num]*(prediction_params[column_num]*self.TH[name] + old_state[column_num]*self.TW[name])


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
                self.TW[named] = torch.zeros_like(param.data)

    def accumulate_gradients(self, target, y, hidden_state, prediction_params):
        with torch.no_grad():

            error = (target - y)

            # if self.grads["prediction_parameters"] is not None:
            #     self.grads["prediction_parameters"] += -1 * (hidden_state) * error
            # else:
            #     self.grads["prediction_parameters"] = -1 * (hidden_state) * error

            # print(error * hidden_state)
            for name, param in self.named_parameters():
                if name in self.TH:
                    column_num = int(name.split(".")[1])
                    if self.grads[name] is not None:
                        # print(hidden_state.shape, self.TW)
                        self.grads[name]+= -1 * error * (prediction_params[column_num] * self.TH[name] + self.TW[name]*hidden_state[column_num])
                    else:
                        self.grads[name] = -1 * error * (prediction_params[column_num] * self.TH[name] + self.TW[name]*hidden_state[column_num])


import numpy as np
import torch
from torch import nn
from multiprocessing import Pool

class Recurrent_Network(nn.Module):
    def __init__(self, input_dimension, total_columns, column_width=10, total_neigbours=1, device="cpu"):
        super(Recurrent_Network, self).__init__()
        self.fc1_weight = nn.Parameter(torch.zeros(input_dimension, column_width, total_columns))
        self.fc1_bias = nn.Parameter(torch.zeros(column_width, total_columns))
        # self.fc2_weight = nn.Parameter(torch.zeros(column_width, column_width, total_columns))
        # self.fc2_bias = nn.Parameter(torch.zeros(column_width, total_columns))
        # self.fc3_weight = nn.Parameter(torch.zeros(column_width, column_width, total_columns))
        # self.fc3_bias = nn.Parameter(torch.zeros(column_width, total_columns))

        self.forget_weight = nn.Parameter(torch.zeros(column_width, total_columns))
        self.input_weight = nn.Parameter(torch.zeros(column_width, total_columns))

        self.forget_bias = nn.Parameter(torch.zeros(total_columns))
        self.input_bias = nn.Parameter(torch.zeros(total_columns))

        for named, param in self.named_parameters():
            torch.nn.init.uniform_(param, -1 * np.sqrt(1 / column_width), np.sqrt(1 / column_width))

        self.no_of_hidden_nodes = total_columns
        self.t = 0

        self.TH = {}
        self.grads = {}
        self.grads_scaled = {}

        for named, param in self.named_parameters():
            self.TH[named] = torch.zeros_like(param.data).to(device)
            self.grads[named] = torch.zeros_like(param.data).to(device)
            self.grads_scaled[named] = torch.zeros_like(param.data).to(device)
            # print(named)

        self.prediction_parameters = nn.Parameter(torch.zeros(total_columns, 10))
        self.grads["prediction_parameters"] = torch.zeros_like(self.prediction_parameters.data).to(device)
        self.grads_scaled["prediction_parameters"] = torch.zeros_like(self.prediction_parameters.data).to(device)

        # torch.nn.init.uniform_(self.prediction_parameters,-1*np.sqrt(1/total_columns), np.sqrt(1/total_columns) )

    def copy_gradient(self):
        for named, param in self.named_parameters():
            param.grad = self.grads[named]

    def copy_gradient_with_error(self, target):
        for named, param in self.named_parameters():
            param.grad = self.grads[named] * target - self.grads_scaled[named]

    def forward(self, x, hidden_state, grad=True, retain_graph=False, bptt=False):

        # features = []
        # for counter, columns in enumerate(self.feature_extractors):
        #     f = columns.forward(x, phi=True, u=False)
        #     features.append(f)

        if not bptt:
            hidden_state = nn.Parameter(hidden_state)

        x = x.view(-1, 1, 1)
        # x_fc1 = torch.relu(torch.sum(x * self.fc1_weight, 0) + self.fc1_bias).unsqueeze(1)
        # x_fc2 = torch.relu(torch.sum(x_fc1 * self.fc2_weight, 0) + self.fc2_bias).unsqueeze(1)
        # x_fc3 = torch.relu(torch.sum(x_fc1 * self.fc3_weight, 0) + self.fc3_bias)

        # i = torch.tanh(torch.sum(x_fc3 * self.input_weight, 0) + self.input_bias)
        # f = torch.sigmoid(torch.sum(x_fc3 * self.forget_weight, 0) + self.forget_bias)

        x_fc1 = torch.relu(torch.sum(x * self.fc1_weight, 0) + self.fc1_bias)
        i = torch.tanh(torch.sum(x_fc1 * self.input_weight, 0) + self.input_bias)
        f = torch.sigmoid(torch.sum(x_fc1 * self.forget_weight, 0) + self.forget_bias)

        h_t = hidden_state * f + i * (1 - f)

        sum = torch.sum(h_t)

        with torch.no_grad():
            grads = None
            if grad:
                grads = torch.autograd.grad(sum, [hidden_state] + list(self.parameters()), allow_unused=True,
                                            retain_graph=retain_graph)

        y = torch.sigmoid(torch.sum(self.prediction_parameters * h_t.view(-1, 1), dim=0))
        return y, h_t, grads

    def update_TH(self, grads):

        with torch.no_grad():
            counter = 0
            grads_dict = {}
            for name, a in self.named_parameters():
                # Counter starts from +1 because first element is the gradient w.r.t the hidden state
                if grads[counter + 1] is not None:
                    grads_dict[name] = grads[counter + 1]
                counter += 1

            counter = 0
            for name, a in self.named_parameters():
                if grads[counter] is not None:
                    if name in self.TH:
                        if len(self.TH[name].shape) == 3:
                            self.TH[name] = grads_dict[name] + self.TH[name] * grads[0].view(1, 1, -1)
                        elif len(self.TH[name].shape) == 2:
                            self.TH[name] = grads_dict[name] + self.TH[name] * grads[0].view(1, -1)
                        elif len(self.TH[name].shape) == 1:
                            self.TH[name] = grads_dict[name] + self.TH[name] * grads[0].view(-1)
                        else:
                            assert (False)
                        # print(torch.max(self.TH[name]), torch.max(grads_dict[name]), torch.max(grads[0].view( -1)))

                counter += 1

    def reset_weights(self):
        for elems in self.params:
            torch.nn.init.normal_(self.params[elems], 0, 0.1)

    def zero_grad(self):
        for a in self.grads:
            self.grads[a] = self.grads[a] * 0
            self.grads_scaled[a] = self.grads_scaled[a] * 0

    def reset_TH(self):
        for named, param in self.named_parameters():
            if "weight" in named or "bias" in named:
                self.TH[named] = self.TH[named] * 0

    def accumulate_gradients(self, target, prediction, hidden_state):
        # loss = F.cross_entropy(y.view(1, -1), target)
        loss = torch.sum((target.view(-1) - prediction.view(-1)) ** 2)
        grads_hidden = torch.autograd.grad(loss, [hidden_state], retain_graph=True)
        grads_parmams = torch.autograd.grad(loss, [self.prediction_parameters], retain_graph=True)

        with torch.no_grad():
            #
            # quit()
            # print("Grad = ", grads_parmams[0])
            if self.grads["prediction_parameters"] is not None:
                self.grads["prediction_parameters"] += grads_parmams[0]
            else:
                self.grads["prediction_parameters"] = grads_parmams[0]

            # print(error * hidden_state)
            for name, param in self.named_parameters():
                if name in self.TH:
                    # column_num = int(name.split(".")[1])
                    # print(grads_hidden[0].shape, self.TH[name].shape, self.grads[name].shape)
                    # quit()
                    if len(self.TH[name].shape) == 3:
                        self.grads[name] += grads_hidden[0].view(1, 1, -1) * self.TH[name]
                    elif len(self.TH[name].shape) == 2:
                        self.grads[name] += grads_hidden[0].view(1, -1) * self.TH[name]
                    elif len(self.TH[name].shape) == 1:
                        self.grads[name] += grads_hidden[0].view(-1) * self.TH[name]
                    else:
                        assert (False)
                    # print(name, torch.max(self.grads[name]), torch.max(self.TH[name]))

    def compute_and_accumulate_gradients(self, prediction, hidden_state):
        with torch.no_grad():

            if self.grads["prediction_parameters"] is not None:
                self.grads["prediction_parameters"] += (hidden_state)
                self.grads_scaled["prediction_parameters"] += (hidden_state) * prediction
            else:
                self.grads["prediction_parameters"] = (hidden_state)
                self.grads_scaled["prediction_parameters"] = (hidden_state) * prediction

            for name, param in self.named_parameters():
                if name in self.TH:
                    column_num = int(name.split(".")[1])
                    if self.grads[name] is not None:
                        self.grads[name] += self.prediction_parameters[column_num] * self.TH[name]

                        self.grads_scaled[name] += self.prediction_parameters[column_num] * self.TH[name] * prediction
                    else:
                        self.grads[name] = self.prediction_parameters[column_num] * self.TH[name]
                        self.grads_scaled[name] = self.prediction_parameters[column_num] * self.TH[name] * prediction

    def decay_gradient(self, decay_fac):
        with torch.no_grad():
            for a in self.grads:
                self.grads[a] = self.grads[a] * decay_fac


class SetOfColumns(nn.Module):
    def __init__(self, total_predictions, input_size, columns_per_prediction, width, device):
        super(SetOfColumns, self).__init__()
        self.list_of_CCNs = nn.ModuleList()
        self.total_predictions = total_predictions
        for a in range(total_predictions):
            self.list_of_CCNs.append(Eligibility_trace(input_size, columns_per_prediction, width, 0, device))


    def forward(self, x, hidden_states):
        outputs = []
        h_ts = []
        grads = []
        with Pool(self.total_predictions):

        for a in range(self.total_predictions):
            prediction, rnn_state, g = self.list_of_CCNs[a](x, hidden_states[a], grad=True, retain_graph=False, bptt=False)
            outputs.append(prediction)
            h_ts.append(rnn_state)
            grads.append(g)

        return outputs, h_ts, grads

    def update_TH(self, grads):
        for a in range(self.total_predictions):
            self.list_of_CCNs[a].update_TH(grads[a])

    def update_trace(self, prediction, hidden_state, lambda_return, gamma ):
        for a in range(self.total_predictions):
            self.list_of_CCNs[a].update_trace(prediction[a], hidden_state[a], lambda_return, gamma)

    def copy_gradient(self, prediction, target):
        for a in range(self.total_predictions):
            # print(prediction[a].shape, target[a].shape)
            # quit()
            self.list_of_CCNs[a].copy_gradient(prediction[a], target[a])


class Eligibility_trace(nn.Module):
    def __init__(self, input_dimension, total_columns, column_width=10, total_neigbours=1, device="cpu"):
        super(Eligibility_trace, self).__init__()
        self.fc1_weight = nn.Parameter(torch.zeros(input_dimension, column_width, total_columns))
        self.fc1_bias = nn.Parameter(torch.zeros(column_width, total_columns))
        # self.fc2_weight = nn.Parameter(torch.zeros(column_width, column_width, total_columns))
        # self.fc2_bias = nn.Parameter(torch.zeros(column_width, total_columns))
        # self.fc3_weight = nn.Parameter(torch.zeros(column_width, column_width, total_columns))
        # self.fc3_bias = nn.Parameter(torch.zeros(column_width, total_columns))

        self.forget_weight = nn.Parameter(torch.zeros(column_width, total_columns))
        self.input_weight = nn.Parameter(torch.zeros(column_width, total_columns))

        self.forget_bias = nn.Parameter(torch.zeros(total_columns))
        self.input_bias = nn.Parameter(torch.zeros(total_columns))

        for named, param in self.named_parameters():
            torch.nn.init.uniform_(param, -1 * np.sqrt(1 / column_width), np.sqrt(1 / column_width))

        self.no_of_hidden_nodes = total_columns
        self.t = 0

        self.TH = {}
        self.grads = {}
        self.grads_scaled = {}

        for named, param in self.named_parameters():
            self.TH[named] = torch.zeros_like(param.data).to(device)
            self.grads[named] = torch.zeros_like(param.data).to(device)
            self.grads_scaled[named] = torch.zeros_like(param.data).to(device)

        self.prediction_parameters = nn.Parameter(torch.zeros(total_columns, 1))
        self.grads["prediction_parameters"] = torch.zeros_like(self.prediction_parameters.data).to(device)
        self.grads_scaled["prediction_parameters"] = torch.zeros_like(self.prediction_parameters.data).to(device)

        # torch.nn.init.uniform_(self.prediction_parameters,-1*np.sqrt(1/total_columns), np.sqrt(1/total_columns) )

    def copy_gradient(self, prediction, target):
        error = prediction - target
        for named, param in self.named_parameters():
            param.grad = self.grads[named]*error

    # def copy_gradient_with_error(self, target, error):
    #     for named, param in self.named_parameters():
    #         param.grad = self.grads[named] * target - self.grads_scaled[named]

    def forward(self, x, hidden_state, grad=True, retain_graph=False, bptt=False):

        if not bptt:
            hidden_state = nn.Parameter(hidden_state)

        x = x.view(-1, 1, 1)
        # print(x.shape, hidden_state.shape, self.fc1_weight.shape)
        x_fc1 = torch.relu(torch.sum(x * self.fc1_weight, 0) + self.fc1_bias)
        i = torch.tanh(torch.sum(x_fc1 * self.input_weight, 0) + self.input_bias)
        f = torch.sigmoid(torch.sum(x_fc1 * self.forget_weight, 0) + self.forget_bias)

        h_t = hidden_state * f + i * (1 - f)

        sum = torch.sum(h_t)

        with torch.no_grad():
            grads = None
            if grad:
                grads = torch.autograd.grad(sum, [hidden_state] + list(self.parameters()), allow_unused=True,
                                            retain_graph=retain_graph)

        y = torch.sigmoid(torch.sum(self.prediction_parameters * h_t.view(-1, 1), dim=0))
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
                # print(name)
                if grads[counter] is not None:
                    if name in self.TH:
                        if len(self.TH[name].shape) == 3:
                            self.TH[name] = grads_dict[name] + self.TH[name] * grads[0].view(1, 1, -1)
                        elif len(self.TH[name].shape) == 2:
                            self.TH[name] = grads_dict[name] + self.TH[name] * grads[0].view(1, -1)
                        elif len(self.TH[name].shape) == 1:
                            self.TH[name] = grads_dict[name] + self.TH[name] * grads[0].view(-1)
                        else:
                            assert (False)
                        # print(torch.max(self.TH[name]), torch.max(grads_dict[name]), torch.max(grads[0].view( -1)))

                counter += 1


    # def reset_weights(self):
    #     for elems in self.params:
    #         torch.nn.init.normal_(self.params[elems], 0, 0.1)
    #
    # def zero_grad(self):
    #     for a in self.grads:
    #         self.grads[a] = self.grads[a] * 0
    #         self.grads_scaled[a] = self.grads_scaled[a] * 0
    #
    # def reset_TH(self):
    #     for named, param in self.named_parameters():
    #         if "weight" in named or "bias" in named:
    #             self.TH[named] = self.TH[named] * 0

    def update_trace(self, prediction, hidden_state, lambda_return, gamma):
        grads_hidden = torch.autograd.grad(prediction, [hidden_state], retain_graph=True)
        grads_parmams = torch.autograd.grad(prediction, [self.prediction_parameters], retain_graph=False)

        with torch.no_grad():

            if self.grads["prediction_parameters"] is not None:
                self.grads["prediction_parameters"] = self.grads["prediction_parameters"] * lambda_return * gamma + grads_parmams[0]
            else:
                self.grads["prediction_parameters"] = grads_parmams[0]

            for name, param in self.named_parameters():
                if name in self.TH:
                    if len(self.TH[name].shape) == 3:
                        self.grads[name] = self.grads[name] * lambda_return * gamma + grads_hidden[0].view(1, 1, -1) * \
                                           self.TH[name]
                    elif len(self.TH[name].shape) == 2:
                        self.grads[name] = self.grads[name] * lambda_return * gamma + grads_hidden[0].view(1, -1) * \
                                           self.TH[name]
                    elif len(self.TH[name].shape) == 1:
                        self.grads[name] = self.grads[name] * lambda_return * gamma + grads_hidden[0].view(-1) * \
                                           self.TH[name]
                    else:
                        assert (False)


class LSTM(nn.Module):
    def __init__(self, input_dimension, total_columns, column_width=10, total_neigbours=1, device="cpu"):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_dimension, total_columns, 1, bias=True)

        self.prediction_parameters = nn.Parameter(torch.zeros(total_columns, 10))

    def forward(self, x, hidden_state, context):
        x = x.view(1, 1, -1)
        output, (hidden_state, context) = self.lstm(x, (hidden_state, context))

        y = torch.sigmoid(torch.sum(self.prediction_parameters * output.view(-1, 1), dim=0))

        return y, hidden_state, context

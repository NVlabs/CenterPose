import torch
import torch.nn as nn
from torch.autograd import Variable
from lib.models.data_parallel import DataParallel


class ConvGRUCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvGRUCell, self).__init__()

        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_features = 4

        self.padding = int((kernel_size - 1) / 2)

        self.Wir = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whr = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wiz = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whz = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Win = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whn = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        self.br = None
        self.bz = None
        self.bin = None
        self.bhn = None

    def forward(self, x, h):
        rt = torch.sigmoid(self.Wir(x).to(x.device) + self.Whr(h).to(x.device) + self.br.to(x.device))  # reset
        zt = torch.sigmoid(self.Wiz(x).to(x.device) + self.Whz(h).to(x.device) + self.bz.to(x.device))  # update
        nt = torch.tanh(self.Win(x).to(x.device) + self.bin.to(x.device) + rt * (
                self.Whn(h).to(x.device) + self.bhn.to(x.device)))  # new

        ht = (1 - zt) * nt + zt * h
        return ht

    def init_hidden(self, batch_size, hidden, shape, device):
        if self.br is None:
            self.br = torch.zeros(1, hidden, shape[0], shape[1]).to(device)
            self.bz = torch.zeros(1, hidden, shape[0], shape[1]).to(device)
            self.bin = torch.zeros(1, hidden, shape[0], shape[1]).to(device)
            self.bhn = torch.zeros(1, hidden, shape[0], shape[1]).to(device)

        else:
            assert shape[0] == self.br.size()[2], 'Input Height Mismatched!'
            assert shape[1] == self.br.size()[3], 'Input Width Mismatched!'
        return torch.zeros(batch_size, hidden, shape[0], shape[1]).to(device)


class ConvGRU(nn.Module):
    # input_channels corresponds to the first input feature map
    # hidden state is a list of succeeding lstm layers.
    def __init__(self, input_channels, hidden_channels, kernel_size, step=1, effective_step=[1]):
        super(ConvGRU, self).__init__()
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self.step = step
        self.effective_step = effective_step
        self._all_layers = []
        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            cell = ConvGRUCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size)
            setattr(self, name, cell)
            self._all_layers.append(cell)

    def forward(self, input):
        internal_state = []
        outputs = []
        for step in range(self.step):
            x = input
            for i in range(self.num_layers):
                # all cells are initialized in the first step
                name = 'cell{}'.format(i)
                if step == 0:
                    bsize, _, height, width = x.size()
                    h = getattr(self, name).init_hidden(batch_size=bsize, hidden=self.hidden_channels[i],
                                                        shape=(height, width), device=x.device)
                    internal_state.append(h)

                # do forward
                h = internal_state[i]
                x = getattr(self, name)(x, h)
                internal_state[i] = x
            # only record effective steps
            if step in self.effective_step:
                outputs.append(x)

        return outputs, x


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    convgru = ConvGRU(input_channels=64, hidden_channels=[64, ], kernel_size=3, step=4,
                      effective_step=[0, 1, 2, 3])

    if torch.cuda.device_count() > 1:
        model = DataParallel(convgru, device_ids=[0, 1]).to(device)
    else:
        model = convgru.to(device)

    loss_fn = torch.nn.MSELoss()

    input = torch.randn(1, 64, 64, 64).to(device)
    target = torch.randn(1, 64, 64, 64).double().to(device)

    output = model(input)
    output = output[0][0].double()
    res = torch.autograd.gradcheck(loss_fn, (output, target), eps=1e-6, raise_exception=True)
    print(res)

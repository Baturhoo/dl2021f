"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.dqn=nn.Sequential(
            nn.Linear(4, 64,bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(64,64,bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(64,1,bias=True)
        )

        self._create_weights()

    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.dqn(x)
        return x

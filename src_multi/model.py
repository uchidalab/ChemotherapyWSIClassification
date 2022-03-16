import os
import sys
from torchvision import models
import torch
import torch.nn as nn

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.model import build_model

in_features = {
    "resnet50": 2048
}


class MultiscaleNet(nn.Module):
    def __init__(
        self,
        base_fe: str,
        num_class: int,
        pretrained: bool = True,
        weight_path_0: str = None,
        weight_path_1: str = None,
        weight_path_2: str = None,
        device=torch.device('cuda')
    ):
        super().__init__()
        self.base_fe = base_fe
        self.num_class = num_class
        self.pretrained = pretrained
        self.weight_path_0 = weight_path_0
        self.weight_path_1 = weight_path_1
        self.weight_path_2 = weight_path_2
        self.device = device
        self.fe0 = self.get_fe(self.weight_path_0)
        self.fe1 = self.get_fe(self.weight_path_1)
        self.fe2 = self.get_fe(self.weight_path_2)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Sequential(
            nn.Linear(in_features=in_features[self.base_fe] * 3,
                      out_features=in_features[self.base_fe], bias=True),
            nn.ReLU(),
            nn.Linear(in_features=in_features[self.base_fe],
                      out_features=self.num_class, bias=True)
        )

    def load_weight(self, model, weight_path):
        pretrained_dict = torch.load(weight_path, map_location=self.device)
        try:
            model.load_state_dict(pretrained_dict, strict=False)
            print(f"load {weight_path}")
        except:
            del pretrained_dict['fc.weight']
            del pretrained_dict['fc.bias']
            model.load_state_dict(pretrained_dict, strict=False)
        return model

    def get_fe(self, weight_path):
        if self.base_fe == "resnet50":
            base_model = models.resnet50(pretrained=self.pretrained)
            if (self.pretrained and weight_path is not None):
                base_model = self.load_weight(base_model, weight_path)
            fe = nn.Sequential(*list(base_model.children()))[:-2]
        else:
            sys.exit(f"{self.base_fe} is invalid")
        return fe

    def forward(self, x0, x1, x2):
        x0 = self.fe0(x0)
        x1 = self.fe1(x1)
        x2 = self.fe2(x2)
        x = torch.cat((x0, x1, x2), dim=1)  # dimは要確認! (ch方向にconcat)
        x = self.avg_pool(x)
        x = x.view(-1, in_features[self.base_fe] * 3)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    device = 'cpu'

    MSN = MultiscaleNet(
        base_fe="resnet50",
        num_class=3,
        pretrained=True,
        weight_path=None,
        device=device
    )

    x0 = torch.rand(32, 3, 256, 256).to(device)
    x1 = torch.rand(32, 3, 256, 256).to(device)
    x2 = torch.rand(32, 3, 256, 256).to(device)

    output = MSN(x0, x1, x2)
    print(output)

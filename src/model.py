from torchvision import models
import torch.nn as nn


def build_model(model_name, num_classes, pretrained=True):
    if model_name == "mobilenet_v2":
        model = models.mobilenet_v2(pretrained=pretrained)
        model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes, bias=True)
    elif model_name == "inception_v3":
        model = models.inception_v3(pretrained=pretrained)
        model.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)
    elif model_name == "resnet18":
        model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)
    elif model_name == "resnet50":
        model = models.resnet50(pretrained=pretrained)
        model.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)
    elif model_name == "resnet101":
        model = models.resnet101(pretrained=pretrained)
        model.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)
    elif model_name == "wide_resnet101":
        model = models.wide_resnet101_2(pretrained=pretrained)
        model.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)
    elif model_name == "alexnet":
        model = models.alexnet(pretrained=pretrained)
        model.classifier[6] = nn.Linear(in_features=4096, out_features=num_classes, bias=True)
    else:
        raise Exception("Unexpected model_name: {}".format(model_name))
    return model

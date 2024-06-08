import torchvision
import torch

def get_model(model_type, cuda = False, classes = None):
    
    if model_type == "vgg16":
        model = torchvision.models.vgg16(weights = "DEFAULT")
        
        if classes is not None:
            biases = model._modules["classifier"][6].bias.clone().detach()
            mask = torch.ones(biases.shape[0]).bool()
            mask[classes] = False
            biases[mask] = -float("Inf")
            model._modules["classifier"][6].bias = torch.nn.parameter.Parameter(biases)
    else:
        raise ValueError(f"model {model_type} not supported")

    model = model.eval()
    
    if cuda:
        model.cuda()
    
    return model


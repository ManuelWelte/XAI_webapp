import torchvision

def get_model(model_type, cuda = False):

    if model_type == "vgg16":
        model = torchvision.models.vgg16(weights = "DEFAULT")
    else:
        raise ValueError(f"model {model_type} not supported")

    model = model.eval()
    
    if cuda:
        model.cuda()
    
    return model


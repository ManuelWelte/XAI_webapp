import torchvision
import torch
import zennit

class ConceptExplainer:
    
    def __init__(self, model, mean, std, cuda = False, composite = zennit.composites.EpsilonGammaBox, composite_args = dict(epsilon = 1e-06, gamma = 0.25)):
        
        self.model = model
        self.cuda = cuda
        self.composite = composite
        self.composite_args = composite_args
        self.mean = mean
        self.std = std

    def box_contraints(self):
        
        lb = (0 - self.mean) / self.std
        hb = (0 - self.mean) / self.std

        return lb.reshape((1, 3, 1, 1)), hb.reshape((1, 3, 1, 1))
         
    def canonizer(self): 

        canons = []
        
        if isinstance(self.model, torchvision.models.vgg16):
            canons += [zennit.canonizers.VGGCanonizer()]
        
        return canons

    def explain(self, x, target_logits = "prediction"):
        
        if self.cuda:
            x.cuda()

        canons = self.canonizer()
        lb, hb = self.box_contraints()

        comp = self.composite(canonizers = canons, low = lb, high = hb, zero_params = "bias", **self.composite_args)

        x.requires_grad = True
        
        if hasattr(x, "grad") and x.grad is not None:
            x.grad.zero_()

        comp.register(self.model)

        out = self.model(x)
        pred = out.argmax(axis = 1)

        if target_logits == "prediction":
            target_logits = pred

        rel_init = torch.eye(1000)[target_logits]
        
        if self.cuda:
            rel_init = rel_init.cuda() 
                               
        out.backward(gradient = rel_init)

        comp.remove()

        return out, pred, x.grad        


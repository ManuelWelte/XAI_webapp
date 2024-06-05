import torchvision
import torch
import zennit

class ConceptExplainer:
    
    def __init__(self, model, mean, std, composite = zennit.composites.EpsilonGammaBox, composite_args = dict(epsilon = 1e-06, gamma = 0.25)):
        
        self.model = model
        self.composite = composite
        self.composite_args = composite_args
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def box_contraints(self):
        
        lb = (0 - self.mean) / self.std
        hb = (1 - self.mean) / self.std
        
        return lb.reshape((1, 3, 1, 1)), hb.reshape((1, 3, 1, 1))
         
    def canonizer(self): 
        canons = []
        
        if isinstance(self.model, torchvision.models.VGG):
            canons += [zennit.canonizers.SequentialMergeBatchNorm()]
        
        return canons

    def explain(self, x, target_logits = "prediction"):
        
        device = torch.device("cuda:0" if x.get_device() != -1 else "cpu")
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

        rel_init = torch.eye(1000)

        out.backward(gradient = rel_init[target_logits.cpu()].to(device))            
        comp.remove()

        return out, pred, x.grad        


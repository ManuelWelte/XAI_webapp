import torchvision
import torch
import zennit

# For LRP, the grad of pre-activations z is identical to that of activations a 
def store_hook(module, inp, outp):
    module.output = outp
    module.output.retain_grad()

class ConceptExplainer:
    
    def __init__(self, model, mean, std, composite = zennit.composites.EpsilonGammaBox, composite_args = dict(epsilon = 1e-06, gamma = 1000)):
        
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
    
    def register_intermediate_hook(self, layer):
        hook = None

        if layer is not None:
            hook = layer.register_forward_hook(store_hook)

        return hook

    def explain(self, x, target_logits = "prediction", layer = None):
        
        device = torch.device("cuda:0" if x.get_device() != -1 else "cpu")
        canons = self.canonizer()
        lb, hb = self.box_contraints()

        comp = self.composite(canonizers = canons, low = lb, high = hb, zero_params = "bias", **self.composite_args)

        x.requires_grad = True
        
        if hasattr(x, "grad") and x.grad is not None:
            x.grad.zero_()

        comp.register(self.model)
        hook = self.register_intermediate_hook(layer)

        out = self.model(x)
        pred = out.argmax(axis = 1)

        if target_logits == "prediction":
            target_logits = pred

        rel_init = torch.eye(1000)
        
      
        out.backward(gradient = rel_init[target_logits.cpu()].to(device))            

        if hook is not None:
            hook.remove()
            intermediate_rel = layer.output.grad 
        
        comp.remove()

        return out, pred, x.grad        


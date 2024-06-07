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
        
    def explain_channel(self, first_pass_info, channel, retain_graph = False):
        
        x = first_pass_info["input_tensor"]
        intm_grad = first_pass_info["intermediate_grad"]
        intm_out = first_pass_info["intermediate_outp"]

        x.grad.zero_()

        with torch.no_grad():
            mask = ~torch.eye(intm_out.shape[1]).bool()[channel]
            grad_start = intm_grad.clone().detach()
            grad_start[mask[None,:]] = 0
        
        intm_out.backward(gradient = grad_start, retain_graph = retain_graph)
        
        if not retain_graph:
            self.active_comp.remove()

        return x.grad

    def explain(self, x, target_logits = "prediction", layer = None):
        
        device = torch.device("cuda:0" if x.get_device() != -1 else "cpu")
        canons = self.canonizer()
        lb, hb = self.box_contraints()

        self.active_comp = self.composite(canonizers = canons, low = lb, high = hb, zero_params = "bias", **self.composite_args)
        x.requires_grad = True
        
        if hasattr(x, "grad") and x.grad is not None:
            x.grad.zero_()

        self.active_comp.register(self.model)
        hook = self.register_intermediate_hook(layer)

        out = self.model(x)
        pred = out.argmax(axis = 1)

        if target_logits == "prediction":
            target_logits = pred

        rel_init = torch.eye(1000)

        out.backward(gradient = rel_init[target_logits.cpu()].to(device), retain_graph = hook is not None)            
        
        first_pass_info = {}

        if hook is not None:
            hook.remove()
            intermediate_rel = layer.output.grad

            if isinstance(layer, torch.nn.Conv2d):
                with torch.no_grad():
                    intermediate_rel = intermediate_rel.sum(axis = (2,3))
        
            first_pass_info["intermediate_rel"] = intermediate_rel
            first_pass_info["intermediate_grad"] = layer.output.grad.clone().detach()
            first_pass_info["intermediate_outp"] = layer.output
            first_pass_info["input_tensor"] = x
        
        else:
            self.active_comp.remove()

        return out, x.grad, first_pass_info        


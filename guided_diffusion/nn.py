
import math
import torch as th
import torch.nn as nn

# PyTorch 1.7 has SiLU,but we support PyTorch 1.5.

class SiLU(nn.Module):
    def forward(self,x):
        return x*th.sigmoid(x)

class GroupNorm32(nn.GroupNorm):
    def forward(self,x):
        return super().forward(x.float()).type(x.dtype)

def conv_nd(dims,*args,**kwargs):
    if dims == 1:
        return nn.Conv1d(*args,**kwargs)
    elif dims == 2:
        return nn.Conv2d(*args,**kwargs)
    elif dims == 3:
        return nn.Conv3d(*args,**kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

def linear(*args,**kwargs):
    return nn.Linear(*args,**kwargs)

def avg_pool_nd(dims,*args,**kwargs):
    if dims == 1:
        return nn.AvgPool1d(*args,**kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args,**kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args,**kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

def update_ema(target_params,source_params,rate=0.99):
    for targ,src in zip(target_params,source_params):
        targ.detach().mul_(rate).add_(src,alpha=1-rate)

def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module

def scale_module(module,scale):
    for p in module.parameters():
        p.detach().mul_(scale)
    return module

def mean_flat(tensor):
    return tensor.mean(dim=list(range(1,len(tensor.shape))))

def normalization(channels):
    return GroupNorm32(32,channels)

def timestep_embedding(timesteps,dim,max_period=10000):
    half=dim//2
    freqs=th.exp(-math.log(max_period)*th.arange(start=0,end=half,dtype=th.float32)/half)).to(th.device("cuda:0"))
    args=timesteps[:,None].float()*freqs[None]
    embedding=th.cat([th.cos(args),th.sin(args)],dim=-1)
    if dim % 2:
        embedding=th.cat([embedding,th.zeros_like(embedding[:,:1])],dim=-1)
    return embedding

    """
    Evaluate a function without caching intermediate activations,allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False,disable gradient checkpointing.
    """

def checkpoint(func,inputs,params,flag):
    if flag:
        args=tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func,len(inputs),*args)
    else:
        return func(*inputs)

class CheckpointFunction(th.autograd.Function):
    @staticmethod
    @th.cuda.amp.custom_fwd
    def forward(ctx,run_function,length,*args):
        ctx.run_function=run_function
        ctx.input_length=length
        ctx.save_for_backward(*args)
        with th.no_grad():
            output_tensors=ctx.run_function(*args[:length])
        return output_tensors
    @staticmethod
    @th.cuda.amp.custom_bwd
    def backward(ctx,*output_grads):
        args=list(ctx.saved_tensors)
        input_indices=[i for (i,x) in enumerate(args) if x.requires_grad]
        if not input_indices:
            return (None,None) + tuple(None for _ in args)
        with th.enable_grad():
            for i in input_indices:
                if i < ctx.input_length:
                    args[i]=args[i].detach().requires_grad_()
                    args[i]=args[i].view_as(args[i])
            output_tensors=ctx.run_function(*args[:ctx.input_length])
        if isinstance(output_tensors,th.tensor):
            output_tensors=[output_tensors]
        out_and_grads=[(o,g) for (o,g) in zip(output_tensors,output_grads) if o.requires_grad]
        if not out_and_grads:
            return (None,None) + tuple(None for _ in args)
        computed_grads=th.autograd.grad(
            [o for (o,g) in out_and_grads],
            [args[i] for i in input_indices],
            [g for (o,g) in out_and_grads])
        input_grads=[None for _ in args]
        for (i,g) in zip(input_indices,computed_grads):
            input_grads[i]=g
        return (None,None) + tuple(input_grads)

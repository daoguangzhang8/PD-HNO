"""Four-rank CPU/Gloo native DDP vs global BS32, using the production PDE loss.

Small FiLM network (not the full FNO); exercises accumulation, high-order
coordinate derivatives, every parameter gradient and two Adam updates.
"""
import os
os.environ.setdefault('MKL_THREADING_LAYER','GNU')
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp
from contextlib import nullcontext
from model.PI_DeepOnet import Pi_DeepONet


class SmallPhysics(nn.Module):
    _compute_pde_residual=Pi_DeepONet._compute_pde_residual
    compute_loss=Pi_DeepONet.compute_loss

    def __init__(self):
        super().__init__()
        self.args=SimpleNamespace(dh=20.,pml=True,pml_total=20,pml_crop=15,
            pde_generation_dh=10.,boundary_type='free_surface',pde_attenuation_alpha=0.,
            film_smooth_weight=0.,continuous_pde_weight=0.)
        self.trunk=nn.Linear(2,8)
        self.film=nn.Linear(2,16)
        self.output=nn.Linear(8,2)
        self.loss_function=nn.MSELoss()

    def forward(self,v,y,bg,freq_batch=None):
        condition=torch.stack((v.mean((1,2,3)),bg.mean((1,2,3))),-1)
        scale,bias=self.film(condition).chunk(2,-1)
        h=torch.sin(self.trunk(y/1000))*(1+scale[:,None])+bias[:,None]
        return self.output(torch.sin(h))


def worker(rank,size,init_path):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+init_path,rank=rank,world_size=size)
    try:
        torch.manual_seed(131)
        m=SmallPhysics();ddp=nn.parallel.DistributedDataParallel(m)
        reference=SmallPhysics();reference.load_state_dict(m.state_dict())
        opt=torch.optim.Adam(m.parameters(),lr=2e-4,weight_decay=1e-4)
        refopt=torch.optim.Adam(reference.parameters(),lr=2e-4,weight_decay=1e-4)
        rng=torch.Generator().manual_seed(77)
        v=1.5+torch.rand(32,1,145,150,generator=rng)
        bg=torch.randn(32,2,145,150,generator=rng)
        labels=torch.randn(32,2,145,150,generator=rng)
        freq=torch.linspace(3,25,32)
        sl=slice(rank*8,(rank+1)*8)
        for step in range(4):
            points=torch.tensor([[20.,40.],[2860.,2960.],[1000.,1200.]])+step*20
            y=points[None].expand(8,-1,-1).clone().requires_grad_(True)
            all_y=points[None].expand(32,-1,-1).clone().requires_grad_(True)
            with (ddp.no_sync() if step%2==0 else nullcontext()):
                pred=ddp(v[sl],y,bg[sl],freq_batch=freq[sl])
                loss=m.compute_loss(pred,v[sl],y,bg[sl],labels[sl],y,1.,1.,0.,freq_batch=freq[sl])[0]
                (loss/2).backward()
            refpred=reference(v,all_y,bg,freq_batch=freq)
            refloss=reference.compute_loss(refpred,v,all_y,bg,labels,all_y,1.,1.,0.,freq_batch=freq)[0]
            (refloss/2).backward()
            average_loss=loss.detach().clone();dist.all_reduce(average_loss);average_loss/=size
            torch.testing.assert_close(average_loss,refloss.detach(),atol=1e-6,rtol=1e-5)
            if step%2:
                for p,q in zip(m.parameters(),reference.parameters()):
                    torch.testing.assert_close(p.grad,q.grad,atol=1e-6,rtol=1e-4)
                opt.step();refopt.step()
                for p,q in zip(m.parameters(),reference.parameters()):
                    torch.testing.assert_close(p,q,atol=1e-6,rtol=1e-5)
                opt.zero_grad(set_to_none=True);refopt.zero_grad(set_to_none=True)
        if rank==0:print('4 ranks x BS8 == BS32: losses, all gradients, two Adam updates PASS',flush=True)
    finally:dist.destroy_process_group()


class FourRankTests(unittest.TestCase):
    @unittest.skipUnless(os.getenv('RUN_DISTRIBUTED_TESTS')=='1','opt-in local process communication')
    def test_native_four_rank_updates(self):
        with tempfile.TemporaryDirectory() as root:
            mp.spawn(worker,args=(4,str(Path(root)/'init')),nprocs=4,join=True)


if __name__=='__main__':unittest.main()

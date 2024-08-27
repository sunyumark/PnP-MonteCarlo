import torch, wandb
from tqdm import tqdm
from abc import ABC
from typing import Optional, Iterable, Tuple
from pmc.utils.local_logger import LocalLogger

class PMCInference(ABC):

    def __init__(
        self, 
        cfg,
        pmc: object,
        callbacks: Optional[Iterable[object]]=None
    ):
        super().__init__()
        self.cfg = cfg # for logging
        self.pmc = pmc
        self.callbacks = callbacks
        if self.cfg.inference.is_wandb_logger:
            self.logger = wandb.init(
                                    reinit = True,
                                    config = dict(cfg),
                                    **dict(self.cfg.logger),
                                )
        else:
            self.logger = LocalLogger(self.cfg)
        

    def init_x0(self, x, initialization):
        # intialization
        if initialization == 'point1':
            x0 = torch.zeros_like(x) + 0.1
        elif initialization == 'zero':
            x0 = torch.zeros_like(x)
        elif initialization == 'rand':
            x0 = 2*torch.rand_like(x)-1
        elif initialization == 'rand01':
            x0 = torch.rand_like(x)
        elif initialization == 'randn':
            x0 = torch.randn_like(x)
        else:
            raise NotImplementedError('Unknown initialization')
        return x0

    def sample(
        self,
        n_samples: int,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        tmax: int = 1000,
        initialization: str ='zero',
    )-> Tuple[torch.Tensor, torch.Tensor]:
        
        '''
        Iterative updation in autonomous diffusion models
        Args:
            n_samples: number of samples to drawn
            batch: (x, y)
            batch_idx: the image index for recording
            tmax: the maximum number of iterations
            initializations: the initialization of the image
        Out:
            x: the samples drawn from posterior [N_samples,C,H,W]
        '''
        
        x, y = batch
        # extend to [N_samples, C, H, W]
        x = torch.tile(x, [n_samples, 1, 1, 1])
        if type(y) is tuple:
            y_list = []
            for i in range(len(y)):
                y_ele = torch.tile(y[i], [n_samples, 1, 1, 1])
                y_list.append(y_ele)
            y = tuple(y_list)
        else:
            y = torch.tile(y, [n_samples, 1, 1, 1])
        # initialization
        xcurr = self.init_x0(x, initialization)
        xcurr_drift = xcurr.detach().clone()

        #-----> on_sample_start <-----#
        if self.callbacks is not None:
            for callback in self.callbacks:
                callback.on_sample_start(self, batch, batch_idx)

        pbar = tqdm(range(tmax), desc='iteration', leave=False)
        for t in pbar:
            
            #-----> on_iteration_start <-----#
            if self.callbacks is not None:
                for callback in self.callbacks:
                    callback.on_iteration_start(self, batch, batch_idx, t)
            
            
            #-----> timer hook <-----#
            if self.callbacks is not None:
                for callback in self.callbacks:
                    if callback.__class__.__name__ == "LocalTimerCallbackModule":
                        callback.on_iteration_start(self, batch, batch_idx, t)
            
            # compute diffusion models
            iteration_outputs = self.pmc(xcurr, y, t, tmax)
            xnext, xnext_drift, _, drift, _, _, _ = iteration_outputs

            #-----> timer hook <-----#
            if self.callbacks is not None:
                for callback in self.callbacks:
                    if callback.__class__.__name__ == "LocalTimerCallbackModule":
                        callback.on_iteration_end(self, iteration_outputs, batch, batch_idx, t)
            
            #-----> on_iteration_end <-----#
            if self.callbacks is not None:
                for callback in self.callbacks:
                    callback.on_iteration_end(self, iteration_outputs, batch, batch_idx, t)
                    
            # update
            xcurr = xnext
            xcurr_drift = xnext_drift

        #-----> on_sample_end <-----#
        if self.callbacks is not None:
            for callback in self.callbacks:
                callback.on_sample_end(self, (xcurr, xcurr_drift), batch, batch_idx)

        return xcurr, xcurr_drift


    def __call__(
        self, 
        dataloader,
        start_batch_idx=0,
        max_num_batches=1,
        n_samples=1,
    ):
        #-----> on_inference_start <-----#
        if self.callbacks is not None:
            for callback in self.callbacks:
                callback.on_inference_start(self)

        for batch_idx, x in tqdm(enumerate(dataloader), total=max_num_batches, desc='batch'):
            # check if terminate
            if batch_idx < start_batch_idx:
                continue
            if batch_idx >= max_num_batches:
                break

            # move image to device
            if self.cfg.accelerator == 'gpu':
                x = x.to('cuda')
            # generate measurements
            y = self.pmc.forward_model(x)
            # get batch (x, y)
            batch = (x, y)

            #-----> on_batch_start <-----#
            if self.callbacks is not None:
                for callback in self.callbacks:
                    callback.on_batch_start(self, batch, batch_idx)

            # run algorithm            
            xrecons, xdrift_recons = self.sample(
                                        n_samples,
                                        batch, 
                                        batch_idx,
                                        **dict(self.cfg.inference.sample_args)
                                    )

            # compute mean & std
            xrecon_mean = xrecons.mean(dim=0, keepdim=True)
            xrecon_std  = xrecons.std(dim=0, keepdim=True)
            xdrift_recon_mean = xdrift_recons.mean(dim=0, keepdim=True)
            xdrift_recon_std  = xdrift_recons.std(dim=0, keepdim=True)
                
            #-----> on_batch_end <-----#
            if self.callbacks is not None:
                for callback in self.callbacks:
                    callback.on_batch_end(self, (xrecons, xdrift_recons), (xrecon_mean, xdrift_recon_mean), (xrecon_std, xdrift_recon_std), batch, batch_idx)
            
        #-----> on_inference_end <-----#
        if self.callbacks is not None:
            for callback in self.callbacks:
                callback.on_inference_end(self)
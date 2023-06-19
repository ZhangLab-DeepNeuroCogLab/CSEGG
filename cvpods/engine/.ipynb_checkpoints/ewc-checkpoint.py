import copy
from copy import deepcopy

import tqdm
from tqdm import tqdm 

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.utils.data


def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)


class EWC(object):
    
    def __init__(self, cfg, model: nn.Module, max_iter, dataloader, task_number, batch_subdivisions = 1, importance=250):
        
        
        self.cfg = cfg
        self.model = model
        self.max_iter = 769
        self.importance = importance 
        self.task_number = task_number 
        self.batch_subdivisions = batch_subdivisions
        
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self.data_loader = dataloader 
        self._data_loader_iter = iter(self.data_loader)
        
        self._precision_matrices = self._diag_fisher()
        
        # print(self.model.device())
        

        for n, p in deepcopy(self.params).items():
            self._means[n] = variable(p.data)               #Implement to make it on the same device 
        

    def _diag_fisher(self,):
        
        """ This function calculates the fischer matrices for the model after training of each task """
        
        print("Calculating Ficher matrix")
        
        precision_matrices = {}                             #This is the fischer matrix 
        
        #initialiing the fischer matrix 
        #TODO :- Add the functionality for it being on the same device
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = variable(p.data)  
        
        # print(precision_matrices)
        
        # if self.task_number == 1 :
        #     return precision_matrices 
        
        loss_dict_summary = {}
        
        self.model.train()

        cfg = self.cfg
        max_iter = self.max_iter
        
        for data, iteration in tqdm(zip(self.data_loader, range(0, max_iter))):
            
            # print(iteration)
            
            self.model.zero_grad()
            #rpn_output, head_output = self.model.module.cal_logit(data)
            #label = head_output.max(1)[1].view(-1)
            
            data_cp = copy.deepcopy(data)
            del data
            
            loss_dict = self.model(data_cp)
            
            for metrics_name, metrics_value in loss_dict.items():
                # Actually, some metrics are not loss, such as
                # top1_acc, top5_acc in classification, filter them out
                if metrics_value.requires_grad:
                    loss_dict[metrics_name] = metrics_value / self.batch_subdivisions
                    #print(metrics_name)

            losses = sum([
                metrics_value for metrics_value in loss_dict.values()
                if metrics_value.requires_grad
            ])
            
            #print(output)
            
            # self.model.eval()
            
            losses.backward()

            for n, p in self.model.named_parameters():
                
                # print(n)
                if p.grad is None:
                    # print(n)
                    # print(p)
                    continue
                    
                # print(n)    
                if p.requires_grad:
                    # print(n)
                    precision_matrices[n].data += p.grad.data ** 2 / max_iter
            
       
        precision_matrices = {n: p for n, p in precision_matrices.items()}
        
        # print(precision_matrices)
        
        return precision_matrices
    
    
    def penalty(self, model: nn.Module):
        
        """ This function calculates the "EWC" loss for the model """
        
        loss = 0
        
        # if self.task_number == 1 :
        #     return loss 
        
        for n, p in model.named_parameters():
            if p.requires_grad:
                _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
                loss += _loss.sum() 
        
        return loss 
        
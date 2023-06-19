import copy
from copy import deepcopy

import tqdm
from tqdm import tqdm 

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.utils.data

from cvpods.modeling.meta_arch.detr import SetCriterion 

def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)


class iCarl(object):
    
    def __init__(self, cfg, old_model: nn.Module, max_iter, dataloader, task_number, batch_subdivisions = 1, alpha = 0.3, temp = 2):
        
        
        self.cfg = cfg
        self.old_model = old_model
        self.task_number = task_number 
        self.batch_subdivisions = batch_subdivisions
        
        self.data_loader = dataloader 
        self._data_loader_iter = iter(self.data_loader)
        
        self.alpha = alpha
        
        self.T = temp 
        
        self.factor = 10
        
        # self.loss_fn_labels = F.CrossEntropyLoss()
        
        self.aux_loss = not cfg.MODEL.DETR.NO_AUX_LOSS
        self.num_classes = cfg.MODEL.DETR.NUM_CLASSES
        
        self.weight_dict = {
            "loss_ce": cfg.MODEL.DETR.CLASS_LOSS_COEFF,
            "loss_bbox": cfg.MODEL.DETR.BBOX_LOSS_COEFF,
            "loss_giou": cfg.MODEL.DETR.GIOU_LOSS_COEFF,
        }
        
        
        if self.aux_loss:
            self.aux_weight_dict = {}
            for i in range(cfg.MODEL.DETR.TRANSFORMER.NUM_DEC_LAYERS - 1):
                self.aux_weight_dict.update({k + f"_{i}": v for k, v in self.weight_dict.items()})
            self.weight_dict.update(self.aux_weight_dict)
        
        losses = ["labels", "boxes", "cardinality"]

        # matcher = HungarianMatcher(
        #     cost_class=cfg.MODEL.DETR.COST_CLASS,
        #     cost_bbox=cfg.MODEL.DETR.COST_BBOX,
        #     cost_giou=cfg.MODEL.DETR.COST_GIOU,
        # )

        # self.criterion = SetCriterion(
        #     self.num_classes,
        #     matcher=matcher,
        #     weight_dict=self.weight_dict,
        #     eos_coef=cfg.MODEL.DETR.EOS_COEFF,
        #     losses=losses,
        #     task_number = self.task_number
        # )
        
        # print(self.model.device())
        

        # for n, p in deepcopy(self.params).items():
        #     self._means[n] = variable(p.data)               #Implement to make it on the same device 
        
        
    def loss_kd(self, outputs, data_cp):
            
        # curr_model.train()
        # curr_model.continual_eval = True 
        self.old_model.eval()
        
        # outputs = curr_model(data_cp)
        with torch.no_grad():
            outputs_k = self.old_model(data_cp)
            
#         print(outputs_k.keys())
#         print(outputs.keys()
        
#         print(outputs_k['pred_logits'].shape)
#         print(outputs['pred_logits'].shape)
            
        loss_kd = self.output_matching_loss(outputs, outputs_k)
        
        # curr_model.continual_eval = False
            
        return loss_kd*(self.alpha)
        
        
    def output_matching_loss(self, outputs, outputs_k):
        
        """ This function calculates the knowledge distillation function based on cross entropy loss """
        
        T = self.T
        
        logits = outputs['pred_logits']
        target = outputs_k['pred_logits']
        
        logits_bbox = outputs['pred_boxes']
        target_bbox = outputs_k['pred_boxes']
        
        num_boxes = target_bbox.shape[1]
        
        dist_loss_bbox = F.l1_loss(logits_bbox, target_bbox, reduction="none")
        
        dist_loss_bbox = dist_loss_bbox.sum()/num_boxes
        
        dist_loss_labels = nn.KLDivLoss()(F.log_softmax(logits/T, dim=1), F.softmax(target/T, dim=1)) * (T*T)
        
        dist_loss_new = (dist_loss_labels + dist_loss_bbox)*self.factor
        
        return dist_loss_new
        
    def convert_outputs_to_gt(self):
        
        """ This function is used to convert the output into gt format to calculate the losses with respect to the current model """
        
        pass
            
            
        

     
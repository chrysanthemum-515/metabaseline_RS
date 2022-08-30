import torch
import torch.nn as nn
import torch.nn.functional as F

import models
import utils
from .models import register


@register('meta-baseline')
class MetaBaseline(nn.Module):

    def __init__(self, encoder, encoder_args={}, method='cos',
                 temp=10., temp_learnable=True):
        super().__init__()
        self.encoder = models.make(encoder, **encoder_args)
        self.method = method

        if temp_learnable:
            self.temp = nn.Parameter(torch.tensor(temp))
        else:
            self.temp = temp

    def forward(self, x_shot, x_query):
        shot_shape = x_shot.shape[:-3]  # (4,5,5) (tasks,way,shot)
        query_shape = x_query.shape[:-3]    # (4,75) (tasks,way*shot)
        img_shape = x_shot.shape[-3:]   # (3,80,80)(c,h,w)

        x_shot = x_shot.view(-1, *img_shape)     # (100,3,80,80)
        x_query = x_query.view(-1, *img_shape)  # (300,3,80,80)
        x_tot = self.encoder(torch.cat([x_shot, x_query], dim=0))
        x_shot, x_query = x_tot[:len(x_shot)], x_tot[-len(x_query):]
        x_shot = x_shot.view(*shot_shape, -1)   # (4,5,5,512)
        x_query = x_query.view(*query_shape, -1)    # (4,75,512)

        if self.method == 'cos':
            x_shot = x_shot.mean(dim=-2)    # mean basic prototype (4,5,512)
            x_shot = F.normalize(x_shot, dim=-1)
            x_query = F.normalize(x_query, dim=-1)
            metric = 'dot'
            #prototype rectification
            logits = F.softmax(utils.compute_logits(
                x_query, x_shot, metric=metric, temp=self.temp).permute(0,2,1),dim=-2)
            perseudo_proto = torch.bmm(logits,x_query) 
            x_shot = F.normalize((x_shot + perseudo_proto),dim=-1)
                       
        elif self.method == 'sqr':
            x_shot = x_shot.mean(dim=-2)
            metric = 'sqr'

        logits = utils.compute_logits(
                x_query, x_shot, metric=metric, temp=self.temp)     # (4,75,5)(tasks,q_way*q_shot,q_way)
        return logits


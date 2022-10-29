import torch
import torch.nn as nn
import torch.nn.functional as F


import models
import utils
import math
from .models import register

@register('meta-attn-baseline')
class MetaAttnBaseline(nn.Module):

    def __init__(self, encoder, encoder_args={}, method='cos',
                 temp=10., temp_learnable=True,
                 # attention params
                 model_dim=512, patchsz=4, num_heads=8, channels=3):
        super().__init__()
        self.encoder = models.make(encoder, **encoder_args)
        self.method = method
        
        self.patchsz = patchsz
        self.kernel = nn.Parameter(torch.randn(model_dim,channels,patchsz,patchsz))
        self.attention = MultiHeadAttention(model_dim,model_dim,model_dim,
                                            model_dim,num_heads,0.5,
                                            patchsz,img_height=16)
        self.get_PN_features = Get_PN_features(split_thresh=0.02)
        if temp_learnable:
            self.temp = nn.Parameter(torch.tensor(temp))
        else:
            self.temp = temp

    def _img2emd(self,img,kernel,stride):
        conv_out = F.conv2d(img,kernel,stride=stride)
        bs, oc, oh, ow = conv_out.shape
        embedding = torch.reshape(conv_out,shape=(bs,oc,oh*ow)).transpose(-1,-2)
        return embedding
    
    def forward(self, x_shot, x_query):
        shot_shape = x_shot.shape[:-3]      # (tasks,way,support_shot)
        query_shape = x_query.shape[:-3]        # (tasks,way*query_shot)
        img_shape = x_shot.shape[-3:]   # (c,h,w)
        
        x_shot = x_shot.view(-1, *img_shape)     # (tasks*ways*s_shots,3,h,w)
        x_query = x_query.view(-1, *img_shape)  # (tasks*ways*q_shots,3,h,w)
        
        # get attn_weights
        patch_emd = self._img2emd(img=torch.cat([x_shot, x_query]),kernel=self.kernel,stride=self.patchsz)
        attn_weights = self.attention(patch_emd,patch_emd,patch_emd,valid_lens=None)
   
        x_tot = self.encoder(torch.cat([x_shot, x_query], dim=0))
        p_feature, n_feature = self.get_PN_features(attn_weights,x_tot)
        
        x_shot, x_query = x_tot[:len(x_shot)], x_tot[-len(x_query):]
        x_shot = x_shot.view(*shot_shape, -1)   # (4,5,5,512)
        x_query = x_query.view(*query_shape, -1)    # (4,75,512)

        if self.method == 'cos':
            x_shot = x_shot.mean(dim=-2)    # mean basic prototype (4,5,512)
            x_shot = F.normalize(x_shot, dim=-1)
            x_query = F.normalize(x_query, dim=-1)
            metric = 'dot'
            # ========================================= prototype rectification
            # logits = F.softmax(utils.compute_logits(x_query, x_shot, metric=metric, temp=self.temp).permute(0,2,1),dim=-2)
            # perseudo_proto = torch.bmm(logits,x_query) 
            # x_shot = F.normalize((x_shot + perseudo_proto),dim=-1)
            # =========================================      
        elif self.method == 'sqr':
            x_shot = x_shot.mean(dim=-2)
            metric = 'sqr'

        logits = utils.compute_logits(
                x_query, x_shot, metric=metric, temp=self.temp)     # (4,75,5)(tasks,q_way*q_shot,q_way)
        return logits
    

# functions used in Attention Module
def sequence_mask(X, valid_len, value=0):
    """Mask irrelevant entries in sequences.

    Defined in :numref:`sec_seq2seq_decoder`"""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X

def masked_softmax(X, valid_lens):
    """通过在最后一个轴上掩蔽元素来执行softmax操作"""
    # X:3D张量，valid_lens:1D或2D张量
    if valid_lens is None:
        return F.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # 最后一轴上被掩蔽的元素使用一个非常大的负值替换，从而其softmax输出为0
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens,
                              value=-1e6)
        return F.softmax(X.reshape(shape), dim=-1)

def transpose_qkv(X, num_heads):
    """为了多注意力头的并行计算而变换形状"""
    # 输入X的形状:(batch_size，查询或者“键－值”对的个数，num_hiddens)
    # 输出X的形状:(batch_size，查询或者“键－值”对的个数，num_heads，num_hiddens/num_heads)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # 输出X的形状:(batch_size，num_heads，查询或者“键－值”对的个数,num_hiddens/num_heads)
    X = X.permute(0, 2, 1, 3)

    # 最终输出的形状:(batch_size*num_heads,查询或者“键－值”对的个数,num_hiddens/num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3])

def transpose_output(X, num_heads):
    """逆转transpose_qkv函数的操作"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)

# implementation of Attention Module
class DotProductAttention(nn.Module):
    """缩放点积注意力"""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # queries的形状：(batch_size，查询的个数，d)
    # keys的形状：(batch_size，“键－值”对的个数，d)
    # values的形状：(batch_size，“键－值”对的个数，值的维度)
    # valid_lens的形状:(batch_size，)或者(batch_size，查询的个数)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # 设置transpose_b=True为了交换keys的最后两个维度
        scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)
        attention_weights = masked_softmax(scores, valid_lens)
        return self.dropout(attention_weights)
        # return torch.bmm(self.dropout(attention_weights), values)   
     
class MultiHeadAttention(nn.Module):
    def __init__(self,key_size,query_size,value_size,num_hiddens,num_heads,dropout,patchsz,
                 img_height,
                 bias=False,**kwargs):
        super(MultiHeadAttention,self).__init__()
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.w_q = nn.Linear(query_size,num_hiddens,bias=bias)
        self.w_k = nn.Linear(key_size,num_hiddens,bias=bias)
        self.w_v = nn.Linear(value_size,num_hiddens,bias=bias)
        self.w_o = nn.Linear(num_hiddens,num_hiddens,bias=bias)
        self.w_o2 = nn.Linear(int(num_heads*((img_height/patchsz)**2)),int((img_height/patchsz)**2))
    
    def forward(self,queries,keys,values,valid_lens):
        queries = transpose_qkv(self.w_q(queries),self.num_heads)   # [batchsz*num_heads,查询或者“键－值”对的个数,num_hiddens/num_heads]
        keys = transpose_qkv(self.w_k(keys),self.num_heads)
        values = transpose_qkv(self.w_v(values),self.num_heads)
        
        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens,
                                                 repeats=self.num_heads,dim=0)
        
        output = self.attention(queries,keys,values,valid_lens)     # [batchsz*num_heads,num_queries,num_queries]

        output_concat = transpose_output(output,self.num_heads)     # [batchsz,num_queries,num_heads*num_queries]
        # TODO:这里通过全连接进行多头融合，有没有其他方法的多头融合呢？
        return self.w_o2(output_concat)

class Get_PN_features(nn.Module):
    def __init__(self,patch_nums=[32,16], split_thresh=0.5):
        super().__init__()
        self.patch_num = patch_nums
        self.split_thresh = nn.Parameter(torch.tensor(split_thresh))
        self.info_merge_layer = nn.AdaptiveAvgPool2d((patch_nums[0],patch_nums[1]))
        
    def forward(self,weight,encoder_out):
        batchsize, weight_h, weight_w = weight.shape
        _ , out_dim = encoder_out.shape
        weight = self.info_merge_layer(weight)  # [bs,patch_n,patch_n]
        positive_feature = torch.zeros(batchsize,self.patch_num[0],self.patch_num[1]).cuda()
        negative_feature = torch.zeros(batchsize,self.patch_num[0],self.patch_num[1]).cuda()
        # TODO: 根据阈值取出对应的正负块
        mask = weight > self.split_thresh
        assert out_dim == self.patch_num[0]*self.patch_num[1], 'out_dim must divide patch_num**2' 
        encoder_out = encoder_out.reshape(batchsize,self.patch_num[0],self.patch_num[1])
        
        positive_feature.masked_scatter(mask,encoder_out)
        negative_feature.masked_scatter(~mask,encoder_out)
        return positive_feature, negative_feature
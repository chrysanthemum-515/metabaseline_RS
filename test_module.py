import torch
import torch.nn as nn
import torch.nn.functional as F
import models
import utils
import math
from models.models import register

class MetaAttnBaseline(nn.Module):

    def __init__(self, encoder, encoder_args={}, method='cos',
                 temp=10., temp_learnable=True,
                 # attention params
                 model_dim=512,patchsz=4,num_heads=8):
        super().__init__()
        self.encoder = models.make(encoder, **encoder_args)
        self.method = method
        self.kernel = nn.Parameter(torch.randn(model_dim,c,patchsz,patchsz))
        
        self.attention = MultiHeadAttention(model_dim,model_dim,model_dim,model_dim,num_heads,0.5)
        
        if temp_learnable:
            self.temp = nn.Parameter(torch.tensor(temp))
        else:
            self.temp = temp

    def forward(self, x_shot, x_query):
        shot_shape = x_shot.shape[:-3]  # (4,5,5) (tasks,way,shot)
        query_shape = x_query.shape[:-3]    # (4,75) (tasks,way*shot)
        img_shape = x_shot.shape[-3:]   # (3,h,w)   (c,h,w)
        
        patch_emd = img2emd(img=x_shot,kernel=kernel,stride=patchsz)
        attn_weights = self.attention(patch_emd,patch_emd,patch_emd,valid_lens=None)
        
        x_shot = x_shot.view(-1, *img_shape)     # (100,3,h,w)
        x_query = x_query.view(-1, *img_shape)  # (300,3,h,w)
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
    
    def img2emd(img,kernel,stride):
        conv_out = F.conv2d(img,kernel,stride=stride)
        bs, oc, oh, ow = conv_out.shape
        embedding = torch.reshape(conv_out,shape=(bs,oc,oh*ow)).transpose(-1,-2)
        return embedding
    
# functions used in Attention Module
def sequence_mask(X, valid_len, value=0):
    """Mask irrelevant entries in sequences."""
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
    # 输出X的形状:(batch_size，查询或者“键－值”对的个数，num_heads，
    # num_hiddens/num_heads)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # 输出X的形状:(batch_size，num_heads，查询或者“键－值”对的个数,
    # num_hiddens/num_heads)
    X = X.permute(0, 2, 1, 3)

    # 最终输出的形状:(batch_size*num_heads,查询或者“键－值”对的个数,
    # num_hiddens/num_heads)
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
        self.attention_weights = masked_softmax(scores, valid_lens=None)
        return self.attention_weights
        # return torch.bmm(self.dropout(self.attention_weights), values)   
     
class MultiHeadAttention(nn.Module):
    def __init__(self,key_size,query_size,value_size,num_hiddens,num_heads,dropout,
                 bias=False,**kwargs):
        super(MultiHeadAttention,self).__init__()
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.w_q = nn.Linear(query_size,num_hiddens,bias=bias)
        self.w_k = nn.Linear(key_size,num_hiddens,bias=bias)
        self.w_v = nn.Linear(value_size,num_hiddens,bias=bias)
        self.w_o = nn.Linear(num_hiddens,num_hiddens,bias=bias)
    
    def forward(self,queries,keys,values,valid_lens):
        queries = transpose_qkv(self.w_q(queries),self.num_heads)   # [20,21,8]->[40,21,4]
        keys = transpose_qkv(self.w_k(keys),self.num_heads)
        values = transpose_qkv(self.w_v(values),self.num_heads)
        
        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens,
                                                 repeats=self.num_heads,dim=0)
        
        # output shape=[batchsz*num_heads,num_queries,num_hiddens/num_heads)
        output = self.attention(queries,keys,values,valid_lens)
        
        return output
        # output_concat = transpose_output(output,self.num_heads)
        # return self.w_o(output_concat)


if __name__ == '__main__':
    def img2emd(img,kernel,stride):
        conv_out = F.conv2d(img,kernel,stride=stride)
        bs, oc, oh, ow = conv_out.shape
        embedding = torch.reshape(conv_out,shape=(bs,oc,oh*ow)).transpose(-1,-2)
        return embedding
    
    tasks,ways,shots,c,h,w = 4,5,1,3,84,84
    patchsz = 4
    model_dim = 512
    num_heads = 2
    x_shot = torch.rand((tasks*ways*shots,3,h,w))
    
    kernel = torch.randn(model_dim,c,patchsz,patchsz)
    patch_emd = img2emd(img=x_shot,kernel=kernel,stride=patchsz)
    attention = MultiHeadAttention(model_dim,model_dim,model_dim,model_dim,num_heads,0.5)
    attn_weights = attention(patch_emd,patch_emd,patch_emd,valid_lens=None)
    print('ok')
    # conv_layer = torch.nn.Conv2d(3,1,3,padding=1)
    # x_shot = conv_layer(x_shot).squeeze(1)  # [20,84,84]
    # num_hiddens, num_heads = 512,2
    # batch_size,num_queries = 4,5    # num_queries = num_patches
    # attention = MultiHeadAttention(num_hiddens,num_hiddens,num_hiddens,
    #                                num_hiddens,num_heads,0.5)
    # X = torch.ones((batch_size,num_queries,num_hiddens))
    
    # print(attention(X,X,X,valid_lens=None).shape)

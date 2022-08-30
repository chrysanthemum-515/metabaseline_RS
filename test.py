import argparse
import os
import yaml

import scipy.stats
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader


import datasets
import models
import utils
import utils.few_shot as fs
from datasets.samplers import CategoriesSampler

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    se = scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return h

def sd(data):
    standard_derivation = np.std(data,ddof=1)
    return standard_derivation

def main(config):
    #### Dataset ####
    n_way, n_shot = 5, args.shot
    n_query = 15
    rng = np.random.RandomState(42)
    seed = rng.randint(0, 999999)
    if args.ep_per_batch is not None:
        ep_per_batch = args.ep_per_batch
    else:
        ep_per_batch = 1

    dataset = datasets.make(config['dataset'],
                                **config['dataset_args'])   # train_phase_train
    utils.log('test dataset: {} (x{}), {}'.format(
            dataset[0][0].shape, len(dataset),
            dataset.n_classes))
    batch_sampler = CategoriesSampler(
            dataset.label, 200,
            n_way, n_shot + n_query,
            ep_per_batch=ep_per_batch)
    loader = DataLoader(dataset, batch_sampler=batch_sampler,
                              num_workers=8, pin_memory=True)


    ########

    #### Model and optimizer ####

    if config.get('load'):
        model_sv = torch.load(config['load'])
        model = models.load(model_sv)
    else:
        model = models.make(config['model'], **config['model_args'])

    if config.get('load_encoder') is not None:
        encoder = models.load(torch.load(config['load_encoder'])).encoder   # map
        model.encoder.load_state_dict(encoder.state_dict())

    if config.get('_parallel'):
        model = nn.DataParallel(model)

    utils.log('num params: {}'.format(utils.compute_n_params(model)))
    model.eval()

    ########

    aves_keys = ['vl', 'va']
    aves = {k: utils.Averager() for k in aves_keys}
    max_epoch = 10
    va_lst = []
    np.random.seed(seed)
    for epoch in range(1, max_epoch + 1):
        
        for data, _ in tqdm(loader, leave=False):
            x_shot, x_query = fs.split_shot_query(
                    data.cuda(), n_way, n_shot, n_query,
                    ep_per_batch=ep_per_batch) 
            with torch.no_grad():
                label = fs.make_nk_label(n_way, n_query,
                        ep_per_batch=ep_per_batch).cuda()

                logits = model(x_shot, x_query).view(-1, n_way)
                loss = F.cross_entropy(logits, label)
                acc = utils.compute_acc(logits, label)
                aves['vl'].add(loss.item())
                aves['va'].add(acc)
                va_lst.append(acc)
        # print('test epoch {}: acc={:.2f} +- {:.2f} (%), loss={:.4f} (@{})'.format(
        #         epoch, aves['va'].item() * 100,
        #         mean_confidence_interval(va_lst) * 100,
        #         aves['vl'].item(), _[-1]))    # 0.95 confidence interval
        print('test epoch {}: acc={:.2f} +- {:.2f}, loss={:.4f} (@{})'.format(
                epoch, aves['va'].item() * 100,
                sd(va_lst),
                aves['vl'].item(), _[-1]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',default='meta-baseline/configs/test.yaml')
    parser.add_argument('--shot',default=5)
    parser.add_argument('--ep_per_batch',default=4)
    parser.add_argument('--gpu', default='6,7')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    if len(args.gpu.split(',')) > 1:
        config['_parallel'] = True
        config['_gpu'] = args.gpu

    utils.set_gpu(args.gpu)

    main(config)


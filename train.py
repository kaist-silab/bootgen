import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools
import time
import argparse
from torch.utils.data import DataLoader
from torch.distributions import Categorical
from dataset import SequenceDataset, ScoreDataset, ZipDataset
from tqdm import tqdm
from lib.acquisition_fn import get_acq_fn
from lib.dataset import get_dataset
from lib.oracle_wrapper import get_oracle
from lib.proxy import get_proxy_model
from lib.utils.distance import is_similar, edit_dist
from lib.utils.env import get_tokenizer
import design_bench
import random
from model.condlstm import CondDecoder
from design_bench.datasets.discrete_dataset import DiscreteDataset
import flexs

DEBUG_MODE = False
USE_CUDA = True
CUDA_NUM = 0

def normalize(dataset,y):
    y = (y - dataset.y.min())/(dataset.y.max()-dataset.y.min())
    return y


def collate(data_list):
    cond_data_list,target_data_list  = zip(*data_list)
    batched_target_data = SequenceDataset.collate(target_data_list)
    batched_cond_data = ScoreDataset.collate(cond_data_list)
    
    return batched_cond_data, batched_target_data

# proxy contruction code following GFN-AL
def construct_proxy(tokenizer,num_token,max_len,hparams):
    proxy = get_proxy_model(tokenizer,num_token,max_len)
    sigmoid = nn.Sigmoid()

    l2r = lambda x: x.clamp(min=0) / 1
    acq_fn = get_acq_fn()
    return acq_fn(proxy, l2r)




# Diversity measurement code following GFN-AL
def mean_pairwise_distances(seqs):
    dists = []
    for pair in itertools.combinations(seqs, 2):
        dists.append(edit_dist(*pair))
    return np.mean(dists)



# Boostrapping using training generator (for infering x), and proxy score function (for infering psuedo score y=proxy(x)) 
def bootstrapping(model, score_dataset,seq_dataset, proxy,num_token,max_len,device):
    
    model.eval()
    # score query to score-conditioned generator
    query = 1.0

    # make 1000 candidate from the score-conditioned generator
    x_tilde = model.decode(query,1000,device,max_len=max_len,start=num_token,temp=1)
    
    # evaluate the score using proxy function
    y_tilde = proxy.eval(x_tilde)
    y_psuedo = y_tilde.cpu().numpy()

    # filtering
    idx = np.argsort(y_psuedo,axis=0)
    y_psuedo = y_psuedo[idx]
    x_tilde = x_tilde.cpu().numpy() 
    x_tilde = x_tilde[idx][-2:].squeeze()
    y_tilde = y_psuedo[-2:]
    y_tilde = y_tilde.squeeze(1)

    # data preprocessing (this code is ugly)
    start_token = np.repeat(num_token,x_tilde.shape[0]).reshape(-1,1)
    x_tilde = np.concatenate((start_token,x_tilde),axis=1).tolist()
    
    # do not update duplicated sample (also ugly code)
    init_len = seq_dataset.__len__()

    for i in range(len(x_tilde)):
        found = False
        for seq in  seq_dataset.get_seq():
            if x_tilde[i] == seq:
                found = True
        if not found:
            seq_dataset.update([x_tilde[i]])
            score_dataset.update(y_tilde[i].reshape(1,-1).tolist())

    # bootstrapped training dataset
    return score_dataset,seq_dataset


# rank-based weighting for training dataset
def rank_weighted_training(model,score_dataset,seq_dataset,optimizer,hparams):
    
    model.train()
    dataset = ZipDataset(score_dataset, seq_dataset)

    # compute score ranking
    scores_np = score_dataset.get_tsrs().view(-1).numpy()
    ranks = np.argsort(np.argsort(-1 * scores_np))
    weights = 1.0 / (1e-2 * len(scores_np) + ranks)
    sampler = torch.utils.data.WeightedRandomSampler(
            weights=weights, num_samples=len(scores_np), replacement=True
            )
    loader = torch.utils.data.DataLoader(
        dataset, 
        sampler=sampler, 
        batch_size=256, 
        collate_fn=collate,
        drop_last=True
        )

    # training with the weighted training dataset
    step = 0
    while step < 10:
        step += 1
        try: 
            batched_data = next(data_iter)
        except:
            data_iter = iter(loader)
            batched_data = next(data_iter)

        batched_cond_data,batched_target_data = batched_data
        batched_cond_data = batched_cond_data.unsqueeze(1)
        
        loss = model(batched_target_data, batched_cond_data)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
    return model, optimizer,loss



def tostr(seqs):
    return ["".join([str(i) for i in x]) for x in seqs]

def train(hparams):

    # offline dataset    
    scores = task.y
    if task_dataset is not None:
        scores = normalize(task_dataset,scores)             
    sequences = task.x 

    # we do not use tokenizer but manually augment sequence with start token. This code is ugly.. 
    start_token = np.repeat(num_token,sequences.shape[0]).reshape(-1,1)
    sequences = np.concatenate((start_token,sequences),axis=1)
    seq_dataset = SequenceDataset(sequences.tolist())
    score_dataset = ScoreDataset(scores.tolist())


  

    # score-condtioned generator initialization. you can augment num_layers, hidden_dim and code_dim, but keep it same for every tasks
    model = CondDecoder(num_layers=2,hidden_dim=512,code_dim=256,num_token=num_token+1)

    optimizer = torch.optim.Adam(model.parameters(), lr=hparams.lr)
    # early stopping for gfp by reffering caribration model results
    if hparams.task == 'gfp':
        for stage in tqdm(range(300)):        
            # bootstrapping should be performed after some training of score-conditioned generator.  
            if stage > 250 and stage%5==0:
                score_dataset, seq_dataset = bootstrapping(model,score_dataset,seq_dataset, proxy,num_token,max_len,device)
            model, optimizer,loss = rank_weighted_training(model,score_dataset,seq_dataset,optimizer,hparams)
            # make validation using this y_calibration value and make tuning of early stopping by mornitering the y_calibration value 
            if stage%50 ==0:
                y_calibaration = calibration(model)
    else:
        for stage in tqdm(range(1500)):        
            # bootstrapping should be performed after some training of score-conditioned generator.  
            if stage > 1250 and stage%5==0:
                score_dataset, seq_dataset = bootstrapping(model,score_dataset,seq_dataset, proxy,num_token,max_len,device)
            model, optimizer,loss = rank_weighted_training(model,score_dataset,seq_dataset,optimizer,hparams)
            if stage%50 ==0:
                y_calibaration = calibration(model)
           
    return model

                    

def inference(model,proxy,num_token,max_len,device,temp=1):
    model.eval()
    score_query = 1.0
    batch = model.decode(score_query,1280, device,max_len=max_len,start=num_token,argmax=False,temp=1)
    # for uniqueness
    unique_batch_reshaped, indices = torch.unique(batch, dim=0, return_inverse=True)

    # Finally, we can reshape the tensor back to its original shape
    B_new = unique_batch_reshaped.size()[0]
    batch = unique_batch_reshaped.reshape(B_new, batch.shape[1])      
    # filtering with proxy model
    y_psuedo = proxy.eval(batch).cpu().numpy()
    idx = np.argsort(y_psuedo,axis=0)
    
    batch = batch.cpu().numpy() 
    batch = batch[idx][-128:].squeeze()

    y = oracle(batch)
    dist100 = mean_pairwise_distances(tostr(batch))
    if task_dataset is not None:
        y = normalize(task_dataset,y)

    return np.percentile(y, 50), np.percentile(y, 100), dist100,y_psuedo.mean(),y,batch



def calibration(model,temp=1):
    model.eval()
    score_query = 1.0
    batch = model.decode(score_query,1280, device,max_len=max_len,start=num_token,argmax=False,temp=1)
    
    # filtering with proxy model
    y_psuedo = proxy.eval(batch).cpu().numpy()


    return y_psuedo.mean()


def evaluation(models,num_token,max_len,device):


    # diverse aggregation
    if len(models)>1:
        ensemble_x = []
        ensemble_y = []
        for model in models:
            top_50, top_1, dist100,y_psuedo,y,x = inference(model,proxy,num_token,max_len,device)
            idx = np.random.permutation(128)
            n_subsamples = min(int(128/len(models)),128-len(ensemble_y)) 
            
            # random sub sampling
            y_rand = y[idx][:n_subsamples]
            x_rand = x[idx][:n_subsamples]
            ensemble_y.append(y_rand)
            ensemble_x.append(x_rand)
        maximum = np.percentile(ensemble_y,100)
        median = np.percentile(ensemble_y,50)
        ensemble_x = np.array(ensemble_x).reshape(128,-1)
        diversity = mean_pairwise_distances(tostr(ensemble_x))
        print("Percentile 50:", median)
        print("Percentile 100:", maximum)
        print("Diversity:", diversity)        

    else:
        model = models[0]
        top_50, top_1, dist100,y_psuedo,y,batch = inference(model,proxy,num_token,max_len,device)
        print("Percentile 50:", top_50)
        print("Percentile 100:", top_1)
        print("Diversity:", dist100)
        



    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="rna1")
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--save_solution', action='store_true')
    parser.add_argument('--load_proxy', action='store_true')
    if USE_CUDA:
        cuda_device_num = CUDA_NUM
        torch.cuda.set_device(cuda_device_num)
        device = torch.device('cuda', cuda_device_num)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        device = torch.device('cpu')
        torch.set_default_tensor_type('torch.FloatTensor')       
    # this is only an official hyperparameters for simple adaptation to new tasks. 
    # Please set higher learning rate (e.g., 5e-5), if the sequence dimension is high.
    parser.add_argument("--lr",type=float,default=1e-5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument('--DA', action='store_true')
    hparams = parser.parse_args()
    if hparams.task=="tfbind":
        from design_bench.datasets.discrete.tf_bind_8_dataset import TFBind8Dataset
        task_dataset = TFBind8Dataset()
        task = design_bench.make('TFBind8-Exact-v0')
        landscape = None
        num_token = 4
        max_len = 8
    elif hparams.task=="gfp":
        from design_bench.datasets.discrete.gfp_dataset import GFPDataset
        task_dataset = GFPDataset()
        task = design_bench.make('GFP-Transformer-v0')
        landscape = None
        num_token = 20
        max_len = 237
    elif hparams.task=="utr":
        from design_bench.datasets.discrete.utr_dataset import UTRDataset
        task_dataset = UTRDataset()
        task = design_bench.make('UTR-ResNet-v0')
        landscape = None
        num_token = 4
        max_len = 50
    # note we use 5000 RNA dataset where the maximum score is about 0.12
    elif hparams.task=="rna1":
        x = np.load('rna_data/RNA1_x.npy')
        y = np.load('rna_data/RNA1_y.npy').reshape(-1,1)
        problem = flexs.landscapes.rna.registry()['L14_RNA1']
        landscape = flexs.landscapes.RNABinding(**problem['params'])
        task = DiscreteDataset(x, y,num_classes=4)
        task_dataset = None 
        num_token = 4
        max_len = 14
    elif hparams.task=="rna2":
        x = np.load('rna_data/RNA2_x.npy')
        y = np.load('rna_data/RNA2_y.npy').reshape(-1,1)
        problem = flexs.landscapes.rna.registry()['L14_RNA2']
        landscape = flexs.landscapes.RNABinding(**problem['params'])
        task = DiscreteDataset(x, y,num_classes=4)
        task_dataset = None 
        num_token = 4
        max_len = 14
    elif hparams.task=="rna3":
        x = np.load('rna_data/RNA3_x.npy')
        y = np.load('rna_data/RNA3_y.npy').reshape(-1,1)
        problem = flexs.landscapes.rna.registry()['L14_RNA3']
        landscape = flexs.landscapes.RNABinding(**problem['params'])
        task = DiscreteDataset(x, y,num_classes=4)
        task_dataset = None 
        num_token = 4
        max_len = 14
    else:
        print("no such a task")
        assert(False)

    # this code is contruction of oracle score function (make available with batched tensor computation) 
    # and training dataset (to make training and validation set from the offline dataset of task.x, task.y)
    # this code is following GFN-AL code
    oracle = get_oracle(hparams.task,landscape)
    dataset = get_dataset(hparams.task, oracle,task_dataset)
    # actually, the tokenizer is useless but we just followed GFN-AL code base
    tokenizer = get_tokenizer(hparams.task)       
    proxy = construct_proxy(tokenizer,num_token,max_len,hparams)
    if hparams.load_proxy:
        # proxy loading for time saving
        proxy.load("pretrained_proxy/{}/proxy.pt".format(hparams.task))
    else:
        # training proxy following GFN-AL
        proxy.update(dataset)    
    if hparams.DA:
        models = []
        for i in range(8):
            seed = 8 * hparams.seed * i + i
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)
            model = train(hparams)
            models.append(model)
        evaluation(models,num_token,max_len,device)
    else:
        seed = hparams.seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        models = []
        model = train(hparams)
        models.append(model)

        evaluation(models,num_token,max_len,device)
        


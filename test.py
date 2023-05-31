import design_bench
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




def tostr(seqs):
    return ["".join([str(i) for i in x]) for x in seqs]



def inference(model,task,task_dataset,oracle,num_token,max_len,device,proxy,temp=1):
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




def evaluation(models,proxy,task, task_dataset,num_token,device, hparams):


    # diverse aggregation
    if len(models)>1:
        ensemble_x = []
        ensemble_y = []
        for model in models:
            top_50, top_1, dist100,y_psuedo,y,x = inference(model,task,task_dataset,oracle,num_token,max_len,device,proxy)
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
        return median,maximum, diversity,ensemble_y,ensemble_x        

    else:
        model = models[0]
        top_50, top_1, dist100,y_psuedo,y,x = inference(model,task,task_dataset,oracle,num_token,max_len,device,proxy)
        print("Percentile 50:", top_50)
        print("Percentile 100:", top_1)
        print("Diversity:", dist100)
        return top_50, top_1, dist100,y,x



    




    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="rna1")
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--save_solution', action='store_true')
    parser.add_argument('--load_proxy', action='store_true')
    
    # this is only an official hyperparameters for simple adaptation to new tasks. 
    # Please set higher learning rate (e.g., 5e-5), if the sequence dimension is high.
    parser.add_argument("--lr",type=float,default=1e-5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument('--DA', action='store_true')
    hparams = parser.parse_args()
    if USE_CUDA:
        cuda_device_num = CUDA_NUM
        torch.cuda.set_device(cuda_device_num)
        device = torch.device('cuda', cuda_device_num)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        device = torch.device('cpu')
        torch.set_default_tensor_type('torch.FloatTensor')       
    
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
        task = DiscreteDataset(x[:5000], y[:5000],num_classes=4)
        task_dataset = None 
        num_token = 4
        max_len = 14
    elif hparams.task=="rna2":
        x = np.load('rna_data/RNA2_x.npy')
        y = np.load('rna_data/RNA2_y.npy').reshape(-1,1)
        problem = flexs.landscapes.rna.registry()['L14_RNA2']
        landscape = flexs.landscapes.RNABinding(**problem['params'])
        task = DiscreteDataset(x[:5000], y[:5000],num_classes=4)
        task_dataset = None 
        num_token = 4
        max_len = 14
    elif hparams.task=="rna3":
        x = np.load('rna_data/RNA3_x.npy')
        y = np.load('rna_data/RNA3_y.npy').reshape(-1,1)
        problem = flexs.landscapes.rna.registry()['L14_RNA3']
        landscape = flexs.landscapes.RNABinding(**problem['params'])
        task = DiscreteDataset(x[:5000], y[:5000],num_classes=4)
        task_dataset = None 
        num_token = 4
        max_len = 14
    else:
        print("no such a task")
        assert(False)
    seed = 1
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    oracle = get_oracle(hparams.task,landscape)
    dataset = get_dataset(hparams.task, oracle,task_dataset)
    tokenizer = get_tokenizer(hparams.task)

    # actually, the tokenizer is useless but we just followed GFN-AL code base
    proxy = construct_proxy(tokenizer,num_token,max_len,hparams)
    proxy.load("pretrained_proxy/{}/proxy.pt".format(hparams.task))    
    if hparams.DA:
        models = []
        for i in range(8):
            model = CondDecoder(num_layers=2,hidden_dim=512,code_dim=256,num_token=num_token+1)
            model.load_state_dict(torch.load('pretrained_generator/'+hparams.task+'/gen-{}.pt'.format(i))['model_state_dict'])            
            models.append(model)
        
        medians = []
        maximums = []
        diversities = []
        for i in range(8):
            median,maximum, diversity,ensemble_y,ensemble_x  = evaluation(models,proxy,task, task_dataset,num_token,device, hparams)
            medians.append(median)
            maximums.append(maximum)
            diversities.append(diversity)
        medians = np.array(medians)
        maximums = np.array(maximums)
        diversities = np.array(diversities)
        print("percenile 100th {} +- {}".format(maximums.mean(), maximums.std()))
        print("percenile 50th {} +- {}".format(medians.mean(), medians.std()))
        print("Diversity {} +- {}".format(diversities.mean(), diversities.std()))
    else:
        medians = []
        maximums = []
        diversities = []
        for i in range(8):
            models = []
            
            model = CondDecoder(num_layers=2,hidden_dim=512,code_dim=256,num_token=num_token+1)
            model.load_state_dict(torch.load('pretrained_generator/'+hparams.task+'/new-gen-{}.pt'.format(i))['model_state_dict'])            
            models.append(model)
            median,maximum, diversity,ensemble_y,ensemble_x  = evaluation(models,proxy,task, task_dataset,num_token,device, hparams)
            medians.append(median)
            maximums.append(maximum)
            diversities.append(diversity)     
        medians = np.array(medians)
        maximums = np.array(maximums)
        diversities = np.array(diversities)       
        print("percenile 100th {} +- {}".format(maximums.mean(), maximums.std()))
        print("percenile 50th {} +- {}".format(medians.mean(), medians.std()))
        print("Diversity {} +- {}".format(diversities.mean(), diversities.std()))


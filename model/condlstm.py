import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F
import math


def compute_sequence_cross_entropy(logits, batched_sequence_data):
    logits = logits[:, :-1]
    targets = batched_sequence_data[:, 1:]

    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

    return loss

class LSTMDecoder(nn.Module):
    def __init__(self, num_layers, hidden_dim, code_dim,num_token=5):
        super(LSTMDecoder, self).__init__()

        self.encoder = nn.Embedding(num_token, hidden_dim)
        self.code_encoder = nn.Linear(code_dim, hidden_dim)
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            batch_first=True,
            num_layers=num_layers,
        )
        self.decoder = nn.Linear(hidden_dim, num_token-1)
        
        
    def forward(self, batched_sequence_data, codes):


  
        codes = codes.unsqueeze(1).expand(-1, batched_sequence_data.size(1), -1)
        sequences_embedding = self.encoder(batched_sequence_data)
        codes_embedding = self.code_encoder(codes)
    
        out = sequences_embedding + codes_embedding
        out, _ = self.lstm(out, None)
        out = self.decoder(out)

        return out


    def decode(self, codes, argmax, max_len,rand = False,start=4,temp=1):

        sample_size = codes.size(0)
        sequences = [torch.full((sample_size, 1), start, dtype=torch.long).to(codes.device)]
        hidden = None


        code_encoder_out = self.code_encoder(codes)
        for i in range(max_len):

            out = self.encoder(sequences[-1])
            out = out + code_encoder_out.unsqueeze(1)
            out, hidden = self.lstm(out, hidden)
            logit = self.decoder(out)
         
     
            prob = torch.softmax(logit/temp, dim=2)
            
            if argmax == True:
                tth_sequences = torch.argmax(logit, dim=2)
            elif rand == True:
                rand_prob = torch.ones_like(prob) * 1/prob.shape[2]
                distribution = Categorical(probs=rand_prob)
                tth_sequences = distribution.sample()                
            
            else:
                distribution = Categorical(probs=prob)
                tth_sequences = distribution.sample()
            sequences.append(tth_sequences)


        sequences = torch.cat(sequences, dim=1)

        return sequences[:,1:]







class CondDecoder(nn.Module):
    def __init__(self,num_token=5,num_layers=2,code_dim=256,hidden_dim=512):
        super(CondDecoder,self).__init__()
        
        # Representation of scores to the high dimensional vector
        self.cond_embedding = nn.Linear(1, code_dim)
        self.regressor = nn.Linear(hidden_dim,1)
        # conditional LSTM generator injected from code embedding. Try to use transformer if you can leverage more data.
        self.decoder = LSTMDecoder(
            num_layers=num_layers, 
            hidden_dim=hidden_dim, 
            code_dim=code_dim,
            num_token=num_token)
    
    def forward(self,batched_target_data, batched_cond_data):

        batched_cond_data = batched_cond_data.float()        
        codes = self.cond_embedding(batched_cond_data)
        logits = self.decoder(batched_target_data, codes)
    
        recon_loss = compute_sequence_cross_entropy(logits, batched_target_data)

        return recon_loss


    def decode(self, query, num_samples,device, max_len,start,temp=1,argmax=False,rand = False):
        with torch.no_grad():

            query_tsr = torch.full((num_samples, 1), query, device=device) 
       
            batched_cond_data = query_tsr 
            #batched_cond_data = torch.distributions.uniform.Uniform(0.6,1.0).sample([num_samples,1]).cuda()
            codes = self.cond_embedding(batched_cond_data)
            batched_sequence_data = self.decoder.decode(codes, argmax=argmax, max_len= max_len,start=start,temp=temp,rand = rand)
        
        return batched_sequence_data



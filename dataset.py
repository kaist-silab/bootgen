import torch
import numpy as np

class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, input_graph):
        super(GraphDataset, self).__init__()
        self.graph_list = input_graph

    def __getitem__(self, idx):
        graph = self.graph_list[idx]
 
        return graph

    def __len__(self):
        return len(self.graph_list)
    @staticmethod
    def collate(data_list):
        return torch.tensor(data_list)

class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, input_sequence):
        super(SequenceDataset, self).__init__()
        self.sequence_list = input_sequence
      
    def __getitem__(self, idx):
        
    
        sequence = self.sequence_list[idx]
        return sequence 

    def update(self, sequence):
        self.sequence_list.extend(sequence)

    def get_seq(self):
        return self.sequence_list

    def __len__(self):
        return len(self.sequence_list)

    def collate(data_list):
        return torch.tensor(data_list).long()

class ScoreDataset(torch.utils.data.Dataset):
    def __init__(self, input_score):
        super(ScoreDataset, self).__init__()
        self.scores = torch.FloatTensor(input_score)
        self.raw_tsrs = self.scores
        self.mean = self.scores.mean()
        self.std = torch.std(self.scores)
    def __getitem__(self, idx):
        return self.scores[idx]

    def update(self, scores):
        new_scores = torch.FloatTensor(scores)

        self.raw_tsrs = torch.cat([self.scores, new_scores], dim=0)
        self.scores = self.raw_tsrs
        self.mean = self.scores.mean()
        self.std = torch.std(self.scores)


    def get_tsrs(self):
        return self.scores

    def __len__(self):
        return self.scores.size(0)

    def collate(data_list):
        return torch.tensor(data_list)
class ZipDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __len__(self):
        return len(self.datasets[0])

    def __getitem__(self, idx):
        return [dataset[idx] for dataset in self.datasets]

    def collate(data_list):
        return [dataset.collate(data_list) for dataset, data_list in zip(self.datasets, zip(*data_list))]



class PairDataset(torch.utils.data.Dataset):
    def __init__(self, input_score, input_sequence):
        super(PairDataset, self).__init__()
        self.scores = torch.FloatTensor(input_score)


    def __getitem__(self, idx):
        return self.scores[idx]
        #return  self.score_list[idx]


    def update(self, scores):
        new_scores = torch.FloatTensor(scores)

        self.raw_tsrs = torch.cat([self.scores, new_scores], dim=0)
        self.scores = self.raw_tsrs
        self.mean = self.scores.mean()
        self.std = torch.std(self.scores)


    def get_tsrs(self):
        return self.scores

    def __len__(self):
        return self.scores.size(0)

    def collate(data_list):
        return torch.tensor(data_list)



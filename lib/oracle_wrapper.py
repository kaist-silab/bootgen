import numpy as np

#from clamp_common_eval.defaults import get_test_oracle
import design_bench
from design_bench.datasets.discrete_dataset import DiscreteDataset



def int_to_dna(integer_sequences):
    # Create a list of nucleotides
    nucleotides = ['T', 'G', 'C', 'A']

    # Convert each integer sequence in the list to a DNA sequence
    sequences = []
    for integer_sequence in integer_sequences:
        sequence = ''.join([nucleotides[i] for i in integer_sequence])
        sequences.append(sequence)

    return sequences

def int_to_rna(integer_sequences):
    # Create a list of nucleotides
    nucleotides = ['U', 'G', 'C', 'A']

    # Convert each integer sequence in the list to a DNA sequence
    sequences = []
    for integer_sequence in integer_sequences:
        sequence = ''.join([nucleotides[i] for i in integer_sequence])
        sequences.append(sequence)

    return sequences


def get_oracle(task,oracle ):
    if task == "gfp":
        return GFPWrapper()
    elif task == "tfbind":
        return TFBind8Wrapper()
    elif task == "utr":
        return UTRWrapper()
    elif task == "rna1":
        return RNA1Wrapper(oracle)
    elif task == "rna2":
        return RNA2Wrapper(oracle)
    elif task == "rna3":
        return RNA3Wrapper(oracle)
    else:
        print("unknown task")
        assert(False)





class UTRWrapper:
    def __init__(self):
        self.task = design_bench.make('UTR-ResNet-v0')

    def __call__(self, x, batch_size=256):
        scores = []
        for i in range(int(np.ceil(len(x) / batch_size))):
            s = self.task.predict(np.array(x[i*batch_size:(i+1)*batch_size])).reshape(-1)
            scores += s.tolist()
        return np.float32(scores)




class GFPWrapper:
    def __init__(self):
        self.task = design_bench.make('GFP-Transformer-v0')

    def __call__(self, x, batch_size=256):
        scores = []
        for i in range(int(np.ceil(len(x) / batch_size))):
            s = self.task.predict(np.array(x[i*batch_size:(i+1)*batch_size])).reshape(-1)
            scores += s.tolist()
        return np.float32(scores)

class TFBind8Wrapper:
    def __init__(self):
        self.task = design_bench.make('TFBind8-Exact-v0')

    def __call__(self, x, batch_size=256):
        scores = []
        for i in range(int(np.ceil(len(x) / batch_size))):
            s = self.task.predict(np.array(x[i*batch_size:(i+1)*batch_size]))
            scores += s.tolist()
        return np.array(scores)


class RNA1Wrapper:
    def __init__(self,oracle):
        x = np.load('rna_data/RNA1_x.npy')
        y = np.load('rna_data/RNA1_y.npy').reshape(-1,1)

        self.task = DiscreteDataset(x, y, num_classes=4)
        self.oracle = oracle
    def __call__(self, x, batch_size=256):
        scores = []
        for i in range(int(np.ceil(len(x) / batch_size))):
            
            batch = np.array(x[i*batch_size:(i+1)*batch_size])
            x1 = int_to_rna(batch.tolist())
            s = np.array(self.oracle.get_fitness(x1)).reshape(-1)
            scores += s.tolist()
        return np.float32(scores)

class RNA2Wrapper:
    def __init__(self,oracle):
        x = np.load('rna_data/RNA2_x.npy')
        y = np.load('rna_data/RNA2_y.npy').reshape(-1,1)

        self.task = DiscreteDataset(x, y, num_classes=4)
        self.oracle = oracle
    def __call__(self, x, batch_size=256):
        scores = []
        for i in range(int(np.ceil(len(x) / batch_size))):
            
            batch = np.array(x[i*batch_size:(i+1)*batch_size])
            x1 = int_to_rna(batch.tolist())
            s = np.array(self.oracle.get_fitness(x1)).reshape(-1)
            scores += s.tolist()
        return np.float32(scores)

class RNA3Wrapper:
    def __init__(self,oracle):
        x = np.load('rna_data/RNA3_x.npy')
        y = np.load('rna_data/RNA3_y.npy').reshape(-1,1)

        self.task = DiscreteDataset(x, y, num_classes=4)
        self.oracle = oracle
    def __call__(self, x, batch_size=256):
        scores = []
        for i in range(int(np.ceil(len(x) / batch_size))):
            
            batch = np.array(x[i*batch_size:(i+1)*batch_size])
            x1 = int_to_rna(batch.tolist())
            s = np.array(self.oracle.get_fitness(x1)).reshape(-1)
            scores += s.tolist()
        return np.float32(scores)
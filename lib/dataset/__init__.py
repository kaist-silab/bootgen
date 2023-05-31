

def get_dataset(task, oracle,task_dataset=None,relabel=False):

    if task=="rna1":
        from lib.dataset.regression import RNA1Dataset
        return RNA1Dataset(oracle)
    elif task=="rna2":
        from lib.dataset.regression import RNA2Dataset
        return RNA2Dataset(oracle)
    elif task=="rna3":
        from lib.dataset.regression import RNA3Dataset
        return RNA3Dataset(oracle)
    elif task=="tfbind":
        from lib.dataset.regression import TFBind8Dataset
        return TFBind8Dataset(oracle)
    elif task=="gfp":
        from lib.dataset.regression import GFPDataset
        return GFPDataset(oracle,task_dataset)
    elif task=="utr":
        from lib.dataset.regression import UTRDataset
        return UTRDataset(oracle,task_dataset)
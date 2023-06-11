# BootGen

Official code for BootGen: "Bootstrapped Training of Score-Conditioned Generator for Offline Design of Biological Sequences".

[![arXiv](https://img.shields.io/badge/arXiv-2306.03111-b31b1b.svg)](https://arxiv.org/abs/2306.03111)
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
![license](https://img.shields.io/badge/license-Apache%202.0-blue.svg?)

## Dependencies

* Python==3.7

```bash
pip install flexs
conda install -c bioconda viennarna
pip install design-bench==2.0.20
pip install polyleven
```


### Running BootGen

**GFP**
```bash
python train.py --DA --task gfp --lr 5e-5
```

**UTR**
```bash
python train.py --DA --task utr --lr 5e-5
```

**TFbind 8**
```bash
python train.py --DA --task tfbind
```

**RNA-A**
```bash
python train.py --DA --task rna1
```

**RNA-B**
```bash
python train.py --DA --task rna2
```

**RNA-C**
```bash
python train.py --DA --task rna3
```

To test BootGen without "diverse aggregation," simply remove the "--DA" flag.

### Cite us
If you find this code useful, please cite our paper:
```
@article{kim2023bootstrapped,
  title={Bootstrapped Training of Score-Conditioned Generator for Offline Design of Biological Sequences},
  author={Kim, Minsu and Berto, Federico and Ahn, Sungsoo and Park, Jinkyoo},
  journal={arXiv preprint arXiv:2306.03111},
  year={2023}
}
```
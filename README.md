# BootGen

This repository provides the source code for our BootGen algorithm, which is titled "Bootstrapped Training of Score-Conditioned Generator for Offline Design of Biological Sequences" and is being submitted to NeurIPS 2023.


## Dependancies

* Python==3.7

```bash
$pip install flexs
$conda install -c bioconda viennarna
$pip install design-bench==2.0.20
$pip install polyleven
```



### BootGen running

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









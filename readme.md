The folder contains the python source code that replicates the experiments of our paper: *Coresets for Wasserstein Distributionally Robust Optimization Problems*

## 1 Installation

#### Platform:

Ubuntu: 20.04

### Environment

environment_ubuntu.yml

```py
conda env create -f environment_ubuntu.yml
```

We utilized the commercial solver MOSEK. Please refer to https://www.mosek.com/products/academic-licenses/ for the license. 

## 2 Description

#### 2.1 Files & Folders

```
│ ...
│ 
├─code
│  │  coreset.py
│  │  DRLR_class.py
│  │  DRLR_letter.py
│  │  DRLR_mnist.py
│  │  environment_ubuntu.yml
│  │  hb_eg.py
│  │  Huber_class.py
│  │  Load_data.py
│  │  SVM_class.py
│  │  svm_le_a.py
│  │  svm_le_m.py
│  │  
│  ├─result
│  └─data
│ 
│ ...
```

### 2.2 Dataset

**Real dataset:**

MNIST: http://yann.lecun.com/exdb/mnist/

LETTER: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#letter

APPLIANCES ENERGY: https://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction

**Attack methods:** 

ALFA: https://github.com/feuerchop/ALFASVMLib

Min-max: https://github.com/kohpangwei/data-poisoning-journal-release

### 2.3 Execute

**Logistic regression:** 

run DRLR_letter.py, DRLR_mnist.py

**SVM:** 

run svm_le_a.py, svm_le_m.py

**Huber regression:**

run hb_eg.py

## 3 Experimental Results

Please refer to our paper. 


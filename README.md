# Deep Randomized Ensembles for Metric Learning

This repository contains a PyTorch(0.4.0) implementation of Deep Randomized Ensembles for Metric Learning(ECCV2018)

Paper link: https://arxiv.org/abs/1808.04469

Prepare the training data and testing data in python dictionary format. 

For example:
```
data_dict_tra  = {'class_tra_01':[image list],
                  'class_tra_02':[image list],
                  'class_tra_03':[image list],
                  ....,
                  'class_tra_XX':[image list]}
                 
data_dict_test = {'class_test_01':[image list],
                  'class_test_02':[image list],
                  'class_test_03':[image list],
                  ....,
                  'class_test_XX':[image list]}
```
                 

Replace data_dict_tra and data_dict_test in line 14 and line 18 of run.py

We only have the color nomarlization info for CUB, CAR, SOP, CIFAR100, In-shop cloth, and PKU vehicleID data. If you use other dataset please add the color nomarlization data in the directory: ```_code/color_lib.py```

We also supply the recall@K accuracy calculation process in the file: ```Recall.ipynb```


### Citation
```

```
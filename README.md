# WDR AT
This directory contains code to reproduce the results in our paper:

"Weight Decay Regularized Adversarial Training for Attacking Angle Imbalance" by Guorong Wang, Jinchuan Tang, Zehua Ding, Shuping Dang, Gaojie Chen.

#### Training datasets
Four datasets are used in our paper, including: MNIST (), CIFAR-10\100 (), and Tiny-ImageNet ().

#### Adversarial Training
To train an adversarial training model called FGSM contrast method, you can execute:
```
python3 FGSM-main.py
```
To train an adversarial training model of our FGSM method, you can execute:
```
python3 FGSM-main-WDR.py
```
To train an adversarial training model called FAT contrast method, you can execute:
```
python3 FAT-main.py
```
To train an adversarial training model of our FAT method, you can execute:
```
python3 FAT-main-WDR.py
```
To train an adversarial training model called PGD contrast method, you can execute:
```
python3 PGD-main.py
```
To train an adversarial training model of our PGD method, you can execute:
```
python3 PGD-main-WDR.py
```

To train an adversarial training model for the comparison method named PGI (code reference: https://github.com/jiaxiaojunQAQ/FGSM-PGI), you can execute:
```
python3 PGI-main.py
```

To train an adversarial training model for our PGI method, you can execute:
```
python3 PGI-main-WDR.py
```

#### Adversarial Attack Evaluation
To evaluate the performance of each adversarial model, you can execute:
```
python3 Evaluation.py
```

For experiments on other comparison methods in the paper, please refer to the following code: 
- `Sub AT`: https://github.com/nblt/Sub-AT
- `MEP AT`： https://github.com/Mengnan-Zhao-Happy/ConvergeSmooth
- `TRADES AT`: https://github.com/geyao1995/advancing-example-exploitation-in-adversarial-training

#### Citation:
Weight Decay Regularized Adversarial Training for Attacking Angle Imbalance

<h2 style="text-align: center;">Predicting protein-protein interaction sites with
high-order interaction Learning</h2>

![The overall architecture of HEGNN-PPIS.](./supp/HEGNN-PPIS.jpg)

Dependency
---
---
```commandline
dgl==1.0.2+cu116
numpy==1.22.4+mkl
numpy==1.24.3
pandas==1.5.3
scikit_learn==1.2.1
torch==1.13.1+cu116
torch_geometric==2.3.1
```

Train and Test
---
---

train:
```commandline
python train.py
```

test:
```commandline
python test.py
```

Output
---
---
```commandline
Test_60_best.pkl
========== Evaluate Test set ==========
Test loss:  0.3313633605837822
Test binary acc:  0.8904443091905052
Test precision: 0.6542010684798446
Test recall:  0.6491566265060241
Test f1:  0.6516690856313498
Test AUC:  0.9031634424589678
Test AUPRC:  0.6948013397793492
Test mcc:  0.5866768793461459
Threshold:  0.25

Test_315-28_best.pkl
========== Evaluate Test set ==========
Test loss:  0.30311310639157113
Test binary acc:  0.8846395918908175
Test precision: 0.5821278342053965
Test recall:  0.6623861779126781
Test f1:  0.6196690875334462
Test AUC:  0.9033724920204723
Test AUPRC:  0.6531124034708858
Test mcc:  0.5536100201290782
Threshold:  0.4

BTest_31-6_best.pkl
========== Evaluate Test set ==========
Test loss:  0.26363127171993256
Test binary acc:  0.9002387448840382
Test precision: 0.5901639344262295
Test recall:  0.6820027063599459
Test f1:  0.632768361581921
Test AUC:  0.9201093105383017
Test AUPRC:  0.6588535502237624
Test mcc:  0.5774103605926706
Threshold:  0.26

UBtest_31-6_best.pkl
========== Evaluate Test set ==========
Test loss:  0.4149226629734039
Test binary acc:  0.8707115092107487
Test precision: 0.4637096774193548
Test recall:  0.48523206751054854
Test f1:  0.4742268041237113
Test AUC:  0.8252349204342279
Test AUPRC:  0.39961346129565745
Test mcc:  0.4006974913649132
Threshold:  0.21
```

Visualization
---
---
![4kbmB.](./supp/4kbmB.jpg)
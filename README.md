# FedProc: Prototypical contrastive federated learning on non-IID data

> Xutong Mu, Yulong Shen, Ke Cheng, Xueli Geng, Jiaxuan Fu, Tao Zhang, Zhiwei Zhang
> *FGCS, 2023*

## Abstract

Federated learning (FL) enables multiple clients to jointly train high-performance deep learning models while maintaining the training data locally. However, it is challenging to accomplish this form of efficient collaborative learning when all the clients’ local data are not independent and identically distributed (i.e., non-IID). Despite extensive efforts to address this challenge, the results for image classification tasks remain inadequate. In this paper, we propose FedProc: prototypical contrastive federated learning. The core idea of this approach is to utilize the prototypes as global knowledge to correct the drift of each client’s local training. Specifically, we designed a local network structure and a global prototype contrast loss to regulate the training of the local model. These efforts make the direction of local optimization consistent with the global optimum such that the global model achieves good performance on non-IID data. Evaluative studies supported by theoretical significance demonstrate that FedProc improves accuracy by 1.6% to 7.9% with an acceptable computational cost compared to state-of-the-art federated learning methods.
[[paper]](https://www.sciencedirect.com/science/article/pii/S0167739X23000262/pdfft?md5=92fad7c3c1b6b952196ac5a8e83c52c3&pid=1-s2.0-S0167739X23000262-main.pdf). 

## Citation

Please cite our paper if you find this code useful for your research.

```
@article{FedProc_FGSC2023,
  title={Fedproc: Prototypical contrastive federated learning on non-iid data},
  author={Mu, Xutong and Shen, Yulong and Cheng, Ke and Geng, Xueli and Fu, Jiaxuan and Zhang, Tao and Zhang, Zhiwei},
  journal={Future Generation Computer Systems},
  volume={143},
  pages={93--104},
  year={2023},
  publisher={Elsevier}
}
```

## Acknowledgement

I would like to express my sincere appreciation to the authors of the [GitHub - QinbinLi/MOON: Model-Contrastive Federated Learning (CVPR 2021)](https://github.com/QinbinLi/MOON)) for their substantial contributions. 

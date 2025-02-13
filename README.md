# Bilinear-Adversarial-Network

code release for "BILINEAR ADVERSARIAL NETWORK FOR FINE-GRAINED DOMAIN ADAPTATION"

ABSTRACT

Fine-grained visual categorization (FGVC) is a challenging task due to larger intra-class variance and inter-class similarity. However, obtaining labelled samples for fine-grained datasets is more challenging than traditional datasets due to the need for expert-level domain knowledge. Consequently, the progress of FGVC has been limited by the availability of well-labelled datasets. In this paper, we propose a domain adaptation approach that leverages domain knowledge learned from existing large-scale fine-grained datasets to unlabeled real-life data, enabling FGVC applications in various daily life domains. Our approach utilizes high-dimensional features extracted by bilinear convolutional networks to bridge the gap between the source and target domains. We explore and compare different variants of bilinear convolutional networks, ultimately identifying the optimal method. Experimental results on two benchmarks demonstrate the effectiveness of our proposed approach. Additionally, detailed ablation experiments verify the contributions of each component in our method.

## Training
* `dann_customize.py` for training the proposed BAN and baselines
  
You may launch the program with `scripts/*.sh`

## Contact
Thanks for your attention! If you have any suggestion or question, you can leave a message here or contact us directly.

* yuwenqing@bupt.edu.cn

## Acknowledgement
Our code is mainly built upon [Transfer-Learning-Library](https://github.com/thuml/Transfer-Learning-Library). We appreciate their unreserved sharing.

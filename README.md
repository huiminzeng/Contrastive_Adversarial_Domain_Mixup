# Unsupervised Domain Adaptation for COVID-19 Information Service with Contrastive Adversarial Domain Mixup

This is the official code for the [ASONAM'22 paper](https://arxiv.org/abs/2210.03250) "Unsupervised Domain Adaptation for COVID-19 Information Service with Contrastive Adversarial Domain Mixup".

<img src=pics/intro.png>

In the real-world application of COVID-19 misinformation detection, a fundamental challenge is the lack of the labeled COVID data to enable supervised end-to-end training of the models, especially at the early stage of the pandemic. To address this challenge, we propose an unsupervised domain adaptation framework using contrastive learning and adversarial domain mixup to transfer the knowledge from an existing source data domain to the target COVID-19 data domain. In particular, to bridge the gap between the source domain and the target domain, our method reduces a radial basis function (RBF) based discrepancy between these two domains. Moreover, we leverage the power of domain adversarial examples to establish an intermediate domain mixup, where the latent representations of the input text from both domains could be mixed during the training process. Extensive experiments on multiple realworld datasets suggest that our method can effectively adapt misinformation detection systems to the unseen COVID-19 target domain with significant improvements compared to the state-ofthe-art baselines.

## Citing 

Please consider citing the following paper if you use our methods in your research:
```
@article{zeng2022unsupervised,
  title={Unsupervised Domain Adaptation for COVID-19 Information Service with Contrastive Adversarial Domain Mixup},
  author={Zeng, Huimin and Yue, Zhenrui and Kou, Ziyi and Shang, Lanyu and Zhang, Yang and Wang, Dong},
  journal={arXiv preprint arXiv:2210.03250},
  year={2022}
}
```

## Scripts for training CADM models.

   - Training baseline models
       ```
       python train_baseline.py --source_data_path './data/LIAR' --source_data_type 'liar';
       ```
   - Training domain discriminators
      ```
      CUDA_VISIBLE_DEVICES=0 python train_discriminator.py --source_data_path './data/LIAR' --source_data_type 'liar' --target_data_path './data/Constraint' --target_data_type 'constraint';
      CUDA_VISIBLE_DEVICES=0 python train_discriminator.py --source_data_path './data/LIAR' --source_data_type 'liar' --target_data_path './data/CoAID' --target_data_type 'coaid';
      CUDA_VISIBLE_DEVICES=0 python train_discriminator.py --source_data_path './data/LIAR' --source_data_type 'liar' --target_data_path './data/ANTiVax' --target_data_type 'antivax';
      ```
   - Perform contrastive adversarial domain mixup
      ```
      python train_rbf.py --alpha 0.1 --epsilon 0.5 --source_data_path './data/LIAR' --source_data_type 'liar' --target_data_path './data/Constraint' --target_data_type 'constraint';
      ```
   - Model evaluation
      ```
     python eval_models.py --eval_mode 'adapted_rbf' --alpha 0.1 --epsilon 0.5 --source_data_path './data/LIAR' --source_data_type 'liar' --target_data_path './data/Constraint' --target_data_type 'constraint';
      ```

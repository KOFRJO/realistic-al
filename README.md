# Realistic-AL (***REVIEWER VERSION***)
Preliminary version of the code for the empirical study in "Toward Realistic Evaluation of Deep Active Learning Algorithms in Image Classification" under review at "CVPR 2023".

## Purpose of this repository
Allowing reviewers to assess the code and verify the soundness of our results concerning methodical implementations guided by the project structure.
 

It is not yet a user-friendly tool which allows obtaining all of our results.

### Project Structure
```
├── src: Main Structure of project
│   ├── data: Datasets, Augmentations & DataModules (Pytorch-Lightning)
│   ├── models: Models with training and evaluation
│   │   ├── callbacks: Callbacks (Pytorch-Lightning)
│   │   ├── networks: Network Architectures
│   ├── plotlib: plotting scripts
│   ├── query: query functions
│   │   ├── batchbald_redux: BatchBald
│   └── utils: helper functions
├── skripts: Self-SL training (based on Pytorch-Lightning Bolts)
    └── config
        ├── data
        └── model
```

## State of the project
Singular trainings can be executed after the installation, but in this preliminary version there are no scripts for one-file execution, and plotting scripts are withheld to keep anonymity. The final open-source version will have all of these in a semi-automated manner to simplify the process of running Active Learning experiments.



## Installation
1. Create new python environment (v.3.8.13)
2. Install PyTorch
3. ```bash pip install -r requirements.txt```

If you have issues with CUDA, install a recent PyTorch version according to their website.

## Usage

To run experiments you need to set the following environment variables

```bash
export EXPERIMENT_ROOT=/absolute/path/to/your/experiments
export DATA_ROOT=/absolute/path/to/datasets
```

Alternatively, you may write them to a file and source that before running experiments, e.g.

```bash
mv example.env .env
```

Then edit `.env` to your needs and run

```bash
source .env
```

### Example runs for CIFAR10
Standard Trained Model
```bash
python src/main.py model=resnet query=random data=cifar10 active=cifar10_low optim=sgd_cosine ++data.val_size=250 ++trainer.seed=12345 ++trainer.max_epochs=200 ++model.dropout_p=0 ++model.learning_rate=0.1 ++model.use_ema=False ++data.transform_train=cifar_randaugmentMC ++trainer.precision=16 ++trainer.batch_size=1024 ++trainer.deterministic=True  ++trainer.experiment_name=cifar10/active-cifar10_low/basic_model-resnet_drop-0_aug-cifar_randaugmentMC_acq-random_ep-200
```

Self-Supervised Model
```bash
python src/main.py model=resnet query=random data=cifar10 active=cifar10_low optim=sgd ++data.val_size=250 ++trainer.seed=12345 ++trainer.max_epochs=80 ++model.dropout_p=0 ++model.learning_rate=0.001 ++model.freeze_encoder=False ++model.use_ema=False ++model.load_pretrained={pathtopretrained} ++data.transform_train=cifar_randaugment ++model.small_head=False ++trainer.precision=32 ++trainer.deterministic=True  ++trainer.experiment_name=cifar10/active-cifar10_low/basic-pretrained_model-resnet_drop-0.5_aug-cifar_randaugment_acq-batchbald_ep-80_freeze-False_smallhead-False
```

Semi-Supervised Model (FixMatch)
```bash
python src/main_fixmatch.py model=resnet_fixmatch data=cifar10 active=cifar10_low query=random optim=sgd_fixmatch ++data.val_size=250 ++model.dropout_p=0 ++model.learning_rate=0.03 ++model.small_head=True ++model.use_ema=False ++model.finetune=False ++model.load_pretrained=Null ++trainer.max_epochs=200 ++trainer.seed=12345 ++data.transform_train=cifar_basic ++sem_sl.eman=False ++trainer.precision=32 ++trainer.deterministic=True  ++trainer.experiment_name=cifar10/active-cifar10_low/fixmatch_model-resnet_fixmatch_drop-0_aug-cifar_basic_acq-random_ep-200
```
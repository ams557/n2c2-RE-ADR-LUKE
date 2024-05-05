# Clinical Relation Extraction of Adverse Drug Reactions from n2c2 2018 Task 2 RE Dataset using LUKE

## Installing Requirements

You can recreate python environment using Anaconda3

```python
conda env create -f requirements.yaml
```

## Rebuilding this Model from Scratch
To ensure smooth run of all modules, ensure the directory structure is exactly the same as the directory structure from the pulled repository & set up a [Weights & Biases](https://wandb.ai/site) (W&B) account.

### Required Directory Structure

```
.
├── README.md
├── datasets
│   └── n2c2
│       └── 2018
│           └── task2
│               └── RE
│                   ├── test
│                   │   ├── <BRAT FILES HERE>
│                   └── train
|                       ├── <BRAT FILES HERE>
├── experiments
│   └── n2c2
│       └── twenty18
│           └── task2
│               └── RE
│                   └── config.py
├── requirements.yaml
├── run_LUKE_2018n2c2-task2_RE.sh
├── src
│   ├── datamodules
│   │   └── LUKE
│   │       └── datamoduleRE.py
│   ├── models
│   │   └── pretrained
│   │       ├── LUKE.py
│   ├── preprocessing.py
│   ├── trainer.py
│   └── utils
│       ├── shared_utils.py
│       └── text_utils.py
```

### W&B Logging
After signing & setting up your W&B account, sign into [W&B](www.wandb.ai) & navigate to the `Authorize` page to retrieve your API key. In the `experiments/n2c2/twenty18/task2/RE/config.py` change the value of the `WANDB_key` variable name your API key.  

### Running the model
To train & test the model from scratch, run either...

```bash
$ python -m src.trainer.py
```
- If on local machine

```bash
$ sbatch run_LUKE_2018n2c2-task2_RE.sh
```
- If using SLURM

# References
- [LUKE: Deep Contextualized Entity Representations with Entity-Aware Self-Attention](https://arxiv.org/abs/2010.01057)
- [Supervised Relation Extraction with LukeForEntityPairClassification](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/LUKE/Supervised_relation_extraction_with_LukeForEntityPairClassification.ipynb#scrollTo=hDkptorP9Koh)
- [Finetune a Relation Classifier with Transformers and LUKE](https://lajavaness.medium.com/finetune-a-relation-classifier-with-transformers-and-luke-6c649440c663)
- [NLPatVCU/RelEx](https://github.com/NLPatVCU/RelEx)

NOTE - References to specific solutions given in code.
# RDA-Net

Official PyTorch implementation of **"EEG-based Motor Imagery Classification with Riemannian Domain Adaptation Network (RDA-Net)"** (Pattern Recognition, under review).

## Requirements

```bash
pip install -r requirements.txt
```

## Data

Place one `.mat` file per subject and session (fields `data` and `label`) in:

```
data/BCICIV_2a/A01T.mat ... A09E.mat
data/BCICIV_2b/B01T.mat ... B09E.mat
data/HGD/H01T.mat ... H14E.mat
```

## Usage

```bash
# Subject-dependent (seeds 0-4)
python main.py --dataset bci2a --protocol sd

# LOSO-SI: strict subject-independent (no target data)
python main.py --dataset bci2a --protocol loso_si --seeds 0

# LOSO-UDA: transductive (unlabelled target session T, tested on session E)
python main.py --dataset bci2a --protocol loso_uda --seeds 0
```

All hyperparameters are defined in one place, `rdanet/config.py`, and equal Table A.1 of the paper.
In both LOSO protocols, the model is selected by the validation loss of a held-out source subject;
target labels are used only once, for the final evaluation.

## Reproducing the paper

```bash
bash scripts/reproduce_all.sh
```

| Paper | Script |
|---|---|
| Tables 1-5, 7-9 | `main.py` (called by `reproduce_all.sh`) |
| Table 2, Table 9 (significance) | `scripts/significance.py` |
| Table 6 (complexity) | `scripts/complexity.py` |
| Table 7 (condition numbers) | `scripts/condition_numbers.py` |



Our research builds upon and improves the [EEG Conformer](https://github.com/eeyhsong/EEG-Conformer) and [RGN](https://github.com/shiroger/Riemannian-Geometry-Networks), and we sincerely thank the creators of these open-source project.


### Abstract:


### Overall Framework:
![architecture of RDA-Net](main/architecture.png)



### Citation
Hope this code can be useful. I would appreciate you citing us in your paper. 😊

## License

MIT


# STIP

It contains a pytorch implementation of the following paper:

* STIP: A Spatiotemporal Information-Preserving and Perception-Augmented Model for High-Resolution Video Prediction.

## Requirements

To install requirements:

```
pip install -r requirements.txt
```


## Quick Start


To test our model on the UCF Sport dataset, run the code in the terminal:

```
python STIP_run.py --dataset ucfsport
```


To test our model on the Human3.6M dataset, run the code in the PowerShell using:

```
python STRPM_run.py --dataset human36m
```

To test our model on the SJTU4K dataset, run the code in the PowerShell using:

```
python STRPM_run.py --dataset sjtu4k
```

The predicted results will be saved into path results/. The training code will be released soon.


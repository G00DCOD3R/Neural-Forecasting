# G-VAR

Graph vector autoregression for one-step-ahead neural forecasting on `cave_proofread` and `cave_all`: $\mathbf{x}_t = \sum_{p=1}^{P}\sum_{k=0}^{K} h_{kp}\,\mathbf{S}^k\mathbf{x}_{t-p}$, with $\mathbf{S}$ built from the connectome. The model is fitted in closed form (ridge), with P and K picked on validation, and compared with a copy-last baseline.

-   `data.py`: data loading, trial boundaries, train/val/test split, shift operator
-   `gvar.py`: the model
-   `train.py`: fitting, P/K selection and evaluation
-   `gvar_forecasting.ipynb`: fits and compares all configurations

```bash
python models/GVAR/train.py --dataset cave_proofread              # graph: connectome | random, optional --P --K
```

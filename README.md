# extrapolating-classifier-accuracy-to-bigger-datasets

## Table of Contents

- [Predicting Classifier Accuracy](#predicting-classifier-accuracy)
- [Reproducing Results](#reproducing-results)

## Predicting Classifier Accuracy

To use the Gaussian process presented in our paper see `notebooks/demo.ipynb`.

```python
likelihood = gpytorch.likelihoods.GaussianLikelihood()
# Note: If you want to use the Gaussian process with an arctan mean function use models.GPArctan() instead.
model = models.GPPowerLaw(X, y, likelihood, epsilon_min=0.05, with_priors=True)
```

To use the Gaussian process presented in our paper see `notebooks/demo.ipynb`.

## Reproducing Results

To reproduce model performance at varying dataset sizes download datasets (see `encode_images/README.md` and `label_images/README.md` for more details) and fit presented classifiers to each dataset (see `src/finetune_2D.py` and `src/finetune_3D.py` for more details). The results from our paper are saved in `experiments/`.

To reproduce learning curves with the results presented presented in our paper see `notebooks/figures.ipynb` and `notebooks/tables.ipynb`.

To use the Gaussian process presented in our paper see `notebooks/demo.ipynb`.

# extrapolating-classifier-accuracy-to-larger-datasets

## Table of Contents

- [Extrapolating Classifier Accuracy](#extrapolating-classifier-accuracy)
- [Citation](#citation)
- [Reproducing Results](#reproducing-results)

## Extrapolating Classifier Accuracy

To use our Gaussian process to extrapolate classifier accuracy to larger datasets see `notebooks/demo.ipynb`.

##### Initializing our Gaussian process

```python
likelihood = gpytorch.likelihoods.GaussianLikelihood()
# Note: If you want to use the Gaussian process with an arctan mean function use models.GPArctan() instead.
model = models.GPPowerLaw(X_train, y_train, likelihood, epsilon_min=0.05, with_priors=True)
```

##### Extrapolating Classifier Accuracy

```python
with torch.no_grad(): predictions = likelihood(model(X_test))
loc = predictions.mean.numpy()
scale = predictions.stddev.numpy()
# Note: If you want to forecast with 20%-80% change lower and upper percentile.
lower, upper = priors.truncated_normal_uncertainty(a=0.0, b=1.0, loc=loc, scale=scale, lower_percentile=0.025, upper_percentile=0.975) 
```

## Citation

```bibtex
@inproceedings{harvey2023probabilistic,
  author={Harvey, Ethan and Chen, Wansu and Kent, David M. and Hughes, Michael C.},
  title={A Probabilistic Method to Predict Classifier Accuracy on Larger Datasets given Small Pilot Data},
  booktitle={Machine Learning for Health},
  year={2023}
}
```

## Reproducing Results

To reproduce model performance at varying dataset sizes 1) download datasets (see `encode_images/README.md` and `label_images/README.md` for more details) and 2) fit classifiers to each dataset (see `src/finetune_2D.py` and `src/finetune_3D.py`). The results from our paper are saved in `experiments/`.

To reproduce learning curves with the results from our paper see `notebooks/figures.ipynb` and `notebooks/tables.ipynb`.

To use our Gaussian process to predict classifier accuracy on larger datasets given small pilot data see `notebooks/demo.ipynb`.

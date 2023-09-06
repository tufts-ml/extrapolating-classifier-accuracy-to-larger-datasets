# extrapolating-classifier-accuracy-to-bigger-datasets

## Table of Contents

- [Predicting Classifier Accuracy](#predicting-classifier-accuracy)
- [Reproducing Results](#reproducing-results)

## Predicting Classifier Accuracy



To use our Gaussian process to predict classifier accuracy on larger datasets given small pilot data  see `notebooks/demo.ipynb`.

##### Initializing our Gaussian process
After initializing our Gaussian proccess train the model until convergence.

```python
likelihood = gpytorch.likelihoods.GaussianLikelihood()
# Note: If you want to use the Gaussian process with an arctan mean function use models.GPArctan() instead.
model = models.GPPowerLaw(X_train, y_train, likelihood, epsilon_min=0.05, with_priors=True)
```

##### Predicting Classifier Accuracy

```python
with torch.no_grad(): predictions = likelihood(model(X_test))
loc = predictions.mean.numpy()
scale = predictions.stddev.numpy()
# Note: If you want to forecast with 20%-80% change lower and upper percentile.
lower, upper = priors.truncated_normal_uncertainty(0.0, 1.0, loc, scale, lower_percentile=0.025, upper_percentile=0.975) 
```

## Reproducing Results

To reproduce model performance at varying dataset sizes download datasets (see `encode_images/README.md` and `label_images/README.md` for more details) and fit presented classifiers to each dataset (see `src/finetune_2D.py` and `src/finetune_3D.py` for more details). The results from our paper are saved in `experiments/`.

To reproduce learning curves with the results presented presented in our paper see `notebooks/figures.ipynb` and `notebooks/tables.ipynb`.

To use our Gaussian process to predict classifier accuracy on larger datasets given small pilot data  see `notebooks/demo.ipynb`.

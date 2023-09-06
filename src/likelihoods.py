import warnings
import numpy as np
import torch
import gpytorch
import linear_operator
# Importing our custom module(s)
import priors

class CustomLikelihood(gpytorch.likelihoods.FixedNoiseGaussianLikelihood):
    r"""
    A Likelihood where noise is a function of some fixed heteroscedastic noise and 
    some learned homoskedastic noise.
    
    TODO: Make sure when calling the likelihood to pass noise=noise_func(X), not
          including the parameter name will result in unexpected behavior.
    
    Example:
        >>> def noise_func(X, b=2, x_n=360):
        >>>     m = (b-1)/(0-x_n)
        >>>     return torch.maximum(m*X + b, torch.tensor(1))
        >>> 
        >>> train_x = torch.randn(55, 2)
        >>> noises = torch.ones(55) * 0.01
        >>> likelihood = CustomLikelihood(noise=noise_func(train_x), learn_additional_noise=True)
        >>> pred_y = likelihood(gp_model(train_x))
        >>>
        >>> test_x = torch.randn(21, 2)
        >>> test_noises = torch.ones(21) * 0.02
        >>> pred_y = likelihood(gp_model(test_x), noise=noise_func(test_x))
    """
    def __init__(self, noise, learn_additional_noise=False, batch_shape=torch.Size(), **kwargs):
        super().__init__(noise=noise, learn_additional_noise=learn_additional_noise, batch_shape=batch_shape, **kwargs)
        
    @property
    def noise(self):
        return self.noise_covar.noise * self.second_noise # Changed addition to multiplication
        
    def _shaped_noise_covar(self, base_shape, *params, **kwargs):
        # Remove 'noise' kwarg (a Tensor) if provided
        kwargs_without_noise = kwargs.copy()
        if 'noise' in kwargs_without_noise:
            del kwargs_without_noise['noise']
        
        if len(params) > 0:
            # we can infer the shape from the params
            shape = None
        else:
            # here shape[:-1] is the batch shape requested, and shape[-1] is `n`, the number of points
            shape = base_shape

        res = self.noise_covar(*params, shape=shape, **kwargs)

        if self.second_noise_covar is not None:
            res = res * self.second_noise_covar(*params, shape=shape, **kwargs_without_noise) # Changed addition to multiplication and removed 'noise' kwarg
        elif isinstance(res, linear_operator.operators.ZeroLinearOperator):
            warnings.warn(
                "You have passed data through a FixedNoiseGaussianLikelihood that did not match the size "
                "of the fixed noise, *and* you did not specify noise. This is treated as a no-op.",
                gpytorch.utils.warnings.GPInputWarning,
            )

        return res
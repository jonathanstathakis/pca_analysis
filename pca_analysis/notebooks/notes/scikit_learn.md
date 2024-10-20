# Scikit Learn

## Design Notes

Scikit-learn is built around estimator objects (`BaseEstimator`) from which all other objects - estimators, classifiers, and transformers inherit their `fit` and *action* methods - `predict`, `transform`, etc^[https://scikit-learn.org/stable/developers/develop.html#estimators]. The library embraces duck typing and as a high level of flexibility is possible^[https://scikit-learn.org/stable/developers/develop.html#rolling-your-own-estimator].

The intended Estimator object API is as follows: objects are initialised with the parameters (hyperparameters) that determine their behavior. Then the `fit` method is called, to which the input `X` is passed. Finally a `predict`/`transform` is called to 
perfom the action^[https://scikit-learn.org/stable/developers/develop.html].

## Style Guide

Scikit-learn follows PEP8. Additionally: use snakecase when naming anything but classes, use relative imports except in the case of unit tests, and follow numpy docstring standard for docstrings^[https://scikit-learn.org/stable/developers/develop.html#coding-guidelines].

## Estimators

Estimators expect data $X$ of size (n_samples, n_features), i.e. a matrix with samples as rows and features as columns. This is referred to as a samples matrix, or design matrix. They are expected to be array-like^[https://scikit-learn.org/stable/getting_started.html#fitting-and-predicting-estimator-basics].

## Custom Estimator/Transformers

Custom Estimator/Transformer templates can be found [here](https://github.com/scikit-learn-contrib/project-template/blob/main/skltemplate/_template.py)

parameters (hyper) should be inputted within the object constructor. All parameters should be used to initialise attributes of the estimator as scikit-learn uses them for model selection. They provide a signature as follows:

```python
def __init__(self, param1=1, param2=2):
    self.param1 = param1
    self.param2 = param2
```

Furthermore no logic should be present in the initialisation block, rather all parameter logic should be within the fit/transform methods. This is all necessary because the model selection classes use the `BaseEstimator.set_params` method to modify the parameters^[https://scikit-learn.org/stable/developers/develop.html#instantiation] and thus the behavior should be the same as when calling `__init__`^[https://scikit-learn.org/stable/developers/develop.html#parameters-and-init]. The logic is that arguments that can be set prior to exposure to the data should be restricted to the `__init__` signature. The `fit` method is then used for input validation and model parameter computation, and rarely for input data associated information^[https://scikit-learn.org/stable/developers/develop.html#fitting]. It should be noted that in order for the greater scikit-learn api to function, unsupervised learning class `fit` methods still require a superfluous y input parameter, set to `y=None`. It is expected that `fit` returns `self` to enable method chaining. Generally speaking the objects will not store the input data.

Estimator attribute naming is as follows: if the attribute is estimated from the data then it is to have a '_' suffix, i.e. `clusters_`^[https://scikit-learn.org/stable/developers/develop.html#estimated-attributes]. These attributes are to be set in the `fit` and `predict`/`transform` methods. This fits both the design philosophy, and is used by the library machinery to check whether the estimator is in a *fitted* state^[https://scikit-learn.org/stable/developers/develop.html#parameters-and-init]. Miscallaneous rules include: if an estimator is iterative, the number of iterations should be named `n_iter`^[https://scikit-learn.org/stable/developers/develop.html#optional-arguments].

Estimator parameters are managed by `get_params` and `set_params` methods. `get_params` simply returns the input parameters, and if `deep=True` then it also recusively returns any subestimator parameters, following a naming pattern of `<estimator>_<parameter>`. In a similar manner, `set_params` takes the input parameters as arguments, modifies the object attributes and returns `self` Technically speaking, `get_params` is optional, but `set_params` is compulsory. Inheriting from `sklearn.base.BaseEstimator` provides `get_params` and `set_params` functionality^[https://scikit-learn.org/stable/developers/develop.html#get-params-and-set-params].

Custom Estimators can be validated through `sklearn.utils.estimator_checks.check_estimator`. Furthermore there is pytest integration^[https://scikit-learn.org/stable/modules/generated/sklearn.utils.estimator_checks.check_estimator.html#sklearn.utils.estimator_checks.check_estimator, https://scikit-learn.org/stable/developers/develop.html#rolling-your-own-estimator].

## Transformers

Transformers are related to estimators but implement a `transform` rather than `predict` method. `transform` outputs `Xt`, the transform off the input `X`^[https://scikit-learn.org/stable/getting_started.html#transformers-and-pre-processors].

### Column Transformers

Column Transformers can be used to transform specific feature columns of `X`^[https://scikit-learn.org/stable/getting_started.html#transformers-and-pre-processors].

## Pipelines

`Pipeline` can be used to combine transformers and estimators in a linear manner. The Pipeline API mimicks the `Estimator` API^[https://scikit-learn.org/stable/getting_started.html#pipelines-chaining-pre-processors-and-estimators]. A Pipeline must consist entirely of transformers except for an ultimate estimator which provides the Pipeline result. When `fit` is called, the Pipeline activates each step, fitting and transforming the data as it is passed through. When complete, the Pipeline wraps the final estimator, providing all attributes and methods of that object^[https://scikit-learn.org/stable/modules/compose.html#pipeline-chaining-estimators].

For a pipeline to provide a `score` method, it requires that the final estimator implements a `score` method that accepts an optional `y` argument^[https://scikit-learn.org/stable/developers/develop.html#pipeline-compatibility].

Using Pipelines is advantagous because it groups all steps under one `fit_predict`/`fit_transform` call, can optimize many estimators/transformers at once, and avoids data leakage during cross validation by managing the sample groups behind the scenes^[https://scikit-learn.org/stable/modules/compose.html#pipeline-chaining-estimators]. 

### Initialising a Pipeline

Pipelines are initialised from a list of two element tuples, where each tuple consists of a key: the label for the step, and an estimator object that provides the action of the step^[https://scikit-learn.org/stable/modules/compose.html#build-a-pipeline].

### Inspecting the Pipeline

Pipeline steps are accessed through its `steps` attribute. This is a list which can be sliced to return subsets of the pipeline. They can also be accessed by name through getitem notation (`pipeline[key]`), or through dots via the `named_steps` attribute^[https://scikit-learn.org/stable/modules/compose.html#access-pipeline-steps].

## Model Evaluation 

Scikit learn provides methods of cross-validation for model evaluation^[https://scikit-learn.org/stable/getting_started.html#model-evaluation].

See [here](https://scikit-learn.org/stable/modules/grid_search.html#model-specific-cross-validation) for a list of cross-validation methods.

## Hyperparameter Searching

Scikit learn provides means of automatic hyperparameter searching for model optimization, including randomised search. On a side note, the developers advise that searching includes all preprocessing in the form of a `Pipeline` to avoid data leakage ^[https://scikit-learn.org/stable/getting_started.html#automatic-parameter-searcheshttps://scikit-learn.org/stable/getting_started.html#automatic-parameter-searches].

During parameter search, a parameter is evaluated by the output of the the score function of the estimator^[https://scikit-learn.org/stable/modules/grid_search.html#specifying-an-objective-metric]. Multiple scoring metrics can be provided via the `scoring` parameter of cross-validation classes^https://scikit-learn.org/stable/modules/grid_search.html#specifying-multiple-metrics-for-evaluation. Parameters of Composite/nested estimators are handled through `<estimator>_<parameter>` syntax for keys of the `param_grid`. This can handle multiple levels of nesting^[https://scikit-learn.org/stable/modules/grid_search.html#composite-estimators-and-parameter-spaces].

## Scikit-Learn and Decomposition

Scikit-learn implements a number of decomposition methods which are discussed [here](https://scikit-learn.org/stable/modules/decomposition.html#)


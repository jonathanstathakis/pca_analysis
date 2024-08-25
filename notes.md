# Notes

A collection of notes taken during the course of this project. They will at the appropriate time be integrated into my PKM, if they fit.

## Multivariate Data

Spectral detection produces high dimensional, multivariate data [@filzmoser_19robustmultivariate_2020, p. 3-396, sec. "3.19.1.2 Visualising Multivariate Data for Outlier Detection"]

## Chemometric Assumptions

Many chemometric regression methods rely on or are derived from least squares, which assumes that data is normally distributed. [@filzmoser_19robustmultivariate_2020, p. 3-395, sec. "3.19.1.1 The Concept of Robustness"]

## Real Data is not Normally Distributed

Real data is rarely perfectly normally distributed. A common source of non-normality is variation in experimental conditions over time, a common occurance in chromatographic separation. Outliers are another source of non-normality [@filzmoser_19robustmultivariate_2020, p. 3-395, sec. "3.19.1.1 The Concept of Robustness"]. 

## Outliers

Outliers can be defined as samples who possess atypical properties when compared with the majority, either due to errors in observation or because they are truly different. [@filzmoser_19robustmultivariate_2020, p. 3-395, sec. "3.19.1.1 The Concept of Robustness"]

## Robust Estimators

Robust methods have been developed to accomodate for outliers while modeling a distribution. Specifically, it assumes that the data distribution consists of a *main* distribution $G$ and another distribution (containing the outliers) $H$, with an error term $\epsilon$: $$G_{ \epsilon }=( 1 - \epsilon ) G + \epsilon H$$ [@filzmoser_19robustmultivariate_2020, p. 3-395, sec. "3.19.1.1 The Concept of Robustness"]. A good estimator will perform equally well in the presence of outliers as not [@filzmoser_19robustmultivariate_2020, p. 3-396, sec. "3.19.1.3 Masking Effect"].

Robust estimators generally work by excluding outliers beyond a threshold, and otherwise lessening the weight of outliers proportionate to their distance from the center of the distribution such that their information still contributes to the model, but not in an outsized manner [@filzmoser_19robustmultivariate_2020, p. 3-396, sec. "3.19.1.6 "]

## Visualising Multivariate Data for Outlier Detction

While it is possible to visually identify outliers in uni or bivariate data, it is virtually impossible in multivariate as we are restricted to three dimensions. The only possible approach would be to plot pairs of variables, each pair at a time. This only shows outliers within the dimensions included, rather than the full complement, meaning that the vast majority of information within the multivariate data is discarded [@filzmoser_19robustmultivariate_2020, p. 3-396, sec. "3.19.1.2 Visualizing Multivariate Data for Outlier identification"]

## Approaches to Detecting Outliers in Multivariate Data

One method is to reduce the data through PCA to its components then observing the loading, scores and biplots. Other methods include partial least squares or cannonical correlation analysis. [@filzmoser_19robustmultivariate_2020, p. 3-396, sec. "3.19.1.3 Masking Effect"]. It is however better to use robust methods as in reality no data is totally normal, and classical methods such as those listed above are very vulnerable to masking and swamp effect.

## Classical Vs. Robust Methods

PCA, partial least squares, Canonical Corerelation Analysis etc. are classical methods, and as such outliers have an effect on the elucidated latent variables to the effect that no outliers are detected. This phenomenon is known as the masking effect [@filzmoser_19robustmultivariate_2020, p. 3-396, sec. "3.19.1.3 Masking Effect"].

## Effects of Outliers on Data Modeling

The presence of outliers can cause unprepared (classical) models to [mask the outliers](#masking-effect) and falsely [exclude valid data points](#swamp-effect) [@filzmoser_19robustmultivariate_2020, p. 3-396, sec. "3.19.1 Introduction"].



## Masking Effect

The *masking effect* is a phenomenon observed in classical data reduction methods such as PCA in which outliers are treated as part of the sample distribution and as such are hidden, or *masked* when observing the latent variables. The common solution is to instead use robust methods [@filzmoser_19robustmultivariate_2020, p. 3-396, sec. "3.19.1.3 Masking Effect"]

## Swamp Effect

The swamping effect describes the habit of outliers to warp nonrobust models such that datapoints that would otherwise be regarded as within the distribution are now outliers. A well known example is the effect that outliers have on linear regession [@filzmoser_19robustmultivariate_2020, p. 3-397, sec. "3.19.1.4 Swamping Effect"]

## PCA

PCA assumes that information is proportionate to variation [@oliveri_05applicationchemometrics_2020, p. 4-103, sec. "4.05.2.3.1 Principal Component Analysis (PCA)"].

### Preprocessing

Minimum preprocessing required in PCA is mean centering [@oliveri_05applicationchemometrics_2020, p. 4-103, sec. "4.05.2.3.1 Principal Component Analysis (PCA)"]

### Scores

After finding the principal components, the samples can be projected into the new space through the new variables. The values obtained are referred to as *scores* [@oliveri_05applicationchemometrics_2020, p. 4-103, sec. "4.05.2.3.1 Principal Component Analysis (PCA)"]

### Loadings

The coefficients of the *linear combination* of the orignal variables to produce the principal components are referred to as *loadings*. The greater the the absolute value of the loadings, the more correlated(?) the principal component is to the loaded variable [@oliveri_05applicationchemometrics_2020, p. 4-103, sec. "4.05.2.3.1 Principal Component Analysis (PCA)"].

### Application

PCA is most famous for visualisation of high dimensional data through score, loading and score-loadings biplots [@oliveri_05applicationchemometrics_2020, p. 4-103, sec. "4.05.2.3.1 Principal Component Analysis (PCA)"].


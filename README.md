# Dev Notes

## Dev Dependencies

This project uses nbstripout as a git attributes filter to avoid commiting content.

## Project Structure

This project is built around an analysis pipeline for the decomposition of chromatospectral images into their individual chemical species. The core data structure is the xarray Dataset/DataTree (Datatree is WIP) which will store the state change. The actions will be performed by a sklearn Pipeline with a wrapper to adapt it for xarray types. The main goal is to develop a set of preprocessing parameters to optimise the decomposition. Thus an interactive parameter selection interface is required, to identify the sharpening, smoothing, baseline correction and partitioning parameters. These parameters are then passed to the `Pipeline` to perform either a direct decomposition or serve as the basis of a cross-validated grid search. The parameters will be stored in dicts and unpacked into the Transformer initialisation. Thus, the program structure is as follows:

1. [ ] Loading the Input Data
    1. [ ] Observation of the data
        1. [ ] statistics
        2. [ ] visualisations
            1. [ ] heatmaps
            2. [ ] Curve overlays
    2. [ ] Preprprocessing and parameter selection
        1. [ ] smoothing
            1. [ ] line plot widget
            2. [ ] store parameters
        2. sharpening
            1. [ ] line plot widget
            2. [ ] store parameters
        3. [ ] (more of 1. and 2.?)
            1. [ ] line plot widget
            2. [ ] store parameters
        4. [ ] baseline correction
            1. [ ] line plot widget
            2. [ ] store parameters
        5. [ ] partitioning
            1. [ ] partitioning line plot widget
            2. [ ] store parameters
        6. [ ] rank estimation by partition
            1. [ ] PCA
                1. [ ] standard scaling (optional)
                2. [ ] PCA
                3. [ ] Viz
                4. [ ] store result
            2. [ ] CORCONDIA
                1. [ ] Standard scaling (optional)
                2. [ ] viz
                3. [ ] store result.
    3. [ ] PARAFAC2 Pipeline/grid search
        1. [ ] run pipeline
        2. [ ] viz factors
        3. [ ] viz components as widget
        4. [ ] viz components as overlay
        5. [ ] viz recon. vs input.


Note that we have had trouble producing single peak decompositions in complicated tensors
and may need to either enforce maximum number of peaks per partition OR invoke some curve fitting
OR secondary PARAFAC2. Also the PCA and CORCONDIA results appear to be somewhat random, possibly
around a distribution, and this should be investigated. We also need to add decomposition metrics
and more rank estimation methods. But this is enough to get going.


# PCA_Analysis

See [MOC](./docs/moc.md) for a description of the contents of this project.

# Experiments

All experiments are stored under "pca_analysis/experiments/notebooks/" with an "experiments" dir for daily (or so) EDA work, and a "notes" dir for more concise summaries of observations, interpretations, important information. For example, some EDA started on 2024-09-02 will be under "experiments", but a summary of it and say work from 2024-09-04 will be in "notes". It would be useful to link the related EDA notebooks in the summaries..

# A Description of Project Contents

The project consists of a package called 'pca_analysis', as that was the motivation for its genesis. It has since pivoted to PARAFAC2 modeling.

The package consists of the following modules:

- notebooks
- parafac2_pipeline
- db_to_cdf
- definitions
- get_sample_data
- run_etl_pipeline_raw

`get_sample_data` contains the functions `get_ids_by_varietal` which can fetch sample ids belonging toa varietal and `get_shiraz_data`, which combines `get_ids_by_varietal` with the `database_etl.get_data` to fetch the shiraz dataset. It also contains `get_zhang_data` which provides a `xr.DataArray` of the GC dataset from @zhang_flexibleimplementationtrilinearity_2022.

notebooks contains the following submodules:

- experiments
- projects

experiments contains EDA notebooks and prototype code modules, for example `parafac2_pipeline` which is an extensive prototype module for the parafac2 decomposition. Projects are special notebooks for managing 'experiment' notebooks, and manage predefined tasks. A project is linked to an experiment through a project value managed through YAML frontmatter in the first cell of an experiment notebook. This creates a heirarchy: main -> project -> experiment. The frontmatter of experiments contain the following fields: title, description, status, project, cdt, conclusion. These contain informative values which are displayed in a project's TOC. title is the display title of the experiment, description is a one-line summary of the goal, status is either 'open' or 'closed' and describes whether the experiment is concluded or not, 'project' is the project to which the experiment belongs, for example 'parafac2', cdt is the creation date of the experiment notebook, conclusion is a one-line conclusive summary of the experiment, to be filled at the end when the status is 'closed'.

`parafac2_app` is a dash web app for...
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

`parafac2_app` is a dash web app for 



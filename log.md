# Log

## 2024-08-15T09:55:23 - Establishing This Log.

As per log entry [[2024-08-14T16:40:17 - A Change of Tack]], I'm decentralising my work and will instead establish certain time periods after which I will manually centralise. Logs, notes, etc. Otherwise I spend too much focus maintaining where I should be moving. Thus this log will pertain to the PCA EDA and will be later integrated into the central log. #pca #project/pca_analysis

2024-08-15T09:58:24 - Getting Started. To proceed with the analysis, I first need to detect outliers. Even that has proved troublesome. A visual inspection indicates that sample 75 is an outlier. But how does one do that programmatically? Research has only provided me with more questions. The fundamentals are that it appears generally that the approach is to unfold the tensor such that each row is an observation with a time, wavelength and absorbance value, then perform PCA on that. But im not too sure about it. Frankly, I will need to find an answer. I'd rather proceed with dimension reduction for this stage rather than attempting PARAFRAC or other, until I have a better grasp of the techniques. On a side note, in the interest of following the local work paradigm mentioned in [the previous entry](#2024-08-15t095523---establishing-this-log), I will create a similar document for notes taken during the course of this project. It, like this document, will be atomised at the appropriate time.

2024-09-12T14:52:28. The dataset is a little bit.. unorganised. Notes on that are even more scattered. I will start constructing useful notes regarding the topic here.

2024-09-12T14:53:07. 3 of the raw samples from the wine deg study are at "/Users/jonathan/uni/0_jono_data/wine-deg-study/raw_uv/ambient" but have been included in the database to bring the total to 104 'raw' samples.

2024-09-13T15:58:37. All detections are in mau units.

2024-09-16T15:07:14. Need to reconstruct the database so its all in main, with a primary key based on a composite of 'st.pk' and 'chm.pk' representing every `chm.pk`th **sampling** of each `st.pk` sample.
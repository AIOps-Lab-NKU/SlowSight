# Project Structure
## Outer Layer Independent Files
### app.py
Stores the interfaces for interaction between the Flask backend and the frontend, in the order of the data management module interface, scatter labeling interface, and line labeling interface.

### call_method.py
Contains the interfaces for calling various algorithms. When adding new algorithms, new interfaces should be added and debugged within this file.

### config.json
Important file name storage.

### environment.yml
Environment configuration document that helps users quickly set up the backend environment.

### file_info.json
Stores the parsed results of the raw files to be labeled, including the types of entities contained in the dataset, machine IDs, and relevant metrics.

### result_info.json
Stores information about the result files, including the algorithms used and the types of result files.

## Cluster, Donut, hwTool
Contains the logic files for various algorithms. Simple algorithms like Kmeans and DBSCAN can be grouped together in Cluster, while more complex ones like Donut and hwTool can be stored separately.

## data
Stores the raw data to be labeled. In addition to the original data files, it also includes a metric_candidates.json file that records the column names of the dataset.

## result
Stores the result datasets, with the first level being the names of the processed datasets and the second level being the algorithms used.

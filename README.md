# Human_T_Cell_Profile

## Project Structure

### 1. [**Code_clusterization.ipynb**](/Code_clusterization.ipynb)

Data Preprocessing: involves deletion of doublets and outliers, standardization, and normalization of the flow cytometry data.
Dimensionality Reduction: implies UMAP to visualize multidimenstional data.
Clusterization: identification of distinct cell populations.
Identification of Cell Populations: the identification of specific cell populations based on their surface markers.

### 2. [**Code_classification.ipynb**](/Code_classification.ipynb)

Model Evaluation and Selection: detection of the optimal machine learning model for the characterization of immune cells.

## Data Acquisition

The dataset used in this study was obtained from [ImmPort](https://www.immport.org/home), a publicly available immunology database and analysis portal. Specifically, the data were sourced under the study accession SDY702, titled ‘Human T Cell Profile’. The particular dataset used in analysis comprises data obtained from immune cells extracted from the Mesenteric lymph node.

The dataset is associated with the research conducted by Thome JJ, Yudanin N, Ohmura Y, Kubota M, Grinshpun B, Sathaliyawala T, Kato T, Lerner H, Shen Y, and Farber DL, and their findings are detailed in the article titled "Spatial map of human T cell compartmentalization and maintenance over decades of life," published in [Cell](https://pubmed.ncbi.nlm.nih.gov/25417158/) on November 6, 2014.

## Setting Up the Conda Environment and working directory

To ensure the reproducibility of the code, it is essential to use the same versions of the packages and libraries as used in the original development. This can be achieved using Conda, a package and environment management system. Follow the steps below to set up your environment:

### Installing Conda

If you don't already have Conda installed, you can download and install it from [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) (a minimal installer for Conda) or [Anaconda](https://www.anaconda.com/download) (which includes Conda, Python, and a few other packages).

### Creating a Conda Environment

Open your terminal: Launch your terminal or command prompt.

Create a new Conda environment: Replace myenv with your desired environment name, and x.x with the Python version you wish to use.

```
conda create --name myenv python=3.11.6 
```

Activate the environment:

```
conda activate myenv
```

### Installing Specific Package Versions

Once your environment is activated, you can install specific versions of packages required for your project. For example:

```
conda install numpy=1.26.2 pandas=2.1.1 scikit-learn=1.3.2 seaborn=0.12.2 flowcal=1.3.0 scipy=1.11.4 umap-learn=0.5.5 hdbscan=0.8.33 joblib=1.2.0
```

### Setting the Working Directory

Before running the analysis scripts, it is essential to set the correct working directory. This ensures that the code can correctly locate and access the data files and other resources. Insert a path to your folder in two scripts:

```
os.chdir("/Users/moonbee/Cyto/Human_T_Cell_Profile/")
```

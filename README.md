# Human_T_Cell_Profile

## Project Structure

### 1. [**Code_clusterization.ipynb**](Code/Code_clusterization.ipynb)

Data Preprocessing: involves deletion of doublets and outliers, standardization, and normalization of the flow cytometry data.

Dimensionality Reduction: implies UMAP to visualize multidimenstional data.

Clusterization: identification of distinct cell populations.

Identification of Cell Populations: the identification of specific cell populations based on their surface markers.

### 2. [**Code_classification.ipynb**](Code/Code_classification.ipynb)

Model Evaluation and Selection: creaton of the machine learning model for the characterization of immune cells. For classification Logistic regression, K-Nearest Neighbor, Naive Bayes, Random Forest, and Neural Network were used. Optimal hyperparameters were set using GridSearchCV().

### 3. [**Classification.py**](Code/Classification.py); 4. [**Clusterization.py**](Code/Clusterization.py)

Generation of outputs (Tables, Models, Images) using console.

## Data Acquisition

The dataset used in this study was obtained from [ImmPort](https://www.immport.org/home), a publicly available immunology database and analysis portal. Specifically, the data were sourced under the study accession SDY702, titled ‘Human T Cell Profile’. The particular dataset used in analysis comprises data obtained from immune cells extracted from the Mesenteric lymph node.

The dataset is associated with the research conducted by Thome JJ, Yudanin N, Ohmura Y, Kubota M, Grinshpun B, Sathaliyawala T, Kato T, Lerner H, Shen Y, and Farber DL, and their findings are detailed in the article titled "Spatial map of human T cell compartmentalization and maintenance over decades of life," published in [Cell](https://pubmed.ncbi.nlm.nih.gov/25417158/) on November 6, 2014.

## Setting Up the Conda Environment and working directory

### Installing Conda

If you don't already have Conda installed, you can download and install it from [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) (a minimal installer for Conda) or [Anaconda](https://www.anaconda.com/download) (which includes Conda, Python, and a few other packages).

### Creating a Conda Environment

Open your terminal: Launch your terminal or command prompt.

Create a new Conda environment:

```
conda create --name myenv python=3.11.6 
```

Activate the environment:

```
conda activate myenv
```

### Installing Specific Package Versions

Once your environment is activated, you can install specific versions of packages required for your project. For this purpose use requirements.txt located in a main folder:

```
pip install -r requirements.txt
```

### Setting the Working Directory

Before running the .ipynb scripts, it is essential to set the correct working directory. This ensures that the code can correctly locate and access the data files and other resources. Insert a path to your folder in two .ipynb scripts:

```
os.chdir("<path-to-your-folder>/Human_T_Cell_Profile/")
```

All .py scripts are launched from the root folder (/Human_T_Cell_Profile) using command line with a specification of a root folder:

```
python /Code/Clusterization.py ./
python /Code/Classification.py ./
```

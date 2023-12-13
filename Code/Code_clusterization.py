# import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import FlowCal
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import linkage
from umap import UMAP
from hdbscan import HDBSCAN

# change working directory to where the data is
import os
os.chdir("/Users/moonbee/Cyto/Human_T_Cell_Profile/")

# ignore warnings
import warnings
warnings.filterwarnings('ignore')

# set the output directory for images 
output_directory_images = os.path.join(os.getcwd(), 'Images')

# read the cytometry data
fcs_table = FlowCal.io.FCSData('Data/Tissue Samples_M LN.596613.fcs')
fcs_table = fcs_table.astype('<f4')  # Convert to little-endian float format

pd_df = pd.DataFrame(fcs_table)

# add colnames
pd_df.columns = fcs_table.channels

# drop the time column
pd_df.drop(columns=['Time'], inplace=True)

# rename colnames of pd_df using biological values
colnames = list("FSC-A|SSC-A|CD14|CD103|HLADR|CD20|CD8|CD4|CD3|CD45RA|CCR7".split("|"))
pd_df.columns = colnames

# merge every graph for each column into one figure
num_cols = len(pd_df.columns)
num_rows = (num_cols // 4) + (num_cols % 4 > 0)

# set the figure size
fig, axs = plt.subplots(num_rows, 4, figsize=(15, 15))
fig.subplots_adjust(hspace=0.15, wspace=0.3)
axs = axs.ravel()

# plot the boxplot of raw data for each column
for i in range(num_rows * 4):
    if i < num_cols:
        axs[i].boxplot(pd_df[pd_df.columns[i]])
        axs[i].set_title(pd_df.columns[i])
        axs[i].set_xticks([])
        axs[i].grid()
    else:
        fig.delaxes(axs[i])

output_path_box_dist_raw = 'boxplot_distribution_of_unprocessed_counts.png'
full_output_path_box_dist_raw = os.path.join(output_directory_images, output_path_box_dist_raw)
plt.savefig(full_output_path_box_dist_raw, dpi=300)

# remove every row with max value in every column
for i in pd_df.columns:
    pd_df_filt = pd_df[pd_df[i] < 262143]

# plot scatterplots for identification of doublets

# merge subplots into one figure 2x2
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
fig.subplots_adjust(hspace=0.15, wspace=0.3)

# First subplot
axs[0, 0].scatter(pd_df_filt['CD3'], pd_df_filt['CD20'], s=0.1)
axs[0, 0].set_xlabel('CD3')
axs[0, 0].set_ylabel('CD20')
axs[0, 0].plot([8000, 8000], [0, 250000], color = 'r', linewidth = 0.5)
axs[0, 0].plot([0, 250000], [4000, 4000], color = 'r', linewidth = 0.5)
axs[0, 0].grid()

# Second subplot
axs[0, 1].scatter(pd_df_filt['CD14'], pd_df_filt['CD20'], s=0.1)
axs[0, 1].set_xlabel('CD14')
axs[0, 1].set_ylabel('CD20')
axs[0, 1].plot([8000, 8000], [0, 250000], color = 'r', linewidth = 0.5)
axs[0, 1].plot([0, 250000], [7000, 7000], color = 'r', linewidth = 0.5)
axs[0, 1].grid()

# Third subplot
axs[1, 0].scatter(pd_df_filt['CD4'], pd_df_filt['CD8'], s=0.1)
axs[1, 0].set_xlabel('CD4')
axs[1, 0].set_ylabel('CD8')
axs[1, 0].plot([12000, 12000], [0, 250000], color = 'r', linewidth = 0.5)
axs[1, 0].plot([0, 250000], [20000, 20000], color = 'r', linewidth = 0.5)
axs[1, 0].grid()

# Hide the fourth subplot
axs[1, 1].axis('off')

output_path_scatter_doub = 'scatterplot_identification_of_doublets.png'
full_output_path_scatter_doub = os.path.join(output_directory_images, output_path_scatter_doub)
plt.savefig(full_output_path_scatter_doub, dpi=300)

# filter cells by the expression of double markers
pd_df_filt = pd_df_filt[((pd_df_filt['CD3'] < 8000) | (pd_df_filt['CD20'] < 4000)) & ((pd_df_filt['CD14'] < 8000) | (pd_df_filt['CD20'] < 7000)) & ((pd_df_filt['CD4'] < 12000) | (pd_df_filt['CD8'] < 20000))]

# Define the biexponential transformation function
def biexponential_transform(x, a, c):
    return np.sign(x) * a * np.log10(1 + np.abs(x) / c)

transformed_data = pd.DataFrame()

a = 1 # unnescessary parameter due to scaling

# define unique percentiles for columns FSC-A and SSC-A
percentiles = {
    'FSC-A': 99,
    'SSC-A': 70
}

for column in pd_df_filt.columns:
    perc = percentiles.get(column, 30) # default percentile is 30
    c = np.percentile(np.abs(pd_df_filt[column]), perc)
    transformed_data[column] = biexponential_transform(pd_df_filt[column], a, c)

# perform normalization
scaler = MinMaxScaler()
scaled_data = pd.DataFrame(scaler.fit_transform(transformed_data), columns=transformed_data.columns)

# merge every graph for each column into one figure
num_cols = len(scaled_data.columns)
num_rows = (num_cols // 4) + (num_cols % 4 > 0)

fig, axs = plt.subplots(num_rows, 4, figsize=(15, 15))
fig.subplots_adjust(hspace=0.15, wspace=0.2)
axs = axs.ravel()

# plot the boxplot of scaled data for each column
for i in range(num_rows * 4):
    if i < num_cols:
        axs[i].boxplot(scaled_data[scaled_data.columns[i]])
        axs[i].set_title(scaled_data.columns[i])
        axs[i].set_xticks([])
        axs[i].grid()
    else:
        fig.delaxes(axs[i]) 

output_path_box_dist_transformed = 'boxplot_distribution_of_transformed_and_scaled_counts.png'
full_output_path_box_dist_transformed = os.path.join(output_directory_images, output_path_box_dist_transformed)
plt.savefig(full_output_path_box_dist_transformed, dpi=300)

# set rownames for scaled_data
scaled_data.index = pd_df_filt.index

# perform dimensionality reduction
umap = UMAP(n_components=2, n_neighbors=100, min_dist=0.3, random_state=42)
reduced_data_UMAP = umap.fit_transform(scaled_data)

# plot the reduced data
plt.figure(figsize=(10, 7))
plt.scatter(reduced_data_UMAP[:, 0], reduced_data_UMAP[:, 1], s=0.05, c='royalblue')
plt.title('UMAP')
plt.xlabel("UMAP_1")
plt.ylabel("UMAP_2")
plt.minorticks_on()

output_path_UMAP = 'UMAP.png'
full_output_path_UMAP = os.path.join(output_directory_images, output_path_UMAP)
plt.savefig(full_output_path_UMAP, dpi=300)

# set color map
sequential_colors = sns.color_palette("Blues", 10)
cmap = ListedColormap(sequential_colors)

# determine the number of rows and columns for the subplot grid
num_cols = len(scaled_data.columns)
num_rows = (num_cols // 2) + (num_cols % 2 > 0)

# create a figure with subplots
fig, axs = plt.subplots(num_rows, 2, figsize=(15, 3 * num_rows))
fig.subplots_adjust(hspace=0.15, wspace=0.05)

# flatten the axis array for easy indexing
axs = axs.ravel()

# loop over each column and create a UMAP scatter plot
for i, column in enumerate(scaled_data.columns):
    ax = axs[i]
    scatter = ax.scatter(reduced_data_UMAP[:, 0], reduced_data_UMAP[:, 1], s=0.05, c=scaled_data[column], cmap=cmap)
    ax.set_title(column)
    ax.set_xlabel("UMAP_1")
    ax.set_ylabel("UMAP_2")
    ax.minorticks_on()
    fig.colorbar(scatter, ax=ax)

# remove any extra axes
for j in range(i + 1, num_rows * 2):
    fig.delaxes(axs[j])

output_path_UMAP_markers = 'UMAP_markers.png'
fulloutput_path_UMAP_markers = os.path.join(output_directory_images, output_path_UMAP_markers)
plt.savefig(fulloutput_path_UMAP_markers, dpi=300)

# perform elbow method to find the optimal number of clusters
sum_of_squared_distances = []
K = range(2, 20)
for k in K:
    km = KMeans(n_clusters=k, n_init=20, random_state=42)
    km = km.fit(scaled_data)
    sum_of_squared_distances.append(km.inertia_)

# Plot the elbow plot
plt.figure(figsize=(10, 7))
plt.plot(K, sum_of_squared_distances, 'o-')
plt.xlabel('k')
plt.ylabel('Sum of squared distances')
plt.title('Elbow Method For Optimal k')
plt.xticks(np.arange(2, 20, 2))
plt.grid()

output_path_elbow = 'elbow_method.png'
fulloutput_path_elbow = os.path.join(output_directory_images, output_path_elbow)
plt.savefig(fulloutput_path_elbow, dpi=300)

# define color palette
cmap_KMeans = ListedColormap(sns.color_palette("Set2", 10))

# define the parameter grid
n_clusters_grid = [5, 6, 7, 8, 9, 10]

# merge subplots into one figure 3x2
fig, axs = plt.subplots(3, 2, figsize=(20, 15))
fig.subplots_adjust(hspace=0.3, wspace=0.2)

# perform the grid search
for n_clusters in n_clusters_grid:
    # define the model
    model = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
    # fit the model
    model.fit(scaled_data)
    # assign a cluster to each example
    yhat = model.predict(scaled_data)
    # retrieve unique clusters
    clusters = np.unique(yhat)
    # create scatter plot for samples from each cluster
    for cluster in clusters:
        # get row indexes for samples with this cluster
        row_ix = np.where(yhat == cluster)
        # create scatter of these samples
        axs[(n_clusters - 5) // 2, (n_clusters - 5) % 2].scatter(reduced_data_UMAP[row_ix, 0], reduced_data_UMAP[row_ix, 1], s=0.05, cmap=cmap_KMeans)
    # show the plot
    axs[(n_clusters - 5) // 2, (n_clusters - 5) % 2].set_title('k = ' + str(n_clusters))
    axs[(n_clusters - 5) // 2, (n_clusters - 5) % 2].set_xlabel("UMAP_1")
    axs[(n_clusters - 5) // 2, (n_clusters - 5) % 2].set_ylabel("UMAP_2")
    axs[(n_clusters - 5) // 2, (n_clusters - 5) % 2].minorticks_on()

output_path_KMeans = 'KMeans_clusterization.png'
fulloutput_path_KMeans = os.path.join(output_directory_images, output_path_KMeans)
plt.savefig(fulloutput_path_KMeans, dpi=300)

# define color palette
cmap_Gauss = ListedColormap(sns.color_palette("Set2", 10))

# define the parameter grid
n_components_grid = [4, 5, 6, 7, 8, 9]

# merge subplots into one figure 3x2
fig, axs = plt.subplots(3, 2, figsize=(20, 15))
fig.subplots_adjust(hspace=0.3, wspace=0.2)

# perform the grid search
for n_components in n_components_grid:
    # define the model
    gmm = GaussianMixture(n_components=n_components, covariance_type='tied', tol = 1e-5, random_state=42)
    # fit the model
    gmm.fit(scaled_data)
    # assign a cluster to each example
    yhat = gmm.predict(scaled_data)
    # retrieve unique clusters
    clusters = np.unique(yhat)
    # create scatter plot for samples from each cluster
    for cluster in clusters:
        # get row indexes for samples with this cluster
        row_ix = np.where(yhat == cluster)
        # create scatter of these samples
        axs[(n_components - 4) // 2, (n_components - 4) % 2].scatter(reduced_data_UMAP[row_ix, 0], reduced_data_UMAP[row_ix, 1], s=0.05, cmap=cmap_Gauss)
    # show the plot
    axs[(n_components - 4) // 2, (n_components - 4) % 2].set_title('k = ' + str(n_components))
    axs[(n_components - 4) // 2, (n_components - 4) % 2].set_xlabel("UMAP_1")
    axs[(n_components - 4) // 2, (n_components - 4) % 2].set_ylabel("UMAP_2")
    axs[(n_components - 4) // 2, (n_components - 4) % 2].minorticks_on()

output_path_Gauss = 'Gaussian_Mixture_clusterization.png'
fulloutput_path_Gauss = os.path.join(output_directory_images, output_path_Gauss)
plt.savefig(fulloutput_path_Gauss, dpi=300)

# define color palette
cmap_hdb = ListedColormap(sns.color_palette("Set2", 10))

# define the parameter grid
min_cluster_size_grid = [5, 10, 20, 50]

# merge subplots into one figure 2x2
fig, axs = plt.subplots(2, 2, figsize=(15, 15))
fig.subplots_adjust(hspace=0.3, wspace=0.2)

# perform the grid search
for num, min_cluster_size in enumerate(min_cluster_size_grid):
    # create an HDBSCAN model
    hdb = HDBSCAN(min_cluster_size=min_cluster_size, gen_min_span_tree=True)

    # fit the model
    yhat = hdb.fit_predict(scaled_data)

    # retrieve unique clusters
    clusters = np.unique(yhat)

    # create scatter plot for samples from each cluster
    for cluster in clusters:
        # get row indexes for samples with this cluster
        row_ix = np.where(yhat == cluster)
        # create scatter of these samples
        axs[num // 2, num % 2].scatter(reduced_data_UMAP[row_ix, 0], reduced_data_UMAP[row_ix, 1], s=0.05, cmap=cmap_hdb)
        
    # show the plot
    axs[num // 2, num % 2].set_title('min_cluster_size = ' + str(min_cluster_size))
    axs[num // 2, num % 2].set_xlabel("UMAP_1")
    axs[num // 2, num % 2].set_ylabel("UMAP_2")
    axs[num // 2, num % 2].minorticks_on()

output_path_HDBSCAN = 'HDBSCAN_clusterization.png'
fulloutput_path_HDBSCAN = os.path.join(output_directory_images, output_path_HDBSCAN)
plt.savefig(fulloutput_path_HDBSCAN, dpi=300)

# visualize the best clustering model and assign cluster names

# define the final model
gmm = GaussianMixture(n_components=8, covariance_type='tied', tol = 1e-5, random_state=42)
# fit the model
gmm.fit(scaled_data)
# assign a cluster to each example
yhat = gmm.predict(scaled_data)

# set clusters for each cell
scaled_data['Cluster'] = yhat

# create a dictionary that maps cluster numbers to cell types
cluster_names = {
    0: "CD4+ T cells", 
    1: "CD20+ B cells", 
    2: "CD20+ B cells", 
    3: "CD8+ T cells", 
    4: "CD20+ B cells", 
    5: "CD14+ Monocytes", 
    6: "CD4+ T cells", 
    7: "Other cells"
    }

# rename clusters 
scaled_data['Population'] = scaled_data['Cluster'].map(cluster_names)

# delete numeral values of clusters
scaled_data = scaled_data.drop(['Cluster'], axis=1)

# visualize the final clustering model
plt.figure(figsize=(10, 7))

# get unique labels
unique_labels = scaled_data['Population'].unique()

# create scatter plot for samples from each cluster
# plot each cluster with a different color and label
for label in unique_labels:
    row_ix = np.where(scaled_data['Population'] == label)
    plt.scatter(reduced_data_UMAP[row_ix, 0], reduced_data_UMAP[row_ix, 1], s=0.05, label=label)

plt.title('Final clustering model')
plt.xlabel("UMAP_1")
plt.ylabel("UMAP_2")
plt.minorticks_on()

# add a legend
plt.legend(title='Population', loc='upper right', fontsize='small', markerscale=6)

output_path_final = 'Gaussian_Mixture_final_clusterization.png'
fulloutput_path_final = os.path.join(output_directory_images, output_path_final)
plt.savefig(fulloutput_path_final, dpi=300)

# sample 10% of the data for faster processing
sampled_data = scaled_data.sample(frac=0.1)

# get the mean of each cluster by each column
cluster_means = sampled_data.groupby('Population').mean()

# perform hierarchical clustering for clusters by mean expression
Z = linkage(cluster_means, 'ward')

# create a clustermap with seaborn
g = sns.clustermap(cluster_means, row_linkage=Z, cmap="Blues", col_cluster=False, yticklabels=True, figsize=(6, 6), vmin=0, vmax=1)

# Rotate the row and column labels
plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45)
plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)

heatmap_output_path = 'heatmap_of_cluster_means.png'
fullheatmap_output_path = os.path.join(output_directory_images, heatmap_output_path)
plt.savefig(fullheatmap_output_path, dpi=300)

# add cluster labels to the original dataframe
pd_df_filt['Population'] = scaled_data["Population"]

# write the final dataframe to a csv file
pd_df_filt.to_csv('Data/processed_data.csv', index=False)
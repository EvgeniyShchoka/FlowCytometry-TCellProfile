# Import packages
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
import os
import sys

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Function to read cytometry data
def read_cytometry_data(file_path):
    fcs_data = FlowCal.io.FCSData(file_path)
    fcs_data = fcs_data.astype('<f4')  # Convert to little-endian float format
    return pd.DataFrame(fcs_data, columns=fcs_data.channels)

# Function for data preprocessing
def preprocess_data(df):
    df.drop(columns=['Time'], inplace=True)
    colnames = "FSC-A|SSC-A|CD14|CD103|HLADR|CD20|CD8|CD4|CD3|CD45RA|CCR7".split("|")
    df.columns = colnames
    return df[df.max(axis=1) < 262143]  # Remove rows with max value in any column

# Function to plot scatterplots for doublet identification and filter doublets
def identify_and_filter_doublets(df, output_directory):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    fig.subplots_adjust(hspace=0.15, wspace=0.3)

    # Configure scatterplots
    axs[0, 0].scatter(df['CD3'], df['CD20'], s=0.1)
    axs[0, 0].set_xlabel('CD3')
    axs[0, 0].set_ylabel('CD20')
    axs[0, 0].plot([8000, 8000], [0, 250000], color='r', linewidth=0.5)
    axs[0, 0].plot([0, 250000], [4000, 4000], color='r', linewidth=0.5)
    axs[0, 0].grid()

    axs[0, 1].scatter(df['CD14'], df['CD20'], s=0.1)
    axs[0, 1].set_xlabel('CD14')
    axs[0, 1].set_ylabel('CD20')
    axs[0, 1].plot([8000, 8000], [0, 250000], color='r', linewidth=0.5)
    axs[0, 1].plot([0, 250000], [7000, 7000], color='r', linewidth=0.5)
    axs[0, 1].grid()

    axs[1, 0].scatter(df['CD4'], df['CD8'], s=0.1)
    axs[1, 0].set_xlabel('CD4')
    axs[1, 0].set_ylabel('CD8')
    axs[1, 0].plot([12000, 12000], [0, 250000], color='r', linewidth=0.5)
    axs[1, 0].plot([0, 250000], [20000, 20000], color='r', linewidth=0.5)
    axs[1, 0].grid()

    # Hide the fourth subplot
    axs[1, 1].axis('off')

    # Save the scatterplot figure
    output_path_scatter_doub = 'scatterplot_identification_of_doublets.png'
    full_output_path_scatter_doub = os.path.join(output_directory, output_path_scatter_doub)
    plt.savefig(full_output_path_scatter_doub, dpi=300)

    # Filter doublets based on marker expression thresholds
    df = df[((df['CD3'] < 8000) | (df['CD20'] < 4000)) & ((df['CD14'] < 8000) | (df['CD20'] < 7000)) & ((df['CD4'] < 12000) | (df['CD8'] < 20000))]

    return df

# Function for biexponential transformation
def biexponential_transform(x, a, c):
    return np.sign(x) * a * np.log10(1 + np.abs(x) / c)

# Function to transform and scale data
def transform_and_scale_data(df, percentiles):
    transformed_data = pd.DataFrame()
    a = 1  # Unnecessary parameter due to scaling
    for column in df.columns:
        c = np.percentile(np.abs(df[column]), percentiles.get(column, 30))
        transformed_data[column] = biexponential_transform(df[column], a, c)
    scaler = MinMaxScaler()
    return pd.DataFrame(scaler.fit_transform(transformed_data), columns=transformed_data.columns)

# Function to plot boxplots
def plot_boxplots(df, output_path, title):
    num_cols = len(df.columns)
    num_rows = (num_cols // 4) + (num_cols % 4 > 0)
    fig, axs = plt.subplots(num_rows, 4, figsize=(15, 15))
    fig.suptitle(title, fontsize=16)
    fig.subplots_adjust(hspace=0.2, wspace=0.3)
    axs = axs.ravel()
    for i in range(num_rows * 4):
        if i < num_cols:
            axs[i].boxplot(df[df.columns[i]])
            axs[i].set_title(df.columns[i])
            axs[i].set_xticks([])
            axs[i].grid()
        else:
            fig.delaxes(axs[i])
    plt.savefig(output_path, dpi=300)

# Function for UMAP dimensionality reduction
def reduce_dimensionality_UMAP(data, n_components=2, n_neighbors=100, min_dist=0.3, random_state=42):
    umap = UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
    return umap.fit_transform(data)

# Function to plot UMAP results
def plot_UMAP(data, output_path):
    plt.figure(figsize=(10, 7))
    plt.scatter(data[:, 0], data[:, 1], s=0.05, c='royalblue')
    plt.title('UMAP')
    plt.xlabel("UMAP_1")
    plt.ylabel("UMAP_2")
    plt.minorticks_on()
    plt.savefig(output_path, dpi=300)

# Function to plot UMAP results colored by markers
def plot_UMAP_markers(data, df, output_directory):
    markers = df.columns
    
    fig, axs = plt.subplots(len(markers) // 2 + len(markers) % 2, 2, figsize=(20, 30))
    fig.subplots_adjust(hspace=0.35, wspace=0.2)
    axs = axs.ravel()
    sequential_colors = sns.color_palette("Blues", 10)
    cmap = ListedColormap(sequential_colors) 

    for i, marker in enumerate(markers):
        scatter = axs[i].scatter(data[:, 0], data[:, 1], s=0.05, c=df[marker], cmap=cmap)
        axs[i].set_title(marker)
        axs[i].set_xlabel("UMAP_1")
        axs[i].set_ylabel("UMAP_2")
        axs[i].minorticks_on()
        fig.colorbar(scatter, ax=axs[i])

    # Remove extra subplots if markers count is odd
    if len(markers) % 2 != 0:
        fig.delaxes(axs[-1])
    
    plt.savefig(os.path.join(output_directory, 'UMAP_markers.png'), dpi=300)


# Function to plot clustering results
def plot_clustering(data, labels, title, output_path):
    plt.figure(figsize=(10, 7))
    plt.scatter(data[:, 0], data[:, 1], s=0.05, c=labels, cmap=ListedColormap(sns.color_palette("Set2", 10)))
    plt.title(title)
    plt.xlabel("UMAP_1")
    plt.ylabel("UMAP_2")
    plt.minorticks_on()
    plt.savefig(output_path, dpi=300)

# Function to find optimal number of clusters using Elbow method
def find_optimal_clusters(data, k_range, output_path):
    sum_of_squared_distances = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=20, random_state=42)
        km.fit(data)
        sum_of_squared_distances.append(km.inertia_)

    plt.figure(figsize=(10, 7))
    plt.plot(k_range, sum_of_squared_distances, 'o-')
    plt.xlabel('k')
    plt.ylabel('Sum of squared distances')
    plt.title('Elbow Method For Optimal k')
    plt.xticks(np.arange(min(k_range), max(k_range)+1, 2))
    plt.grid()
    plt.savefig(output_path, dpi=300)

# Function to perform KMeans clustering
def perform_KMeans_clustering(data, n_clusters_list, output_directory):
    for n_clusters in n_clusters_list:
        model = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
        labels = model.fit_predict(data)
        plot_title = f'KMeans Clustering with k={n_clusters}'
        output_path = os.path.join(output_directory, f'KMeans_k{n_clusters}.png')
        plot_clustering(data, labels, plot_title, output_path)

# Function to perform Gaussian Mixture clustering
def perform_GaussianMixture_clustering(data, n_components_list, output_directory):
    for n_components in n_components_list:
        model = GaussianMixture(n_components=n_components, covariance_type='tied', tol=1e-5, random_state=42)
        labels = model.fit_predict(data)
        plot_title = f'Gaussian Mixture Clustering with n_components={n_components}'
        output_path = os.path.join(output_directory, f'GaussianMixture_n{n_components}.png')
        plot_clustering(data, labels, plot_title, output_path)

# Function to perform HDBSCAN clustering
def perform_HDBSCAN_clustering(data, min_cluster_size_list, output_directory):
    for min_cluster_size in min_cluster_size_list:
        model = HDBSCAN(min_cluster_size=min_cluster_size, gen_min_span_tree=True)
        labels = model.fit_predict(data)
        plot_title = f'HDBSCAN Clustering with min_cluster_size={min_cluster_size}'
        output_path = os.path.join(output_directory, f'HDBSCAN_minSize{min_cluster_size}.png')
        plot_clustering(data, labels, plot_title, output_path)

# Function to assign cluster names and visualize the final model
def visualize_final_model(data, reduced_data_UMAP, output_path):
    gmm = GaussianMixture(n_components=8, covariance_type='tied', tol=1e-5, random_state=42)
    gmm.fit(data)
    yhat = gmm.predict(data)

    # Set clusters for each cell and map to cell types
    data['Cluster'] = yhat
    cluster_names = {
        0: "CD4+ T cells", 1: "CD20+ B cells", 2: "CD20+ B cells",
        3: "CD8+ T cells", 4: "CD20+ B cells", 5: "CD14+ Monocytes",
        6: "CD4+ T cells", 7: "Other cells"
    }
    data['Population'] = data['Cluster'].map(cluster_names)
    data.drop(['Cluster'], axis=1, inplace=True)

    # Visualize the final clustering model
    plt.figure(figsize=(10, 7))
    unique_labels = data['Population'].unique()
    for label in unique_labels:
        row_ix = np.where(data['Population'] == label)
        plt.scatter(reduced_data_UMAP[row_ix, 0], reduced_data_UMAP[row_ix, 1], s=0.05, label=label)
    plt.title('Final clustering model')
    plt.xlabel("UMAP_1")
    plt.ylabel("UMAP_2")
    plt.minorticks_on()
    plt.legend(title='Population', loc='upper right', fontsize='small', markerscale=6)
    plt.savefig(output_path, dpi=300)

# Function to create a heatmap of cluster means
def create_cluster_heatmap(data, output_path):
    sampled_data = data.sample(frac=0.1)
    cluster_means = sampled_data.groupby('Population').mean()
    Z = linkage(cluster_means, 'ward')
    g = sns.clustermap(cluster_means, row_linkage=Z, cmap="Blues", col_cluster=False, yticklabels=True, figsize=(6, 6), vmin=0, vmax=1)
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45)
    plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)
    plt.savefig(output_path, dpi=300)

# Function to write final processed data to CSV
def write_processed_data_to_csv(data, file_path):
    data.to_csv(file_path, index=False)

# Main function
def main():
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <path_to_main_directory>")
        sys.exit(1)

    # Define directories
    main_directory = sys.argv[1]
    data_directory = os.path.join(main_directory, "Data")
    images_directory = os.path.join(main_directory, "Images")

    # Define file path
    fcs_file_path = os.path.join(data_directory, 'Tissue Samples_M LN.596613.fcs')
    cytometry_data = read_cytometry_data(fcs_file_path)

    # Preprocess data
    preprocessed_data = preprocess_data(cytometry_data)

    # Plot boxplots
    plot_boxplots(preprocessed_data, os.path.join(images_directory, 'boxplot_distribution_of_unprocessed_counts.png'), "Unprocessed Counts")

    # Identify and filter doublets
    pd_df_filt = identify_and_filter_doublets(preprocessed_data, images_directory)

    # Transform and scale data
    unique_percentiles = {'FSC-A': 99, 'SSC-A': 70}
    scaled_data = transform_and_scale_data(pd_df_filt, unique_percentiles)

    # Plot boxplots
    plot_boxplots(scaled_data, os.path.join(images_directory, 'boxplot_distribution_of_transformed_and_scaled_counts.png'), "Transformed and Scaled Counts")

    # Set row indices of scaled_data to match pd_df_filt
    scaled_data.index = pd_df_filt.index

    # Dimensionality reduction with UMAP
    reduced_data_UMAP = reduce_dimensionality_UMAP(scaled_data)
    plot_UMAP(reduced_data_UMAP, os.path.join(images_directory, 'UMAP.png'))

    # UMAP visualization colored by markers
    plot_UMAP_markers(reduced_data_UMAP, scaled_data, images_directory)

    # Find optimal number of clusters
    find_optimal_clusters(scaled_data, range(2, 20), os.path.join(images_directory, 'elbow_method.png'))

    # Perform clustering
    perform_KMeans_clustering(reduced_data_UMAP, [5, 6, 7, 8, 9, 10], images_directory)
    perform_GaussianMixture_clustering(reduced_data_UMAP, [4, 5, 6, 7, 8, 9], images_directory)
    perform_HDBSCAN_clustering(reduced_data_UMAP, [5, 10, 20, 50], images_directory)

    # Visualize the final clustering model
    visualize_final_model(scaled_data, reduced_data_UMAP, os.path.join(images_directory, 'Gaussian_Mixture_final_clusterization.png'))

    # Create a heatmap of cluster means
    create_cluster_heatmap(scaled_data, os.path.join(images_directory, 'heatmap_of_cluster_means.png'))

    # add cluster labels to the original dataframe
    pd_df_filt['Population'] = scaled_data["Population"]

    # Write the final processed data to a CSV file
    write_processed_data_to_csv(pd_df_filt, os.path.join(data_directory, 'LN_labeled_data.csv'))

if __name__ == "__main__":
    main()

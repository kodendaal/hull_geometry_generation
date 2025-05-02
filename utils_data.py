#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   database_metric_analysis.py
@Time    :   2024/08/27 09:24:17
@Author  :   Kirsten Odendaal 
@Version :   1.0
@Contact :   k.odendaal@marin.nl
@License :   (C)Copyright 2024-20XX, Kirsten Odendaal
@Desc    :   Script to assess the encoded distance metrics
'''

# general package
import numpy as np
import pandas as pd
import os

# metric package
from scipy.spatial.distance import directed_hausdorff

# machine learning projection package
import umap.umap_ as umap
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

# visualize package
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

# Render the visualizations in browsers
pio.renderers.default = "browser"

######################################################


def load_geometries_from_folder(folder_path):
    """
    Load and process encoded 3D geometries from text files in a specified directory.

    Args:
        folder_path (str): The path to the directory containing the encoded geometry text files. 
                           Each text file is expected to contain 3D coordinates (x, y, z) in a 
                           comma-separated format.

    Returns:
        list of np.ndarray: A list where each element is a NumPy array representing 
                            a valid 3D geometry loaded from a text file. Each array 
                            has shape (800, 3).
        np.ndarray: An array of integer labels corresponding to the geometries.

    Raises:
        OSError: If the specified folder path does not exist or is not accessible.
        ValueError: If a file's content cannot be loaded as a 2D array or if the 
                    array does not have exactly 3 columns (for x, y, z coordinates).
    """
    geometries = []
    labels = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            try:
                geometry_array = np.loadtxt(file_path, delimiter=',')
                geometry, _ = scale_xyz_coordinates(geometry_array)
                label = os.path.basename(file_path)[:3]
                # print(geometry.shape)
                
                # if geometry.shape == (800,3): # this should be the correct shape - some encodings not correct FDS series
                if (geometry.shape[1] == 3) and (geometry.shape == (800,3)):
                    geometries.append(geometry)
                    labels.append(label)
                else:
                    print(f"Skipping file {filename} due to incorrect shape: {geometry.shape}")
            
            except Exception as e:
                print(f"Error loading file {filename}: {e}")
     
    return geometries, labels


def hausdorff_distance(set1, set2):
    """
    Compute the Hausdorff distance between two 3D point sets.

    Args:
        set1 (np.ndarray): A NumPy array of shape (800, 3) representing the first set of 3D points.
        set2 (np.ndarray): A NumPy array of shape (800, 3) representing the second set of 3D points.

    Returns:
        float: The Hausdorff distance between the two point sets.
    """
    d1 = directed_hausdorff(set1, set2)[0]
    d2 = directed_hausdorff(set2, set1)[0]
    
    return max(d1, d2)


def scale_xyz_coordinates(input_array):
    """
    Scale the x-coordinate of an Nx3 numpy array to the range [-1, 0] and apply
    the same scale factor to the y and z coordinates. Scaling aligns with initial
    encoded sign convention (x: -, y: +, z: -)

    Args:
        input_array (np.ndarray): Nx3 numpy array where each row is [x, y, z].

    Returns:
        tuple: A tuple containing:
            - scaled_array (np.ndarray): The scaled Nx3 array.
            - scale_factor (float): The scale factor used for y and z.
    """
    # Extract x, y, z coordinates
    x = input_array[:, 0]
    y = input_array[:, 1]
    z = input_array[:, 2]

    # Calculate x_min and x_max
    x_min = x.min()
    x_max = x.max()

    # Scale x to range [-1, 1]
    x_scaled = 2 * (x - x_min) / (x_max - x_min) - 1

    # Calculate the scaling factor based on x's range
    scale_factor = 2 / (x_max - x_min)

    # Apply the same scaling factor to y and z
    y_scaled = (y - y.min()) * scale_factor
    z_scaled = (z - z.min()) * scale_factor

    # Combine the scaled x, y, and z into one array
    scaled_array = np.column_stack((x_scaled, y_scaled, z_scaled))

    return scaled_array, scale_factor


def calculate_distance_matrix(geometries):
    """
    Calculate the pairwise Hausdorff distance matrix for a list of geometries.

    Args:
        geometries (list of np.ndarray): A list of 3D geometries.

    Returns:
        np.ndarray: A symmetric distance matrix where each element [i, j] 
                    represents the Hausdorff distance between geometry i and geometry j.
    """
    num_geometries = len(geometries)
    distance_matrix = np.zeros((num_geometries, num_geometries))

    for i in range(num_geometries):
        for j in range(i + 1, num_geometries):
            distance_matrix[i, j] = hausdorff_distance(geometries[i], geometries[j])
            distance_matrix[j, i] = distance_matrix[i, j]
    
    return distance_matrix


def perform_umap(distance_matrix, n_components=2, random_state=42):
    """
    Perform UMAP for dimensionality reduction on a distance matrix.

    Args:
        distance_matrix (np.ndarray): A precomputed distance matrix.
        n_components (int, optional): The number of dimensions to reduce to. Defaults to 2.
        random_state (int, optional): Seed for random number generation. Defaults to 42.

    Returns:
        np.ndarray: The resulting UMAP embedding.
    """
    reducer = umap.UMAP(min_dist=0.6, n_neighbors=75, n_components=n_components, random_state=random_state)
    
    return reducer.fit_transform(distance_matrix)


def plot_umap_with_clustering(embedding, labels=None, random_state=42):
    """
    Plot the UMAP embedding with clustering.

    Args:
        embedding (np.ndarray): The UMAP embedding.
        labels (np.ndarray, optional): Cluster labels. Defaults to None.
        random_state (int, optional): Seed for random number generation. Defaults to 42.
    """
    if labels is None:
        kmeans = KMeans(n_clusters=3, random_state=random_state)
        labels = kmeans.fit_predict(embedding)
    
    # v_class = [ 'con', 'dsy', 'dtc', 'fds', 'jbc', 'kcs', 'kvl', 'my1', 'my2', 'my3', 'nss']
    v_class = ['MY3', 'MY1', 'MY2', 'DSY', 'FDS', 'NSS']
    class_labels = [v_class[label] for label in labels]

#    Define a color map for each class
    color_map = {
        'con': '#1f77b4',  # Blue
        'DSY': '#ff7f0e',  # Orange
        'dtc': '#2ca02c',  # Green
        'FDS': '#d62728',  # Red
        'jbc': '#9467bd',  # Purple
        'kcs': '#8c564b',  # Brown
        'kvl': '#e377c2',  # Pink
        'MY1': '#7f7f7f',  # Gray
        'MY2': '#bcbd22',  # Olive
        'MY3': '#17becf',  # Cyan
        'NSS': '#daa520'  # Goldenrod
    }

    if embedding.shape[1] > 2: 
        fig = px.scatter_3d(
            x=embedding[:, 0],
            y=embedding[:, 1],
            z=embedding[:, 2],
            color=class_labels,
            labels={'x': 'UMAP 1', 'y': 'UMAP 2', 'z': 'UMAP 3'},
            # text=[f"{v_class[label]}-{i}" for i, label in enumerate(labels)],
            title='UMAP Projection of 3D Geometries'
        )
        fig.update_traces(marker=dict(size=5, opacity=0.5))
        fig.show()
        
    else:
        plt.figure(figsize=(8, 6))
        # Plot each class separately with its corresponding color
        for i, class_name in enumerate(v_class):
            # Get indices of points belonging to the current class
            class_indices = [idx for idx, label in enumerate(labels) if v_class[label] == class_name]
            
            # Scatter the points for the current class
            plt.scatter(
                embedding[class_indices, 0], embedding[class_indices, 1],
                c=color_map[class_name], label=class_name, s=10, alpha=0.5
            )

        # Add title and labels
        plt.title('UMAP Projection of 3D Geometries')
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')

        # Add a legend to show the class colors
        plt.legend(title='Classes')

        # Show the plot
        plt.show()


def perform_tsne(distance_matrix, perplexity=3, n_components=2, random_state=42):
    """
    Perform t-SNE on a precomputed distance matrix and return the 2D embedding.

    Args:
        distance_matrix (np.ndarray): A precomputed distance matrix.
        perplexity (int, optional): Perplexity parameter for t-SNE. Defaults to 3.
        n_components (int, optional): The number of dimensions to reduce to. Defaults to 2.
        random_state (int, optional): Seed for random number generation. Defaults to 42.

    Returns:
        np.ndarray: The resulting t-SNE embedding.
    """
    tsne = TSNE(n_components=n_components, perplexity=perplexity, metric='precomputed', init='random', random_state=random_state)
    
    return tsne.fit_transform(distance_matrix)


def plot_tsne_with_clustering(embedding, labels=None, random_state=42):
    """
    Plot the t-SNE embedding with clustering.

    Args:
        embedding (np.ndarray): The t-SNE embedding.
        labels (np.ndarray, optional): Cluster labels. Defaults to None.
        random_state (int, optional): Seed for random number generation. Defaults to 42.
    """
    if labels is None:
        kmeans = KMeans(n_clusters=3, random_state=random_state)
        labels = kmeans.fit_predict(embedding)

    # v_class = [ 'con', 'dsy', 'dtc', 'fds', 'jbc', 'kcs', 'kvl', 'my1', 'my2', 'my3', 'nss']
    v_class = ['MY3', 'MY1', 'MY2', 'DSY', 'FDS', 'NSS']
    class_labels = [v_class[label] for label in labels]

#    Define a color map for each class
    color_map = {
        'con': '#1f77b4',  # Blue
        'DSY': '#ff7f0e',  # Orange
        'dtc': '#2ca02c',  # Green
        'FDS': '#d62728',  # Red
        'jbc': '#9467bd',  # Purple
        'kcs': '#8c564b',  # Brown
        'kvl': '#e377c2',  # Pink
        'MY1': '#7f7f7f',  # Gray
        'MY2': '#bcbd22',  # Olive
        'MY3': '#17becf',  # Cyan
        'NSS': '#daa520'  # Goldenrod
    }
    
    if embedding.shape[1] > 2:
        fig = px.scatter_3d(
            x=embedding[:, 0],
            y=embedding[:, 1],
            z=embedding[:, 2],
            color=class_labels,
            labels={'x': 't-SNE 1', 'y': 't-SNE 2', 'z': 't-SNE 3'},
            # text=[f"{v_class[label]}-{i}" for i, label in enumerate(labels)],
            title='t-SNE Projection of 3D Geometries'
        )
        fig.update_traces(marker=dict(size=5, opacity=0.8))
        fig.show()
    else:
        # Create figure
        plt.figure(figsize=(8, 6))
        # Plot each class separately with its corresponding color
        for i, class_name in enumerate(v_class):
            # Get indices of points belonging to the current class
            class_indices = [idx for idx, label in enumerate(labels) if v_class[label] == class_name]
            
            # Scatter the points for the current class
            plt.scatter(
                embedding[class_indices, 0], embedding[class_indices, 1],
                c=color_map[class_name], label=class_name, s=10, alpha=0.5
            )

        # Add title and labels
        plt.title('t-SNE Projection of 3D Geometries')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')

        # Add a legend to show the class colors
        plt.legend(title='Classes')

        # Show the plot
        plt.show()
    

        
###########################################################

def main():
    # Specify the folder where your encoded geometry files are located
    folder_path = r'P:\70xxx\701xx\7018x\70180\RD\301 - HullGAN\Encode'

    # Load the geometries
    all_geoms = []
    all_labels = []
    for fi in os.listdir(folder_path):
        if fi.endswith('.gh'):
            continue
        else:
            print(fi)
            encode_path = os.path.join(folder_path, fi)
            geometries, labels = load_geometries_from_folder(encode_path)
            all_geoms.extend(geometries)  # Append the geometries
            all_labels.extend(labels)  # Append the labels if available

    # Check if geometries were loaded successfully
    if not all_geoms:
        print("No valid geometries found in the specified folder.")
        return

    print(f"Loaded {len(all_geoms)} geometries from the folder.")

    # Calculate the pairwise Hausdorff distance matrix
    distance_matrix = calculate_distance_matrix(geometries)

    # Perform UMAP for dimensionality reduction
    labels, _ = pd.factorize(all_labels)
    embedding = perform_umap(distance_matrix, n_components=2)
    plot_umap_with_clustering(embedding, labels=labels)

    # # # # Perform t-SNE for dimensionality reduction
    # embedding = perform_tsne(distance_matrix, perplexity=25, n_components=2, random_state=42)
    # plot_tsne_with_clustering(embedding, labels=labels)


if __name__ == "__main__":
    # main()
    # Specify the folder where your encoded geometry files are located
    folder_path = r'P:\70xxx\701xx\7018x\70180\RD\301 - HullGAN\Encode'

    # Load the geometries
    all_geoms = []
    all_labels = []
    for fi in os.listdir(folder_path):
        if fi.endswith('.gh'):
            continue
        else:
            print(fi)
            encode_path = os.path.join(folder_path, fi)
            geometries, labels = load_geometries_from_folder(encode_path)
            all_geoms.extend(geometries)  # Append the geometries
            all_labels.extend(labels)  # Append the labels if available

    geos = np.transpose(all_geoms, (0,2,1)).reshape(len(all_geoms), 3, 40, 20)
    
    # Calculate the pairwise Hausdorff distance matrix
    distance_matrix = calculate_distance_matrix(all_geoms)
    labels, _ = pd.factorize(all_labels)
    
    # Perform UMAP for dimensionality reduction
    # embedding = perform_umap(distance_matrix, n_components=3)
    # plot_umap_with_clustering(embedding, labels=labels)
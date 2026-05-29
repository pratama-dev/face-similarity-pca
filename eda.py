import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from collections import Counter

def plot_class_distribution(labels):
    """Bar chart distribusi jumlah gambar per orang."""
    counts = Counter(labels)
    df = pd.DataFrame(counts.items(), columns=['Person', 'Count'])
    df = df.sort_values('Count', ascending=False)
    
    fig = px.bar(df, x='Person', y='Count',
                 title='Distribusi Jumlah Gambar per Orang',
                 color='Count', color_continuous_scale='blues')
    return fig

def plot_explained_variance(pca):
    """Scree plot explained variance ratio."""
    cumsum = np.cumsum(pca.explained_variance_ratio_) * 100
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=pca.explained_variance_ratio_ * 100,
        mode='lines+markers', name='Per Komponen',
        line=dict(color='royalblue')
    ))
    fig.add_trace(go.Scatter(
        y=cumsum, mode='lines+markers',
        name='Kumulatif', line=dict(color='red', dash='dash')
    ))
    fig.add_hline(y=95, line_dash="dot", line_color="green",
                  annotation_text="95% variance")
    fig.update_layout(
        title='Explained Variance Ratio PCA',
        xaxis_title='Komponen PCA',
        yaxis_title='Variance (%)'
    )
    return fig

def plot_eigenfaces(pca, img_size=(100, 100), n=10):
    """Tampilkan eigenfaces pertama."""
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i, ax in enumerate(axes.flat):
        if i < n and i < len(pca.components_):
            eigenface = pca.components_[i].reshape(img_size)
            ax.imshow(eigenface, cmap='gray')
            ax.set_title(f'Eigenface {i+1}')
        ax.axis('off')
    plt.suptitle('Top 10 Eigenfaces', fontsize=14)
    plt.tight_layout()
    return fig

def plot_pca_scatter(X_pca, labels):
    """Scatter plot 2D komponen pertama PCA."""
    df = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'Person': labels
    })
    fig = px.scatter(df, x='PC1', y='PC2', color='Person',
                     title='Distribusi Wajah di Ruang PCA (2 Komponen Pertama)',
                     hover_data=['Person'])
    return fig

def plot_mean_face(X, img_size=(100, 100)):
    """Tampilkan mean face."""
    mean_face = X.mean(axis=0).reshape(img_size)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(mean_face, cmap='gray')
    ax.set_title('Mean Face (Wajah Rata-rata)')
    ax.axis('off')
    return fig
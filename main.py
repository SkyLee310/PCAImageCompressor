import numpy as np
from numpy import linalg as LA
from PIL import Image
import os
import matplotlib.pyplot as plt
import time

def preprocess_channel(channel):
    mean = np.mean(channel, axis=0)
    centered_data = channel - mean
    return centered_data, mean

def pca_svd_method(A_centered):
    start_time = time.time()

    U, Sigma, VT = np.linalg.svd(A_centered, full_matrices=False)

    execution_time = time.time() - start_time
    return U, Sigma, VT, execution_time


def pca_eigen_method(A_centered):
    start_time = time.time()

    cov_matrix = np.cov(A_centered, rowvar=False)

    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    execution_time = time.time() - start_time
    return eigenvalues, eigenvectors, execution_time


def find_k_for_95_variance(singular_values_or_eigenvalues, is_singular=True):

    if is_singular:
        variance = singular_values_or_eigenvalues ** 2
    else:
        variance = singular_values_or_eigenvalues

    total_variance = np.sum(variance)
    cumulative_variance = np.cumsum(variance) / total_variance

    k = np.argmax(cumulative_variance >= 0.95) + 1
    return k, cumulative_variance


def compress_image_comparison(image_path):
    img = Image.open(image_path).convert('RGB')
    original_data = np.array(img, dtype=float)
    h, w, c = original_data.shape
    print(f"Original Image: {w}x{h}, File Size: {os.path.getsize(image_path) / 1024:.2f} KB")
    reconstructed_svd = np.zeros_like(original_data)
    reconstructed_eigen = np.zeros_like(original_data)
    svd_total_time = 0
    eigen_total_time = 0
    k_95 = 0
    for i in range(3):
        channel = original_data[:, :, i]

        centered_channel, mean = preprocess_channel(channel)

        U, Sigma, VT, t_svd = pca_svd_method(centered_channel)
        svd_total_time += t_svd

        eigenvals, eigenvecs, t_eigen = pca_eigen_method(centered_channel)
        eigen_total_time += t_eigen

        k, _ = find_k_for_95_variance(Sigma, is_singular=True)
        k_95 = max(k_95, k)

        re_svd = U[:, :k] @ np.diag(Sigma[:k]) @ VT[:k, :]
        reconstructed_svd[:, :, i] = re_svd + mean

        V_k = eigenvecs[:, :k]
        re_eigen = (centered_channel @ V_k) @ V_k.T
        reconstructed_eigen[:, :, i] = re_eigen + mean

    reconstructed_svd = np.clip(reconstructed_svd, 0, 255).astype('uint8')
    reconstructed_eigen = np.clip(reconstructed_eigen, 0, 255).astype('uint8')

    print("-" * 30)
    print(f"Performance Comparison (k={k_95} for 95% variance):")
    print(f"Total SVD Time: {svd_total_time:.4f}s")
    print(f"Total Eigen Time: {eigen_total_time:.4f}s")
    print(f"Efficiency Winner: {'SVD' if svd_total_time < eigen_total_time else 'Eigen Decomposition'}")
    print("-" * 30)

    fig, axes = plt.subplots(1, 3, figsize=(20, 8))

    axes[0].imshow(original_data.astype('uint8'))
    axes[0].set_title(f"Original Image\n({w}x{h})", fontsize=14, fontweight='bold')
    axes[0].axis('off')

    #SVD
    axes[1].imshow(reconstructed_svd)
    axes[1].set_title(f"SVD Method\nk={k_95} (95% Var)", fontsize=14, fontweight='bold')
    axes[1].set_xlabel(f"Time: {svd_total_time:.4f}s", fontsize=12)
    axes[1].axis('off')

    #Eigen
    axes[2].imshow(reconstructed_eigen)
    axes[2].set_title(f"Eigen Method\nk={k_95} (95% Var)", fontsize=14, fontweight='bold')
    axes[2].set_xlabel(f"Time: {eigen_total_time:.4f}s", fontsize=12)
    axes[2].axis('off')

    winner = 'SVD' if svd_total_time < eigen_total_time else 'Eigen Decomposition'
    summary_text = (
        f"PCA Image Compression Analysis\n"
        f"--------------------------------------------------\n"
        f"Target Variance Retention: 95%  |  Optimal k: {k_95}\n"
        f"SVD Execution Time: {svd_total_time:.4f}s\n"
        f"Eigen Execution Time: {eigen_total_time:.4f}s\n"
        f"Efficiency Winner: {winner}"
    )

    plt.figtext(0.5, 0.02, summary_text, ha="center", fontsize=12,
                bbox={"facecolor": "orange", "alpha": 0.1, "pad": 10}, fontfamily='monospace')

    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    plt.suptitle(f"Matrix Decomposition Comparison: {filename}", fontsize=18, y=0.98)
    plt.show()

try:
    filename='tokyo.jpeg'
    compress_image_comparison(filename)
except KeyboardInterrupt:
    print("Session Terminated Successfully")
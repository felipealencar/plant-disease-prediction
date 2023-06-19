
import numpy as np
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy.random import random
from scipy.linalg import sqrtm
import tensorflow as tf
from scipy.linalg import sqrtm
import cv2
from sklearn.metrics import jaccard_score


def preprocess_array(array):
    # Normalize array to range [-1, 1]
    normalized_array = (array - 0.5) * 2.0
    # Add third channel by replicating the second channel
    extra_channel = np.zeros_like(array[..., :1])
    expanded_array = np.concatenate([normalized_array, extra_channel], axis=-1)

    return expanded_array


def calculate_fid(real_images, generated_images):
    # Load pre-trained InceptionV3 model
    inception_model = tf.keras.applications.InceptionV3(include_top=False, pooling='avg')

    # Generate feature vectors for both arrays
    features1 = inception_model.predict(real_images)
    features2 = inception_model.predict(generated_images)

    # Compute mean and covariance matrices
    mean1 = np.mean(features1, axis=0)
    mean2 = np.mean(features2, axis=0)
    cov1 = np.cov(features1, rowvar=False)
    cov2 = np.cov(features2, rowvar=False)

    # Add a small positive constant to the covariance matrices
    eps = 1e-6
    cov1 += eps * np.eye(cov1.shape[0])
    cov2 += eps * np.eye(cov2.shape[0])

    # Calculate square root of the product of covariances (covariance square root)
    diff = mean1 - mean2
    cov_sqrt = sqrtm(cov1.dot(cov2)).real

    # Calculate FID
    fid = np.sum(diff ** 2) + np.trace(cov1 + cov2 - 2 * cov_sqrt)

    # Calculate FID error
    num_samples = real_images.shape[0] + generated_images.shape[0]
    fid_error = fid / num_samples

    return fid, fid_error


def scores(real_images, generated_images):
    num_images = real_images.shape[0]

    # Calculate FID
    fid, fid_error = calculate_fid(real_images, generated_images)
    print('Bhattacharyya started...')
    # Calculate Bhattacharyya distance and error
    bhattacharyya, bhattacharyya_error = calculate_bhattacharyya(real_images, generated_images)
    print('Bhattacharyya done.')

    # Calculate Chi-Square distance and error
    print('Chi-square started...')
    chi_square, chi_square_error = calculate_chi_square(real_images, generated_images)
    print('Chi-square done.')

    print('Correlation started...')
    # Calculate Correlation coefficient
    correlation, correlation_error = calculate_correlation(real_images, generated_images)
    print('Correlation done.')

    # Calculate Intersection and error
    intersection, intersection_error = calculate_intersection(real_images, generated_images)

    print('Done!')
    return fid, fid_error, bhattacharyya, bhattacharyya_error, chi_square, chi_square_error, correlation, correlation_error, intersection, intersection_error


def calculate_bhattacharyya(real_images, generated_images, num_samples=1000):
    hist_real, _ = np.histogram(real_images.flatten(), bins=256, range=(0, 1))
    hist_generated, _ = np.histogram(generated_images.flatten(), bins=256, range=(0, 1))
    total_samples = real_images.size + generated_images.size

    bhattacharyya_values = []
    for _ in range(num_samples):
        sampled_real = np.random.choice(real_images.flatten(), size=real_images.size, replace=True)
        sampled_generated = np.random.choice(generated_images.flatten(), size=generated_images.size, replace=True)
        sampled_hist_real, _ = np.histogram(sampled_real, bins=256, range=(0, 1))
        sampled_hist_generated, _ = np.histogram(sampled_generated, bins=256, range=(0, 1))

        b_coeff = np.sum(np.sqrt(sampled_hist_real * sampled_hist_generated)) / np.sqrt(hist_real.sum() * hist_generated.sum())
        bhattacharyya_values.append(b_coeff)

    bhattacharyya = np.mean(bhattacharyya_values)
    margin_error = 1.96 * np.std(bhattacharyya_values)

    return bhattacharyya, margin_error


def calculate_bootstrap_metric(real_images, generated_images, metric_function, num_bootstrap=1000):
    num_samples = real_images.shape[0] + generated_images.shape[0]
    bootstrap_metrics = []

    for _ in range(num_bootstrap):
        sampled_real_indices = np.random.choice(
            range(real_images.shape[0]), size=num_samples, replace=True)
        sampled_generated_indices = np.random.choice(
            range(generated_images.shape[0]), size=num_samples, replace=True)
        sampled_real = real_images[sampled_real_indices]
        sampled_generated = generated_images[sampled_generated_indices]
        metric = metric_function(sampled_real, sampled_generated)
        bootstrap_metrics.append(metric)

    metric_mean = np.mean(bootstrap_metrics)
    metric_error_margin = np.percentile(bootstrap_metrics, [2.5, 97.5])  # 95% confidence interval

    return metric_mean, metric_error_margin


def calculate_chi_square(real_images, generated_images, num_samples=1000):
    observed_real, _ = np.histogram(real_images.flatten(), bins=256, range=(0, 1))
    observed_generated, _ = np.histogram(generated_images.flatten(), bins=256, range=(0, 1))
    expected = (observed_real + observed_generated) / 2.0

    chi_square_values = []
    for _ in range(num_samples):
        sampled_real = np.random.choice(real_images.flatten(), size=real_images.size, replace=True)
        sampled_generated = np.random.choice(generated_images.flatten(), size=generated_images.size, replace=True)
        sampled_observed_real, _ = np.histogram(sampled_real, bins=256, range=(0, 1))
        sampled_observed_generated, _ = np.histogram(sampled_generated, bins=256, range=(0, 1))
        sampled_expected = (sampled_observed_real + sampled_observed_generated) / 2.0
        sampled_chi_square = np.sum((sampled_observed_generated - sampled_expected) ** 2 / (sampled_expected + 1e-10))
        chi_square_values.append(sampled_chi_square)

    chi_square = np.mean(chi_square_values)
    margin_error = 1.96 * np.std(chi_square_values)

    return chi_square, margin_error


def calculate_correlation(real_images, generated_images, num_samples=1000):
    correlation_values = []
    for _ in range(num_samples):
        sampled_real = np.random.choice(real_images.flatten(), size=real_images.size, replace=True)
        sampled_generated = np.random.choice(generated_images.flatten(), size=generated_images.size, replace=True)
        sampled_correlation = np.corrcoef(sampled_real, sampled_generated)[0, 1]
        correlation_values.append(sampled_correlation)

    correlation = np.mean(correlation_values)
    margin_error = 1.96 * np.std(correlation_values)

    return correlation, margin_error


def calculate_intersection(real_images, generated_images, num_samples=1000):
    real_binary = np.where(real_images > 0.5, 1, 0)
    generated_binary = np.where(generated_images > 0.5, 1, 0)

    intersection_values = []
    for _ in range(num_samples):
        sampled_real = np.random.choice(real_binary.flatten(), size=real_binary.size, replace=True)
        sampled_generated = np.random.choice(generated_binary.flatten(), size=generated_binary.size, replace=True)
        sampled_intersection = jaccard_score(sampled_real, sampled_generated)
        intersection_values.append(sampled_intersection)

    intersection = np.mean(intersection_values)
    margin_error = 1.96 * np.std(intersection_values)

    return intersection, margin_error

# TODO: ADD THE Structural Similarity Index (SSIM).
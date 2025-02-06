import os
import numpy as np
import pickle
import tenseal as ts
from tqdm import tqdm
import logging
from joblib import Parallel, delayed
import pandas as pd
import pandas as pd
import numpy as np
import sys
from deepface import DeepFace
import kaggle
import time
# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)
import kagglehub




def measure_encrypted_matrix_size(encrypted_matrix):
    """
    Measures the approximate memory size of an encrypted similarity matrix in bytes.

    :param encrypted_matrix: List of encrypted similarity scores.
    :return: Total size in bytes.
    """
    total_size = 0

    for row in encrypted_matrix:
        for element in row:
            total_size += sys.getsizeof(element)  # Approximate memory footprint

    size_mb = total_size / (1024 * 1024)  # Convert to MB
    print(f"Estimated Encrypted Matrix Size: {size_mb:.2f} MB")
    return size_mb


def compare_top10(cleartext_file="top10.csv", decrypted_file="top10_dec.csv"):
    """
    Compares the homomorphic (decrypted) top-10 closest templates vs. cleartext top-10.
    
    Measures the percentage of test samples where the i-th most similar template
    is the same in both files.

    :param cleartext_file: Path to the cleartext top-10 indices file.
    :param decrypted_file: Path to the decrypted top-10 indices file.
    """
    # Load both files
    cleartext_top10 = pd.read_csv(cleartext_file).values
    decrypted_top10 = pd.read_csv(decrypted_file).values

    # Number of test samples
    num_samples = cleartext_top10.shape[0]
    num_ranks = cleartext_top10.shape[1]  # 10 columns (Rank_1 to Rank_10)

    # Compute percentage of matching indices for each rank
    matching_percentages = []
    for i in range(num_ranks):
        matches = np.sum(cleartext_top10[:, i] == decrypted_top10[:, i])
        percentage_match = (matches / num_samples) * 100
        matching_percentages.append(percentage_match)

    # Print results
    print(f"Top-10 Comparison Results:")
    for i, match in enumerate(matching_percentages):
        print(f"Rank {i+1}: {match:.2f}% matches")

    return matching_percentages



def compare_scores(cleartext_file="scores.csv", decrypted_file="scores_dec.csv"):
    """
    Compares the homomorphic (decrypted) scores vs. the cleartext scores.
    
    Computes the average, standard deviation, max, and min absolute difference 
    between each (i, j) value in the two matrices.

    :param cleartext_file: Path to the cleartext scores file.
    :param decrypted_file: Path to the decrypted scores file.
    """
    # Load both matrices
    cleartext_scores = pd.read_csv(cleartext_file).values
    decrypted_scores = pd.read_csv(decrypted_file).values

    # Ensure both matrices have the same shape
    if cleartext_scores.shape != decrypted_scores.shape:
        raise ValueError(f"Matrix shape mismatch: {cleartext_scores.shape} vs {decrypted_scores.shape}")

    # Compute absolute differences
    differences = np.abs(cleartext_scores - decrypted_scores)

    # Compute required statistics
    avg_diff = np.mean(differences)
    std_diff = np.std(differences)
    max_diff = np.max(differences)
    min_diff = np.min(differences)

    # Print results
    print("Score Comparison Results:")
    print(f"Average Absolute Difference: {avg_diff:.6f}")
    print(f"Standard Deviation: {std_diff:.6f}")
    print(f"Maximum Absolute Difference: {max_diff:.6f}")
    print(f"Minimum Absolute Difference: {min_diff:.6f}")

    return avg_diff, std_diff, max_diff, min_diff





def compute_cleartext_top10_indices(scores_file="scores.csv", output_file="top10.csv"):
    """
    Computes the top-10 closest templates for each test sample from the similarity scores.

    :param scores_file: CSV file containing squared Euclidean distances.
    :param output_file: Name of the CSV file to save the top-10 indices.
    """
    logger.info(f"Loading scores from {scores_file}...")

    # Load squared Euclidean distances
    df = pd.read_csv(scores_file)

    # Compute top-10 indices (smallest distances = most similar)
    top_10_indices = np.argsort(df.values, axis=1)[:, :10]

    # Save to CSV
    df_top10 = pd.DataFrame(top_10_indices)
    df_top10.to_csv(output_file, index=False, header=[f"Rank_{i+1}" for i in range(10)])

    logger.info(f"Top-10 closest template indices saved to {output_file}")

def compute_cleartext_distances(test_embeddings, train_embeddings, output_file="scores.csv"):
    """
    Computes squared Euclidean distances between each test embedding and all train embeddings.

    :param test_embeddings: np.array of shape (n, d), test set embeddings
    :param train_embeddings: np.array of shape (m, d), training set embeddings
    :param output_file: Name of the CSV file to save similarity scores
    """
    logger.info("Computing cleartext similarity scores...")
    start=time.time()
    # Compute squared Euclidean distance: ||x - y||^2
    n = test_embeddings.shape[0]
    m = train_embeddings.shape[0]
    distances = np.zeros((n, m))

    for i in range(n):
        # Compute distances between test[i] and all training samples
        distances[i] = np.sum((train_embeddings - test_embeddings[i])**2, axis=1)
    end=time.time()
    logger.info(f"Runtime to compute similarity over cleartext: {end-start:.4f} seconds")
    # Save to CSV
    df = pd.DataFrame(distances)
    df.to_csv(output_file, index=False)

    logger.info(f"Cleartext similarity scores saved to {output_file}")

def save_decrypted_scores(encrypted_distance_matrix, original_shape, output_file="scores_dec.csv"):
    """
    Decrypts the encrypted squared Euclidean distance matrix, unpacks it, and saves it to a CSV file.

    :param encrypted_distance_matrix: List of encrypted distance values.
    :param original_shape: Tuple (rows, cols) of the original matrix before packing.
    :param output_file: Name of the CSV file to save the decrypted matrix.
    """
    print(encrypted_distance_matrix.shape)
    logger.info("Decrypting and unpacking distance matrix...")
    start=time.time()
    decrypted_flat = []

    # Decrypt each element from the encrypted matrix
    for row in encrypted_distance_matrix:
        for distance in row:
            decrypted_values = distance.decrypt()  # Decrypt returns a packed list of values
            decrypted_flat.extend(decrypted_values)  # Flatten them into a single list

    # Convert to numpy array
    decrypted_flat = np.array(decrypted_flat)

    # Trim padding: Ensure we only keep the first (original_shape[0] * original_shape[1]) elements
    expected_size = original_shape[0] * original_shape[1]
    if len(decrypted_flat) > expected_size:
        logger.warning(f"Trimming {len(decrypted_flat) - expected_size} extra decrypted values due to padding.")
        decrypted_flat = decrypted_flat[:expected_size]

    # Reshape back to original matrix dimensions
    decrypted_matrix = decrypted_flat.reshape(original_shape)
    end=time.time()
    logger.info(f"Runtime to compute similarity over ciphertext: {end-start:.4f} seconds")
    # Save to CSV
    df = pd.DataFrame(decrypted_matrix)
    df.to_csv(output_file, index=False)

    logger.info(f"Decrypted distance matrix saved to {output_file}")




def save_top10_indices(scores_file="scores_dec.csv", output_file="top10_dec.csv"):
    """
    Reads the decrypted squared Euclidean distance matrix, computes the top-10 closest
    template indices for each sample, and saves them to a CSV file.

    :param scores_file: CSV file containing the decrypted distance matrix.
    :param output_file: Name of the CSV file to save the top-10 indices.
    """
    logger.info(f"Loading scores from {scores_file}...")

    # Load decrypted scores
    df = pd.read_csv(scores_file)

    # Compute top-10 indices (smallest distances mean closer matches)
    top_10_indices = np.argsort(df.values, axis=1)[:, :10]  # Get indices of the 10 smallest values

    # Save to CSV
    df_top10 = pd.DataFrame(top_10_indices)
    df_top10.to_csv(output_file, index=False, header=[f"Rank_{i+1}" for i in range(10)])

    logger.info(f"Top-10 closest template indices saved to {output_file}")

def encrypted_squared_euclidean_distance(encrypted_vec1, encrypted_vec2):
    """
    Computes squared Euclidean distance between two encrypted CKKS vectors.
    """
    diff = encrypted_vec1 - encrypted_vec2
    squared_diff = diff * diff  # Element-wise square
    return squared_diff.sum()  # Sum to get final distance

# Load embeddings
def load_embeddings(embeddings_file,lfw_path,model_name,calc=False):
    if os.path.exists(embeddings_file) and calc==False:
        print("Loading saved embeddings...\n")
        with open(embeddings_file, "rb") as f:
            return pickle.load(f)
    else:
        # Download latest version
        # Define the directory where you want to save the dataset
        custom_directory = "datasets/lfw"

        # Ensure the directory exists
        os.makedirs(custom_directory, exist_ok=True)

        # Download the dataset
        kaggle.api.dataset_download_files("jessicali9530/lfw-dataset", path=custom_directory, unzip=True)
        
        print("Calculating embeddings...")
        start=time.time()
        embeddings = {}
        for identity in tqdm(os.listdir(lfw_path), desc="Processing identities"):
            identity_path = os.path.join(lfw_path, identity)
            if os.path.isdir(identity_path):
                for image_name in os.listdir(identity_path):
                    image_path = os.path.join(identity_path, image_name)
                    embedding = DeepFace.represent(
                        img_path=image_path,
                        model_name=model_name,
                        enforce_detection=False
                    )
                    embeddings[image_path] = embedding
        end=time.time()
        logger.info(f"Runtime to extract embeddings with {model_name}: {end-start:.4f} seconds")
        with open(embeddings_file, "wb") as f:
            pickle.dump(embeddings, f)
        return embeddings
    
# Extract numeric embeddings
def extract_numeric_embeddings(embeddings):
    numeric_embeddings = [item[0]["embedding"] for item in embeddings.values() if isinstance(item, list) and len(item) > 0 and "embedding" in item[0]]
    logger.info("Numeric embeddings extracted.")
    return np.array(numeric_embeddings)

# Split embeddings into train and test
def split_embeddings(embeddings, train_ratio):
    num_train = int(len(embeddings) * train_ratio)
    train_embeddings = embeddings[:num_train]
    test_embeddings = embeddings[num_train:]
    logger.info("Embeddings split into train and test sets.")
    return train_embeddings, test_embeddings

# Apply low precision
def apply_low_precision(embeddings, bit_size):
    dtype_map = {4: np.int8, 8: np.uint8, 16: np.float16, 32: np.float32}
    return embeddings.astype(dtype_map.get(bit_size, embeddings.dtype))

# Initialize CKKS context
def initialize_ckks_context(poly_modulus_degree, scale, coeff_mod_bit_sizes):
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=poly_modulus_degree,
        coeff_mod_bit_sizes=coeff_mod_bit_sizes
    )
    context.global_scale = scale
    context.generate_galois_keys()
    context.generate_relin_keys()
    return context

# Encrypt embeddings
def encrypt_embeddings(embeddings, context, packing_size):
    encrypted_embeddings = []
    for i in range(0, len(embeddings), packing_size):
        chunk = embeddings[i:i + packing_size]
        if len(chunk) < packing_size:
            chunk = np.vstack([chunk, np.zeros((packing_size - len(chunk), embeddings.shape[1]))])
        encrypted_embeddings.append(ts.ckks_vector(context, chunk.flatten().tolist()))
    logger.info("Embeddings encrypted.")
    return encrypted_embeddings

# Batch data
def batch_data(data, batch_size):
    batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
    logger.info("Data batched into %d batches of size %d.", len(batches), batch_size)
    return batches

# Compute squared Euclidean distance for a batch
def process_batch(test_batch, train_embeddings):
    batch_distances = []
    for test_vec in tqdm(test_batch, desc="Processing test vectors", ncols=100):
        row_distances = []
        for train_vec in train_embeddings:
            distance = encrypted_squared_euclidean_distance(test_vec, train_vec)
            row_distances.append(distance)  # Decrypt for final result
        batch_distances.append(row_distances)
    return batch_distances

# Compute Squared Euclidean Distance Matrix
def compute_squared_euclidean_distance_matrix(test_embeddings, train_embeddings, batch_size, parallel_jobs):
    distance_matrix = []

    # Batch the test embeddings
    test_batches = batch_data(test_embeddings, batch_size)

    # Use Parallel to process batches in parallel
    results = Parallel(n_jobs=parallel_jobs, backend="threading")(
        delayed(process_batch)(batch, train_embeddings)
        for batch in tqdm(test_batches, desc="Processing test batches", ncols=100)
    )

    for batch_result in results:
        distance_matrix.extend(batch_result)

    logger.info("Squared Euclidean distance computation completed.")
    return np.array(distance_matrix)

# Main Function with All Adjustable Parameters
def biometric_identification(
    file_path,
    train_embeddings,
    test_embeddings,
    train_ratio=0.8,
    precision=16,
    poly_modulus_degree=8192,
    scale=2**40,
    coeff_mod_bit_sizes=[60, 40, 40, 60],
    packing_size=32,
    batch_size=16,
    parallel_jobs=-1,
):
    """
    Perform biometric identification using encrypted squared Euclidean distance.
    
    Parameters:
    - file_path (str): Path to embeddings file.
    - train_ratio (float): Ratio of data used for training.
    - precision (int): Bit precision (4, 8, 16, 32).
    - poly_modulus_degree (int): Polynomial modulus degree for CKKS.
    - scale (int): Global scale for CKKS.
    - coeff_mod_bit_sizes (list): Bit sizes for coefficient modulus.
    - packing_size (int): Number of elements packed per encryption.
    - batch_size (int): Batch size for processing test embeddings.
    - parallel_jobs (int): Number of parallel jobs (-1 uses all available cores).

    Returns:
    - np.array: Squared Euclidean distance matrix.
    """
    logger.info("Starting biometric identification.")
    import gc
    # file_path="../partA/embeddings_Facenet.pkl" # Change this to your dataset location
    # model_name = "Facenet"
    # # Step 1: Load Embeddings
    # embeddings = load_embeddings(file_path,lfw_path,model_name,calc=False)

    # # Step 2: Extract Numeric Embeddings
    # numeric_embeddings = extract_numeric_embeddings(embeddings)

    # # Step 3: Split Train and Test
    # train_embeddings, test_embeddings = split_embeddings(numeric_embeddings, train_ratio)

    # Step 4: Apply Low Precision
    train_embeddings = apply_low_precision(train_embeddings, precision)
    test_embeddings = apply_low_precision(test_embeddings, precision)

    # Step 5: Initialize CKKS Context
    context = initialize_ckks_context(poly_modulus_degree, scale, coeff_mod_bit_sizes)
    start=time.time()
    # Step 6: Encrypt Data
    encrypted_train_embeddings = encrypt_embeddings(train_embeddings, context, packing_size)
    encrypted_test_embeddings = encrypt_embeddings(test_embeddings, context, packing_size)
    end=time.time()
    logger.info(f"Runtime to encrypt embeddings : {end-start:.4f} seconds")
    del train_embeddings, test_embeddings, context
    gc.collect()
    start=time.time()
    # Step 7: Compute Squared Euclidean Distance
    distance_matrix = compute_squared_euclidean_distance_matrix(
        encrypted_test_embeddings, encrypted_train_embeddings, batch_size, parallel_jobs
    )
    end=time.time()
    logger.info(f"Runtime to compute similarity over ciphertext: {end-start:.4f} seconds")
    logger.info("Biometric identification completed.")
    return distance_matrix

# Example Usage
if __name__ == "__main__":
    file_path="../embeddings_Facenet.pkl"
    train_ratio=0.8
    precision=16
    poly_modulus_degree=8192
    scale=2**40
    coeff_mod_bit_sizes=[60, 40, 40, 60]
    packing_size=32
    batch_size=12
    parallel_jobs=-1  # Use all available cores

    lfw_path = "datasets/lfw/lfw-deepfunneled/lfw-deepfunneled"  # Change this to your dataset location
    model_name = "Facenet"
    # Step 1: Load Embeddings
    embeddings = load_embeddings(file_path,lfw_path,model_name,calc=False)
    # Step 2: Extract Numeric Embeddings
    numeric_embeddings = extract_numeric_embeddings(embeddings)

    # Step 3: Split Train and Test
    train_embeddings, test_embeddings = split_embeddings(numeric_embeddings, 0.8)
    compute_cleartext_distances(test_embeddings,train_embeddings)
    compute_cleartext_top10_indices()

    results = biometric_identification(
        file_path,
        train_embeddings,
        test_embeddings,
        train_ratio,
        precision,
        poly_modulus_degree,
        scale,
        coeff_mod_bit_sizes,
        packing_size,
        batch_size,
        parallel_jobs,  # Use all available cores
        
    )
    print("Squared Euclidean Distance Matrix:")
    # Get the original matrix shape from scores.csv
    original_shape = pd.read_csv("scores.csv").shape
    # Call save_decrypted_scores with correct shape
    save_decrypted_scores(results, original_shape, "scores_dec.csv")
    save_top10_indices()
    compare_scores("scores.csv", "scores_dec.csv")
    compare_top10("top10.csv", "top10_dec.csv")
    encrypted_size = measure_encrypted_matrix_size(results)
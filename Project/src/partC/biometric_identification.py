import os
import numpy as np
import pickle
import tenseal as ts
from tqdm import tqdm
import logging
from joblib import Parallel, delayed
# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Step 1: Load Embeddings
def load_embeddings(file_path):
    """
    Load embeddings from a .pkl file.
    """
    with open(file_path, "rb") as f:
        embeddings = pickle.load(f)
    logger.info("Embeddings loaded from file.")
    return embeddings

# Step 2: Extract Numeric Embeddings
def extract_numeric_embeddings(embeddings):
    """Extract numeric embeddings from the raw data."""
    numeric_embeddings = [item[0]["embedding"] for item in embeddings.values() if isinstance(item, list) and len(item) > 0 and "embedding" in item[0]]
    logger.info("Numeric embeddings extracted.")
    return np.array(numeric_embeddings)

# Step 3: Split Embeddings into Train and Test
def split_embeddings(embeddings, train_ratio=0.8):
    num_train = int(len(embeddings) * train_ratio)
    train_embeddings = embeddings[:num_train]
    test_embeddings = embeddings[num_train:]
    logger.info("Embeddings split into train and test sets.")
    return train_embeddings, test_embeddings

# Step 4: Apply Low Precision
def apply_low_precision(embeddings, bit_size):
    if bit_size == 4:
        return embeddings.astype(np.int8)
    elif bit_size == 8:
        return embeddings.astype(np.uint8)
    elif bit_size == 16:
        return embeddings.astype(np.float16)
    elif bit_size == 32:
        return embeddings.astype(np.float32)
    return embeddings

# Step 5: Initialize CKKS Context
def initialize_ckks_context(poly_modulus_degree=16384, scale=2**40):
    """
    Initialize CKKS context with appropriate scaling.
    """
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=poly_modulus_degree,
        coeff_mod_bit_sizes=[60, 40, 40, 60]  # Add extra 40-bit level
    )
    context.global_scale = scale
    context.generate_galois_keys()
    context.generate_relin_keys()
    return context

# Step 6: Encrypt Data
def encrypt_embeddings(embeddings, context):
    packing_size = 32
    encrypted_embeddings = []
    for i in range(0, len(embeddings), packing_size):
        chunk = embeddings[i:i + packing_size]
        if len(chunk) < packing_size:
            chunk = np.vstack([chunk, np.zeros((packing_size - len(chunk), embeddings.shape[1]))])
        encrypted_embeddings.append(ts.ckks_vector(context, chunk.flatten().tolist()))
    logger.info("Embeddings encrypted.")
    return encrypted_embeddings

# Step 7: Precompute Norms
def compute_encrypted_norms(encrypted_embeddings):
    encrypted_norms = []
    for embedding in tqdm(encrypted_embeddings, desc="Computing encrypted norms", ncols=100):
        encrypted_norm = embedding.dot(embedding)
        encrypted_norms.append(encrypted_norm)
    logger.info("Encrypted norms computed.")
    return encrypted_norms

def batch_data(data, batch_size):
    batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
    logger.info("Data batched into %d batches of size %d.", len(batches), batch_size)
    return batches

# Define process_batch as a global function
def process_batch(test_batch, train_embeddings, train_norms):
    batch_similarities = []
    for test_vec, test_norm in tqdm(test_batch, desc="Processing test vectors", ncols=100):
        row_similarities = []
        for train_vec, train_norm in zip(train_embeddings, train_norms):
            # Compute encrypted dot product
            encrypted_dot_product = test_vec.dot(train_vec)
            dot_product = encrypted_dot_product.decrypt()[0]

            # Compute similarity
            similarity = dot_product / (test_norm * train_norm).decrypt()[0]
            row_similarities.append(similarity)
        batch_similarities.append(row_similarities)
    return batch_similarities

def compute_similarity(test_embeddings, test_norms, train_embeddings, train_norms, batch_size=512):
    similarity_matrix = []
    batch_size=16

    # Batch the test embeddings and norms
    test_batches = batch_data(list(zip(test_embeddings, test_norms)), batch_size)
    # Use Parallel to process batches in parallel
    results = Parallel(n_jobs=-1, backend="threading")(
        delayed(process_batch)(batch, train_embeddings, train_norms)
        for batch in tqdm(test_batches, desc="Processing test batches", ncols=100)
    )

    for batch_result in results:
        similarity_matrix.extend(batch_result)

    logger.info("Cosine similarity computation completed.")
    return np.array(similarity_matrix)

# Main Function
def biometric_identification(
    file_path, method="quantization", precision=8, train_ratio=0.8, poly_modulus_degree=8192, scale=2**40
):
    logger.info("Starting biometric identification.")
    import gc
    # Step 1: Load Embeddings
    embeddings = load_embeddings(file_path)

    # Step 2: Extract Numeric Embeddings
    numeric_embeddings = extract_numeric_embeddings(embeddings)

    # Step 3: Split Train and Test
    train_embeddings, test_embeddings = split_embeddings(numeric_embeddings, train_ratio)

    # Step 4: Apply Low Precision
    train_embeddings = apply_low_precision(train_embeddings, precision)
    test_embeddings = apply_low_precision(test_embeddings, precision)
    # Step 5: Initialize CKKS Context
    context = initialize_ckks_context(poly_modulus_degree, scale)
    
    # Step 6: Encrypt Data
    encrypted_train_embeddings = encrypt_embeddings(train_embeddings, context)
    encrypted_test_embeddings = encrypt_embeddings(test_embeddings, context)
    del embeddings,numeric_embeddings,context
    gc.collect()
    # Step 7: Precompute Norms
    encrypted_train_norms = compute_encrypted_norms(encrypted_train_embeddings)
    del train_embeddings
    gc.collect()
    encrypted_test_norms = compute_encrypted_norms(encrypted_test_embeddings)
    del test_embeddings
    gc.collect()
    # Step 8: Compute Cosine Similarity
    similarity_matrix = compute_similarity(encrypted_test_embeddings, encrypted_test_norms, encrypted_train_embeddings, encrypted_train_norms)

    logger.info("Biometric identification completed.")
    return similarity_matrix

# Example Usage
if __name__ == "__main__":
    results = biometric_identification(
        file_path="../partA/embeddings_Facenet.pkl",
        method="quantization",
        precision=16,
        train_ratio=0.8
    )
    print("Similarity Matrix:")
    print(results)

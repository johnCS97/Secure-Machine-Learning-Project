import time
import numpy as np
from tqdm import tqdm
import tenseal as ts  # TenSEAL library for homomorphic encryption
import logging

lfw_path = "./datasets/lfw/lfw-deepfunneled"  # Change this to your dataset location
model_name = "VGG-Face"
embeddings_file = "embeddings.pkl"  # File to save embeddings
# Helper function to generate random vectors
def generate_random_vector(dim, precision=3):
    """Generates a random vector with specified dimensions and precision."""
    return np.round(np.random.rand(dim), precision)

# Function to compute similarity on encrypted data
def compute_encrypted_similarity(vector1, vector2, context):
    """
    Computes the similarity score between two encrypted vectors using TenSEAL.
    
    Args:
        vector1: First plaintext vector.
        vector2: Second plaintext vector.
        context: TenSEAL context for homomorphic encryption.

    Returns:
        Decrypted similarity score.
    """
    # Encrypt the vectors
    encrypted_vec1 = ts.ckks_vector(context, vector1.tolist())
    encrypted_vec2 = ts.ckks_vector(context, vector2.tolist())

    # Compute encrypted dot product
    encrypted_dot_product = encrypted_vec1.dot(encrypted_vec2)

    # Compute encrypted norms
    encrypted_norm1 = encrypted_vec1.dot(encrypted_vec1)
    encrypted_norm2 = encrypted_vec2.dot(encrypted_vec2)

    # Decrypt results
    dot_product = encrypted_dot_product.decrypt()[0]
    norm1 = encrypted_norm1.decrypt()[0]
    norm2 = encrypted_norm2.decrypt()[0]

    # Calculate similarity score
    similarity_score = dot_product / (np.sqrt(norm1) * np.sqrt(norm2))
    return similarity_score

if __name__ == "__main__":
    # Configuration
    DIM = 4096  # Dimensionality of vectors
    REPEATS = 100  # Number of repetitions for benchmarking

    # Initialize lists to store runtime details for each step
    runtime_step1 = []  # For vector generation
    runtime_step2 = []  # For key generation
    runtime_step3 = []  # For encryption
    runtime_step4 = []  # For encrypted computation

    # Accuracy differences
    accuracy_diffs = []

    for _ in tqdm(range(REPEATS), ncols=100, desc="Computing Encrypted Similarity"):
        # Step 1: Generate new random vectors for each repetition
        start_time = time.time()
        vector1 = generate_random_vector(DIM)
        vector2 = generate_random_vector(DIM)
        runtime_step1.append(time.time() - start_time)

        # Step 2: Set up a new TenSEAL context and generate keys
        start_time = time.time()
        context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
        context.global_scale = 2**40
        context.generate_galois_keys()
        context.generate_relin_keys()
        runtime_step2.append(time.time() - start_time)

        # Step 3: Encrypt the vectors
        start_time = time.time()
        encrypted_vec1 = ts.ckks_vector(context, vector1.tolist())
        encrypted_vec2 = ts.ckks_vector(context, vector2.tolist())
        runtime_step3.append(time.time() - start_time)

        # Step 4: Compute similarity on encrypted data
        start_time = time.time()
        encrypted_dot_product = encrypted_vec1.dot(encrypted_vec2)
        encrypted_norm1 = encrypted_vec1.dot(encrypted_vec1)
        encrypted_norm2 = encrypted_vec2.dot(encrypted_vec2)
        dot_product = encrypted_dot_product.decrypt()[0]
        norm1 = encrypted_norm1.decrypt()[0]
        norm2 = encrypted_norm2.decrypt()[0]
        similarity_encrypted = dot_product / (np.sqrt(norm1) * np.sqrt(norm2))
        runtime_step4.append(time.time() - start_time)

        # Compute similarity in cleartext for comparison
        dot_product_cleartext = np.dot(vector1, vector2)
        norm1_cleartext = np.linalg.norm(vector1)
        norm2_cleartext = np.linalg.norm(vector2)
        similarity_cleartext = dot_product_cleartext / (norm1_cleartext * norm2_cleartext)

        # Record accuracy difference
        accuracy_diff = abs(similarity_encrypted - similarity_cleartext)
        accuracy_diffs.append(accuracy_diff)

    # Report accuracy results
    accuracy_report = (
        f"Average Accuracy Difference: {np.mean(accuracy_diffs):.6}\n"
        f"Standard Deviation of Accuracy Difference: {np.std(accuracy_diffs):.6f}\n"
        f"Max Accuracy Difference: {np.max(accuracy_diffs):.6f}\n"
    )

    runtime_report = (
        f"Step 1 (Vector Generation): Avg = {np.mean(runtime_step1):.4f}s, Std = {np.std(runtime_step1):.4f}s, Max = {np.max(runtime_step1):.4f}s\n"
        f"Step 2 (Key Generation): Avg = {np.mean(runtime_step2):.4f}s, Std = {np.std(runtime_step2):.4f}s, Max = {np.max(runtime_step2):.4f}s\n"
        f"Step 3 (Encryption): Avg = {np.mean(runtime_step3):.4f}s, Std = {np.std(runtime_step3):.4f}s, Max = {np.max(runtime_step3):.4f}s\n"
        f"Step 4 (Encrypted Computation): Avg = {np.mean(runtime_step4):.4f}s, Std = {np.std(runtime_step4):.4f}s, Max = {np.max(runtime_step4):.4f}s\n"
    )

    # Export results to a file
    with open("results_report.txt", "w") as f:
        f.write("Accuracy Results:\n")
        f.write(accuracy_report)
        f.write("\nRuntime Results:\n")
        f.write(runtime_report)

    # Print results to console as well
    print("Accuracy Results:")
    print(accuracy_report)
    print("Runtime Results:")
    print(runtime_report)

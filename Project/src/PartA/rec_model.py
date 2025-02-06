import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from joblib import Parallel, delayed
from sklearn.metrics import precision_score, recall_score, f1_score
from deepface import DeepFace

# Load embeddings

lfw_path = "./datasets/lfw/lfw-deepfunneled"  # Change this to your dataset location
model_name = "Facenet"
embeddings_file = "embeddings"+"_"+model_name+".pkl"

def get_numeric_embeddings(embeddings):
    keys = list(embeddings.keys())
    numeric_embeddings = np.array(
        [emb[0]['embedding'] for emb in embeddings.values() if emb],
        dtype=np.float64
    )
    return keys, numeric_embeddings

# Quantization function
def quantize_embedding(embedding, num_levels):
    min_val, max_val = np.min(embedding), np.max(embedding)
    if max_val == min_val:
        return np.full_like(embedding, min_val)
    normalized = (embedding - min_val) / (max_val - min_val)
    quantized = np.round(normalized * (num_levels - 1))
    return (quantized / (num_levels - 1)) * (max_val - min_val) + min_val

# Bit Reduction Function
def apply_bit_reduction(embeddings, bit_size):
    if bit_size == 4:
        return embeddings.astype(np.int8)
    elif bit_size == 8:
        return embeddings.astype(np.uint8)
    elif bit_size == 16:
        return embeddings.astype(np.float16)
    elif bit_size == 32:
        return embeddings.astype(np.float32)
    return embeddings

# Evaluation function
def evaluate_similarity_vectorized(similarity_matrix, keys, method_name, threshold=0.8):
    def calculate_is_same_identity(keys):
        def compare_pairs(i, keys):
            return [os.path.dirname(keys[i]) == os.path.dirname(keys[j]) for j in range(i + 1, len(keys))]
        
        results = Parallel(n_jobs=-1)(
            delayed(compare_pairs)(i, keys) for i in tqdm(range(len(keys)), desc=method_name, ncols=100)
        )
        return np.array([item for sublist in results for item in sublist])
    
    is_same_identity = calculate_is_same_identity(keys)
    triu_indices = np.triu_indices_from(similarity_matrix, k=1)
    similarity_scores = similarity_matrix[triu_indices]
    matches = similarity_scores > threshold
    accuracy = np.mean(matches == is_same_identity)
    
    return accuracy, is_same_identity, matches

def measure_performance(method_name, keys, embeddings, apply_method, *method_args):
        print(f"Processing: {method_name}")
        start_time = time.time()
        processed_embeddings = apply_method(embeddings, *method_args)
        similarity_matrix = cosine_similarity(processed_embeddings)
        accuracy, is_same_identity, matches = evaluate_similarity_vectorized(similarity_matrix, keys, method_name)
        elapsed_time = time.time() - start_time
        precision = precision_score(is_same_identity, matches, zero_division=0)
        recall = recall_score(is_same_identity, matches)
        f1 = f1_score(is_same_identity, matches)
        memory_usage = processed_embeddings.nbytes / (1024 * 1024)
        
        print(f"{method_name} - Accuracy: {accuracy:.4f}, Memory Usage: {memory_usage:.2f} MB, Time: {elapsed_time:.2f}s")
        print(f"{method_name} - Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, \n")
        
        return accuracy,elapsed_time, memory_usage,precision, recall, f1

def load_embeddings(embeddings_file):
    if os.path.exists(embeddings_file):
        print("Loading saved embeddings...\n")
        with open(embeddings_file, "rb") as f:
            return pickle.load(f)
    else:
        print("Calculating embeddings...")
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
        with open(embeddings_file, "wb") as f:
            pickle.dump(embeddings, f)
        return embeddings

if __name__ == "__main__":
    embeddings = load_embeddings(embeddings_file)
    keys, numeric_embeddings = get_numeric_embeddings(embeddings)
    performance_results = {"bit_reduction": [], "quantization": [], "combined": []}
    
    # Full Precision
    performance_results["full"] = measure_performance("Full Precision", keys, numeric_embeddings, lambda x: x)

    # Bit Reduction
    for bit_size in [4, 8, 16, 32]:
        performance_results["bit_reduction"].append(
            (bit_size, measure_performance(f"{bit_size}-bit", keys, numeric_embeddings, apply_bit_reduction, bit_size))
        )

    # Quantization
    for levels in [8, 16, 32, 64]:
        performance_results["quantization"].append(
            (levels, measure_performance(f"Quantization Levels={levels}", keys, numeric_embeddings,
                                         lambda emb, num_levels: np.array([quantize_embedding(e, num_levels) for e in emb]), levels))
        )

    # Plot results
    fig, axs = plt.subplots(3, 2, figsize=(14, 18))
    metrics = ["Accuracy", "Computation Time", "Memory Usage", "Precision", "Recall", "F1 Score"]
    positions = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)]

    for i, (metric, pos) in enumerate(zip(metrics, positions)):
        for key in ["bit_reduction", "quantization"]:
            if performance_results[key]:  # Ensure there's data to plot
                x_vals, y_vals = zip(*[(x, y[i]) for x, y in performance_results[key]])
                axs[pos].plot(x_vals, y_vals, 'o-', label=key.replace("_", " ").title())
        axs[pos].set_title(f"{metric} Comparison")
        axs[pos].legend()

    plt.tight_layout()
    plt.show()
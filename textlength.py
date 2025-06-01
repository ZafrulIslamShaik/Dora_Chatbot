# import json
# import os

# # Replace this with the actual path to your JSON file
# INPUT_FILE = "FINAL_processed_documents_cleaned.json"
# OUTPUT_FILE = "chunk_character_counts.json"

# def count_characters_per_chunk(input_path, output_path):
#     if not os.path.exists(input_path):
#         print(f"❌ File not found: {input_path}")
#         return

#     with open(input_path, "r", encoding="utf-8") as f:
#         data = json.load(f)

#     counts = []
#     for item in data:
#         chunk_number = item.get("metadata", {}).get("chunk_number")
#         text = item.get("text", "")
#         if isinstance(text, list):  # In case 'text' is a list
#             text = text[0] if text else ""
#         counts.append({
#             "chunk_number": chunk_number,
#             "char_count": len(text)
#         })

#     # Save as a report
#     with open(output_path, "w", encoding="utf-8") as f_out:
#         json.dump(counts, f_out, indent=2, ensure_ascii=False)

#     print(f"✅ Character count saved to: {output_path}")

# if __name__ == "__main__":
#     count_characters_per_chunk(INPUT_FILE, OUTPUT_FILE)















import json
import numpy as np
import os
import matplotlib.pyplot as plt

INPUT_FILE = "FINAL_processed_documents_cleaned.json"

def analyze_chunk_sizes(input_path):
    if not os.path.exists(input_path):
        print(f"❌ File not found: {input_path}")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sizes = []
    chunk_map = []

    for item in data:
        chunk_number = item.get("metadata", {}).get("chunk_number")
        text = item.get("text", "")
        if isinstance(text, list):
            text = text[0] if text else ""
        char_count = len(text)
        sizes.append(char_count)
        chunk_map.append((chunk_number, char_count))

    sizes = np.array(sizes)

    # Descriptive statistics
    mean = np.mean(sizes)
    median = np.median(sizes)
    std = np.std(sizes)
    q1 = np.percentile(sizes, 25)
    q3 = np.percentile(sizes, 75)
    iqr = q3 - q1
    min_val = np.min(sizes)
    max_val = np.max(sizes)

    # Outlier detection using IQR
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = [(i, s) for i, s in chunk_map if s < lower_bound or s > upper_bound]

    print("\n📊 Chunk Statistics:")
    print(f"Total Chunks: {len(sizes)}")
    print(f"Mean: {mean:.2f}")
    print(f"Median: {median}")
    print(f"Std Dev: {std:.2f}")
    print(f"Min: {min_val}, Max: {max_val}")
    print(f"IQR: {iqr:.2f}")
    print(f"Outlier Thresholds: < {lower_bound:.2f} or > {upper_bound:.2f}")
    print(f"Outliers Found: {len(outliers)}")

    for chunk_id, size in outliers[:10]:  # show first 10 outliers
        print(f"  ⚠️ Chunk {chunk_id} has {size} characters")

    # Optional: plot histogram
    plt.hist(sizes, bins=30, color='skyblue', edgecolor='black')
    plt.title("Distribution of Chunk Sizes (Characters)")
    plt.xlabel("Chunk Size")
    plt.ylabel("Frequency")
    plt.axvline(mean, color='red', linestyle='dashed', linewidth=1, label=f"Mean: {mean:.1f}")
    plt.axvline(median, color='green', linestyle='dashed', linewidth=1, label=f"Median: {median}")
    plt.legend()
    plt.tight_layout()
    plt.savefig("chunk_size_histogram.png")
    plt.show()

if __name__ == "__main__":
    analyze_chunk_sizes(INPUT_FILE)



# import json
# import os
# import numpy as np
# from collections import defaultdict

# # Use your existing file paths
# INPUT_FILE = "FINAL_processed_documents_cleaned.json"
# OUTPUT_FILE = "chunk_character_counts.json"
# RECHUNKED_OUTPUT = "optimized_chunks.json"

# def analyze_and_rechunk(input_path, stats_output_path, rechunked_output_path):
#     if not os.path.exists(input_path):
#         print(f"❌ File not found: {input_path}")
#         return

#     with open(input_path, "r", encoding="utf-8") as f:
#         data = json.load(f)

#     # --- PART 1: STATISTICAL ANALYSIS ---
#     counts = []
#     for item in data:
#         chunk_number = item.get("metadata", {}).get("chunk_number")
#         text = item.get("text", "")
#         if isinstance(text, list):  # In case 'text' is a list
#             text = text[0] if text else ""
#         counts.append({
#             "chunk_number": chunk_number,
#             "char_count": len(text),
#             "original_index": data.index(item)  # Keep track of position
#         })

#     # Extract just the character counts for statistical analysis
#     char_counts = [item["char_count"] for item in counts]
    
#     # Calculate quartiles and IQR for outlier detection
#     q1 = np.percentile(char_counts, 25)
#     median = np.percentile(char_counts, 50)
#     q3 = np.percentile(char_counts, 75)
#     iqr = q3 - q1
    
#     # Define outlier boundaries
#     lower_bound = max(300, q1 - 1.5 * iqr)  # Don't go below 300 chars minimum
#     upper_bound = q3 + 1.5 * iqr
    
#     # Identify outliers
#     small_outliers = [item for item in counts if item["char_count"] < lower_bound]
#     large_outliers = [item for item in counts if item["char_count"] > upper_bound]
#     normal_chunks = [item for item in counts if lower_bound <= item["char_count"] <= upper_bound]
    
#     # Calculate optimal chunking parameters
#     min_chunk_size = max(500, lower_bound)  # Minimum viable chunk
#     target_chunk_size = median  # Target for rechunking
#     max_chunk_size = min(2500, upper_bound)  # Cap maximum size
    
#     # Add statistics to the counts output
#     statistics = {
#         "total_chunks": len(counts),
#         "quartiles": {"q1": q1, "median": median, "q3": q3},
#         "iqr": iqr,
#         "outlier_boundaries": {"lower": lower_bound, "upper": upper_bound},
#         "outlier_counts": {
#             "small_outliers": len(small_outliers),
#             "large_outliers": len(large_outliers),
#             "normal_chunks": len(normal_chunks)
#         },
#         "chunking_parameters": {
#             "min_chunk_size": min_chunk_size,
#             "target_chunk_size": target_chunk_size,
#             "max_chunk_size": max_chunk_size
#         }
#     }
    
#     # Save statistics and counts
#     output_data = {
#         "chunk_counts": counts,
#         "statistics": statistics
#     }
    
#     with open(stats_output_path, "w", encoding="utf-8") as f_out:
#         json.dump(output_data, f_out, indent=2, ensure_ascii=False)
    
#     print(f"✅ Character counts and statistics saved to: {stats_output_path}")
#     print(f"📊 Statistical Summary:")
#     print(f"   - Total chunks: {len(counts)}")
#     print(f"   - Q1: {q1:.0f}, Median: {median:.0f}, Q3: {q3:.0f}")
#     print(f"   - IQR: {iqr:.0f}")
#     print(f"   - Small outliers: {len(small_outliers)}, Large outliers: {len(large_outliers)}")
#     print(f"   - Recommended chunking parameters:")
#     print(f"     * Min chunk size: {min_chunk_size:.0f} chars")
#     print(f"     * Target chunk size: {target_chunk_size:.0f} chars")
#     print(f"     * Max chunk size: {max_chunk_size:.0f} chars")
    
#     return statistics

# if __name__ == "__main__":
#     analyze_and_rechunk(INPUT_FILE, OUTPUT_FILE, RECHUNKED_OUTPUT)
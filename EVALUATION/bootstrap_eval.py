import json
import os
import numpy as np
import pandas as pd
from scipy import stats
import random
from collections import defaultdict
from functools import reduce


NUM_BOOTSTRAP_SAMPLES = 1000
CONFIDENCE_INTERVAL = 95  
COUNT_FOLDER = "count"
OUTPUT_FOLDER = "bootstrap_results"

START_QUESTION_ID = 1  
END_QUESTION_ID = 400  

# Create output folder if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def load_data_from_count_folder(start_qid, end_qid):
    """
    Load all JSON files from the count folder and filter by question ID range.
    
    Args:
        start_qid: Starting question ID number (e.g., 1 for Q001)
        end_qid: Ending question ID number (e.g., 100 for Q100)
    """
    all_data = []
    filtered_data = []
    
    # Format question ID range for filtering
    start_qid_formatted = int(start_qid)
    end_qid_formatted = int(end_qid)
    
    # List all count files
    count_files = [f for f in os.listdir(COUNT_FOLDER) if f.startswith("count_") and f.endswith(".json")]
    
    print(f"\nFound {len(count_files)} files in the count folder:")
    for file_name in count_files:
        print(f"  - {file_name}")
        file_path = os.path.join(COUNT_FOLDER, file_name)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                file_data = json.load(f)
                all_data.extend(file_data)
                
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    for item in all_data:
        qid = item.get("question_id", "")
        
        if qid.startswith("Q"):
            try:
                qid_num = int(qid[1:])
                if start_qid_formatted <= qid_num <= end_qid_formatted:
                    filtered_data.append(item)
            except ValueError:
                continue
    
    # Report filtering results
    total_questions = len(set(item.get("question_id", "") for item in filtered_data))
    print(f"\nFiltered data to {total_questions} unique questions from Q{start_qid:03d} to Q{end_qid:03d}")
    
    expected_count = end_qid_formatted - start_qid_formatted + 1
    if total_questions < expected_count:
        missing_count = expected_count - total_questions
        print(f"Note: {missing_count} question IDs within the specified range are missing from the dataset")
    
    return filtered_data

def aggregate_scores(data):
    """
    Aggregate scores across evaluator models and metrics.
    Returns a dataframe with one aggregated score per question per configuration.
    """
    aggregated_data = []
    
    # Group data by question_id, method, k, version
    question_configs = {}
    
    for item in data:
        qid = item.get("question_id")
        method = item.get("method")
        k = item.get("k")
        version = item.get("version")
        
        key = (qid, method, k, version)
        if key not in question_configs:
            question_configs[key] = item
    
    # Process each question/configuration
    for (qid, method, k, version), item in question_configs.items():

        metric_averages = {}
        
        for metric in ["faithfulness", "completeness", "relevancy"]:
            if metric in item:
                # Get scores for all evaluator models for this metric
                model_scores = []
                for model in ["Mistral", "gemma", "Phi3"]:
                    if model in item[metric]:
                        model_scores.append(item[metric][model])
                

                if model_scores:
                    metric_averages[metric] = sum(model_scores) / len(model_scores)
        

        if metric_averages:
            # Calculate arithmetic mean
            arith_avg = sum(metric_averages.values()) / len(metric_averages)
            
            # Calculate geometric mean
            # Ensure no zeros or negative values
            safe_scores = [max(score, 0.001) for score in metric_averages.values()]
            geo_avg = reduce(lambda x, y: x * y, safe_scores) ** (1.0 / len(safe_scores))
            
            # Add to aggregated data
            aggregated_data.append({
                "question_id": qid,
                "method": method,
                "k": k,
                "version": version,
                "arith_avg_score": arith_avg,
                "geo_avg_score": geo_avg,
                # Store individual metric averages for detailed analysis
                "faithfulness_avg": metric_averages.get("faithfulness"),
                "completeness_avg": metric_averages.get("completeness"),
                "relevancy_avg": metric_averages.get("relevancy")
            })
    
    # Convert to dataframe
    df = pd.DataFrame(aggregated_data)
    
    # Print data summary
    print(f"\nData Summary:")
    print(f"  - Total questions: {df['question_id'].nunique()}")
    print(f"  - Methods: {', '.join(df['method'].unique())}")
    print(f"  - K values: {', '.join(str(k) for k in sorted(df['k'].unique()))}")
    print(f"  - Versions: {', '.join(df['version'].unique())}")
    
    return df

def stratified_bootstrap_sample(data_df, metric_col, n_bootstrap=1000, confidence=95):
    """
    Perform stratified bootstrap sampling to estimate the mean and confidence interval.
    Stratifies by question_id to ensure fair representation.
    
    Args:
        data_df: DataFrame containing scores with a question_id column
        metric_col: Column name to use for the metric
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence interval percentage
    
    Returns:
        mean, lower_ci, upper_ci
    """
    bootstrap_means = []
    original_mean = data_df[metric_col].mean()
    
    # Group data by question_id
    question_groups = data_df.groupby('question_id')
    question_ids = list(question_groups.groups.keys())
    
    for _ in range(n_bootstrap):
        # For each bootstrap iteration
        bootstrap_sample = []
        
        # Sample question_ids with replacement
        sampled_question_ids = random.choices(question_ids, k=len(question_ids))
        
        # For each sampled question, take all corresponding rows
        for qid in sampled_question_ids:
            question_data = question_groups.get_group(qid)
            # If there are multiple rows for this question (multiple configs), 
            # randomly select one with replacement
            if len(question_data) > 1:
                sampled_row = question_data.sample(n=1, replace=True)
            else:
                sampled_row = question_data
            
            bootstrap_sample.append(sampled_row[metric_col].values[0])
        
        bootstrap_means.append(np.mean(bootstrap_sample))
    
    # Calculate confidence interval
    lower_percentile = (100 - confidence) / 2
    upper_percentile = 100 - lower_percentile
    
    lower_ci = np.percentile(bootstrap_means, lower_percentile)
    upper_ci = np.percentile(bootstrap_means, upper_percentile)
    
    return original_mean, lower_ci, upper_ci

def compare_configurations(df):
    """
    Compare different configurations using stratified bootstrapping.
    
    Returns:
        Dictionary with results for each configuration and metric
    """
    results = {}
    
    # Analyze with arithmetic mean
    print(f"\nStratified Bootstrap Analysis - Arithmetic Mean ({NUM_BOOTSTRAP_SAMPLES} samples, {CONFIDENCE_INTERVAL}% CI):")
    print(f"{'Configuration':<25} {'Mean':<10} {'CI':<20} {'Sample Size':<15}")
    print(f"{'-'*70}")
    
    arith_config_results = []
    for (method, k, version), group in df.groupby(["method", "k", "version"]):
        # Perform stratified bootstrapping - pass the DataFrame for this configuration
        mean, lower_ci, upper_ci = stratified_bootstrap_sample(
            group, 
            metric_col="arith_avg_score",
            n_bootstrap=NUM_BOOTSTRAP_SAMPLES, 
            confidence=CONFIDENCE_INTERVAL
        )
        
        config_name = f"{method}, k={k}, {version}"
        ci_str = f"[{lower_ci:.4f}, {upper_ci:.4f}]"
        print(f"{config_name:<25} {mean:.4f}     {ci_str:<20} {len(group):<15}")
        
        arith_config_results.append({
            "method": method,
            "k": k,
            "version": version,
            "mean": mean,
            "lower_ci": lower_ci,
            "upper_ci": upper_ci,
            "sample_size": len(group)
        })
    
    # Sort by mean score (descending)
    arith_config_results = sorted(arith_config_results, key=lambda x: x["mean"], reverse=True)
    
    # Analyze with geometric mean
    print(f"\nStratified Bootstrap Analysis - Geometric Mean ({NUM_BOOTSTRAP_SAMPLES} samples, {CONFIDENCE_INTERVAL}% CI):")
    print(f"{'Configuration':<25} {'Mean':<10} {'CI':<20} {'Sample Size':<15}")
    print(f"{'-'*70}")
    
    geo_config_results = []
    for (method, k, version), group in df.groupby(["method", "k", "version"]):
        # Perform stratified bootstrapping - pass the DataFrame for this configuration
        mean, lower_ci, upper_ci = stratified_bootstrap_sample(
            group, 
            metric_col="geo_avg_score",
            n_bootstrap=NUM_BOOTSTRAP_SAMPLES, 
            confidence=CONFIDENCE_INTERVAL
        )
        
        config_name = f"{method}, k={k}, {version}"
        ci_str = f"[{lower_ci:.4f}, {upper_ci:.4f}]"
        print(f"{config_name:<25} {mean:.4f}     {ci_str:<20} {len(group):<15}")
        
        geo_config_results.append({
            "method": method,
            "k": k,
            "version": version,
            "mean": mean,
            "lower_ci": lower_ci,
            "upper_ci": upper_ci,
            "sample_size": len(group)
        })
    
    # Sort by mean score (descending)
    geo_config_results = sorted(geo_config_results, key=lambda x: x["mean"], reverse=True)
    
    results = {
        "arithmetic": arith_config_results,
        "geometric": geo_config_results
    }
    
    return results

def holm_bonferroni_correction(p_values):
    """
    Apply Holm-Bonferroni correction to p-values.
    
    Args:
        p_values: List of p-values
        
    Returns:
        List of corrected p-values
    """
    n = len(p_values)
    if n <= 1:
        return p_values
    
    # Sort p-values and keep track of original indices
    sorted_indices = np.argsort(p_values)
    sorted_p_values = [p_values[i] for i in sorted_indices]
    
    # Apply Holm-Bonferroni correction
    corrected_p_values = [0] * n
    for i, p in enumerate(sorted_p_values):
        corrected_p = p * (n - i)
        corrected_p = min(corrected_p, 1.0)  # Ensure p-value doesn't exceed 1
        
        # Ensure monotonicity (each correction is at least as severe as the next)
        if i > 0 and corrected_p < corrected_p_values[i-1]:
            corrected_p = corrected_p_values[i-1]
        
        corrected_p_values[i] = corrected_p
    
    # Restore original order
    original_order_corrected = [0] * n
    for i, orig_idx in enumerate(sorted_indices):
        original_order_corrected[orig_idx] = corrected_p_values[i]
    
    return original_order_corrected

def statistical_significance_test(df, metric_col):
    """
    Perform statistical significance tests between configurations with Holm-Bonferroni correction.
    
    Returns:
        Dictionary with pairwise p-values
    """
    significance_results = []
    
    print(f"\nStatistical Significance Tests for {metric_col} (with Holm-Bonferroni correction):")
    print(f"{'Configuration 1':<25} {'Configuration 2':<25} {'p-value':<10} {'Corrected p':<15} {'Significant':<10}")
    print(f"{'-'*85}")
    
    # Get all configurations
    configs = []
    for (method, k, version), group in df.groupby(["method", "k", "version"]):
        config_name = f"{method}_{k}_{version}"
        configs.append({
            "name": config_name,
            "method": method,
            "k": k,
            "version": version,
            "scores": group[metric_col].values,
            "data": group  # Store the group data for potential further analysis
        })
    
    # Perform pairwise Mann-Whitney U tests
    p_values = []
    for i, config1 in enumerate(configs):
        for j, config2 in enumerate(configs):
            if i < j:  # Only do each pair once
                u_stat, p_value = stats.mannwhitneyu(
                    config1["scores"], 
                    config2["scores"],
                    alternative='two-sided'
                )
                
                p_values.append(p_value)
                
                significance_results.append({
                    "config1": config1["name"],
                    "config2": config2["name"],
                    "config1_str": f"{config1['method']}, k={config1['k']}, {config1['version']}",
                    "config2_str": f"{config2['method']}, k={config2['k']}, {config2['version']}",
                    "p_value": p_value,
                    "index": len(significance_results)
                })
    
    # Apply Holm-Bonferroni correction
    corrected_p_values = holm_bonferroni_correction(p_values)
    
    # Update significance results with corrected p-values
    for i, result in enumerate(significance_results):
        result["corrected_p_value"] = corrected_p_values[i]
        result["significant"] = result["corrected_p_value"] < 0.05
    
    # Sort results by p-value for display
    significance_results = sorted(significance_results, key=lambda x: x["p_value"])
    
    # Display results
    significant_found = False
    for result in significance_results:
        if result["significant"]:
            significant_found = True
            print(f"{result['config1_str']:<25} {result['config2_str']:<25} {result['p_value']:.4f}    {result['corrected_p_value']:.4f}       {'Yes' if result['significant'] else 'No':<10}")
    
    # If no significant results were found
    if not significant_found:
        print("  No statistically significant differences found after Holm-Bonferroni correction")
    
    return significance_results

def analyze_by_factor(df):
    """
    Analyze performance by individual factors (method, k, version).
    Uses stratified bootstrap sampling for more robust results.
    """
    results = {}
    
    # Analyze using arithmetic mean
    print(f"\nPerformance Analysis By Factor - Arithmetic Mean:")
    arith_factor_results = analyze_by_factor_for_metric(df, "arith_avg_score", "Arithmetic Mean")
    
    # Analyze using geometric mean
    print(f"\nPerformance Analysis By Factor - Geometric Mean:")
    geo_factor_results = analyze_by_factor_for_metric(df, "geo_avg_score", "Geometric Mean")
    
    results = {
        "arithmetic": arith_factor_results,
        "geometric": geo_factor_results
    }
    
    return results

def analyze_by_factor_for_metric(df, metric_col, metric_name):
    """Helper function to analyze performance by factor for a specific metric"""
    factor_results = {}
    
    # Analyze by method
    print(f"\n  Method Comparison ({metric_name}):")
    method_results = []
    
    for method, group in df.groupby("method"):
        # Stratified bootstrap by question_id
        mean, lower_ci, upper_ci = stratified_bootstrap_sample(
            group, 
            metric_col=metric_col,
            n_bootstrap=NUM_BOOTSTRAP_SAMPLES, 
            confidence=CONFIDENCE_INTERVAL
        )
        
        method_results.append({
            "method": method,
            "mean": mean,
            "lower_ci": lower_ci,
            "upper_ci": upper_ci,
            "sample_size": len(group)
        })
    
    # Sort by mean score
    method_results = sorted(method_results, key=lambda x: x["mean"], reverse=True)
    
    # Display results
    print(f"  {'Method':<10} {'Mean':<10} {'CI':<20} {'Sample Size':<15}")
    print(f"  {'-'*60}")
    for result in method_results:
        ci_str = f"[{result['lower_ci']:.4f}, {result['upper_ci']:.4f}]"
        print(f"  {result['method']:<10} {result['mean']:.4f}     {ci_str:<20} {result['sample_size']:<15}")
    
    # Analyze by k value
    print(f"\n  K Value Comparison ({metric_name}):")
    k_results = []
    
    for k, group in df.groupby("k"):
        # Stratified bootstrap by question_id
        mean, lower_ci, upper_ci = stratified_bootstrap_sample(
            group, 
            metric_col=metric_col,
            n_bootstrap=NUM_BOOTSTRAP_SAMPLES, 
            confidence=CONFIDENCE_INTERVAL
        )
        
        k_results.append({
            "k": k,
            "mean": mean,
            "lower_ci": lower_ci,
            "upper_ci": upper_ci,
            "sample_size": len(group)
        })
    
    # Sort by mean score
    k_results = sorted(k_results, key=lambda x: x["mean"], reverse=True)
    
    # Display results
    print(f"  {'K Value':<10} {'Mean':<10} {'CI':<20} {'Sample Size':<15}")
    print(f"  {'-'*60}")
    for result in k_results:
        ci_str = f"[{result['lower_ci']:.4f}, {result['upper_ci']:.4f}]"
        print(f"  {result['k']:<10} {result['mean']:.4f}     {ci_str:<20} {result['sample_size']:<15}")
    
    # Analyze by version
    print(f"\n  Version Comparison ({metric_name}):")
    version_results = []
    
    for version, group in df.groupby("version"):
        # Stratified bootstrap by question_id
        mean, lower_ci, upper_ci = stratified_bootstrap_sample(
            group, 
            metric_col=metric_col,
            n_bootstrap=NUM_BOOTSTRAP_SAMPLES, 
            confidence=CONFIDENCE_INTERVAL
        )
        
        version_results.append({
            "version": version,
            "mean": mean,
            "lower_ci": lower_ci,
            "upper_ci": upper_ci,
            "sample_size": len(group)
        })
    
    # Sort by mean score
    version_results = sorted(version_results, key=lambda x: x["mean"], reverse=True)
    
    # Display results
    print(f"  {'Version':<10} {'Mean':<10} {'CI':<20} {'Sample Size':<15}")
    print(f"  {'-'*60}")
    for result in version_results:
        ci_str = f"[{result['lower_ci']:.4f}, {result['upper_ci']:.4f}]"
        print(f"  {result['version']:<10} {result['mean']:.4f}     {ci_str:<20} {result['sample_size']:<15}")
    
    return {
        "method": method_results,
        "k": k_results,
        "version": version_results
    }

def generate_report(config_results, factor_results, arith_significance, geo_significance, start_qid, end_qid):
    """Generate a summary report of the bootstrap analysis."""
    report = []
    
    # 1. Overall Best Configurations
    best_config_arith = config_results["arithmetic"][0]
    best_config_geo = config_results["geometric"][0]
    
    report.append("# RAG Configuration Evaluation Report")
    report.append(f"\n## Analysis Range: Q{start_qid:03d} to Q{end_qid:03d}")
    report.append("\n## Statistical Methodology")
    report.append("- **Metrics**: Both Arithmetic Mean and Geometric Mean were used for robust evaluation")
    report.append("- **Stratified Bootstrap Sampling**: Ensures fair representation of all question IDs")
    report.append("- **Multiple Test Correction**: Holm-Bonferroni method to control Type I error rate")
    report.append(f"- **Bootstrap Samples**: {NUM_BOOTSTRAP_SAMPLES}")
    report.append(f"- **Confidence Interval**: {CONFIDENCE_INTERVAL}%")
    
    report.append("\n## Overall Best Configuration")
    
    # By Arithmetic Mean
    report.append("\n### By Arithmetic Mean")
    report.append(f"- **Method**: {best_config_arith['method']}")
    report.append(f"- **K value**: {best_config_arith['k']}")
    report.append(f"- **Version**: {best_config_arith['version']}")
    report.append(f"- **Average Score**: {best_config_arith['mean']:.4f}")
    report.append(f"- **95% Confidence Interval**: [{best_config_arith['lower_ci']:.4f}, {best_config_arith['upper_ci']:.4f}]")
    
    # By Geometric Mean
    report.append("\n### By Geometric Mean")
    report.append(f"- **Method**: {best_config_geo['method']}")
    report.append(f"- **K value**: {best_config_geo['k']}")
    report.append(f"- **Version**: {best_config_geo['version']}")
    report.append(f"- **Average Score**: {best_config_geo['mean']:.4f}")
    report.append(f"- **95% Confidence Interval**: [{best_config_geo['lower_ci']:.4f}, {best_config_geo['upper_ci']:.4f}]")
    
    # 2. Configuration Rankings - Arithmetic Mean
    report.append("\n## Configuration Rankings by Arithmetic Mean")
    report.append("\n| Rank | Configuration | Average Score | 95% CI |")
    report.append("| ---- | ------------- | ------------- | ------ |")
    
    for i, result in enumerate(config_results["arithmetic"]):
        config_str = f"{result['method']}, k={result['k']}, {result['version']}"
        ci_str = f"[{result['lower_ci']:.4f}, {result['upper_ci']:.4f}]"
        report.append(f"| {i+1} | {config_str} | {result['mean']:.4f} | {ci_str} |")
    
    # 3. Configuration Rankings - Geometric Mean
    report.append("\n## Configuration Rankings by Geometric Mean")
    report.append("\n| Rank | Configuration | Average Score | 95% CI |")
    report.append("| ---- | ------------- | ------------- | ------ |")
    
    for i, result in enumerate(config_results["geometric"]):
        config_str = f"{result['method']}, k={result['k']}, {result['version']}"
        ci_str = f"[{result['lower_ci']:.4f}, {result['upper_ci']:.4f}]"
        report.append(f"| {i+1} | {config_str} | {result['mean']:.4f} | {ci_str} |")
    
    # 4. Factor Analysis
    report.append("\n## Factor Analysis")
    
    # Method analysis - Arithmetic Mean
    report.append("\n### Retrieval Method Comparison - Arithmetic Mean")
    report.append("\n| Method | Average Score | 95% CI |")
    report.append("| ------ | ------------- | ------ |")
    
    for result in factor_results["arithmetic"]["method"]:
        ci_str = f"[{result['lower_ci']:.4f}, {result['upper_ci']:.4f}]"
        report.append(f"| {result['method']} | {result['mean']:.4f} | {ci_str} |")
    
    # Method analysis - Geometric Mean
    report.append("\n### Retrieval Method Comparison - Geometric Mean")
    report.append("\n| Method | Average Score | 95% CI |")
    report.append("| ------ | ------------- | ------ |")
    
    for result in factor_results["geometric"]["method"]:
        ci_str = f"[{result['lower_ci']:.4f}, {result['upper_ci']:.4f}]"
        report.append(f"| {result['method']} | {result['mean']:.4f} | {ci_str} |")
    
    # K value analysis - Arithmetic Mean
    report.append("\n### K Value Comparison - Arithmetic Mean")
    report.append("\n| K Value | Average Score | 95% CI |")
    report.append("| ------- | ------------- | ------ |")
    
    for result in factor_results["arithmetic"]["k"]:
        ci_str = f"[{result['lower_ci']:.4f}, {result['upper_ci']:.4f}]"
        report.append(f"| {result['k']} | {result['mean']:.4f} | {ci_str} |")
    
    # K value analysis - Geometric Mean
    report.append("\n### K Value Comparison - Geometric Mean")
    report.append("\n| K Value | Average Score | 95% CI |")
    report.append("| ------- | ------------- | ------ |")
    
    for result in factor_results["geometric"]["k"]:
        ci_str = f"[{result['lower_ci']:.4f}, {result['upper_ci']:.4f}]"
        report.append(f"| {result['k']} | {result['mean']:.4f} | {ci_str} |")
    
    # Version analysis - Arithmetic Mean
    report.append("\n### Version Comparison - Arithmetic Mean")
    report.append("\n| Version | Average Score | 95% CI |")
    report.append("| ------- | ------------- | ------ |")
    
    for result in factor_results["arithmetic"]["version"]:
        ci_str = f"[{result['lower_ci']:.4f}, {result['upper_ci']:.4f}]"
        report.append(f"| {result['version']} | {result['mean']:.4f} | {ci_str} |")
    
    # Version analysis - Geometric Mean
    report.append("\n### Version Comparison - Geometric Mean")
    report.append("\n| Version | Average Score | 95% CI |")
    report.append("| ------- | ------------- | ------ |")
    
    for result in factor_results["geometric"]["version"]:
        ci_str = f"[{result['lower_ci']:.4f}, {result['upper_ci']:.4f}]"
        report.append(f"| {result['version']} | {result['mean']:.4f} | {ci_str} |")
    
    # 5. Statistical Significance for Arithmetic Mean
    report.append("\n## Statistical Significance - Arithmetic Mean")
    
    arith_significant_results = [r for r in arith_significance if r["significant"]]
    
    if arith_significant_results:
        report.append("\nConfigurations with statistically significant differences (p < 0.05 after Holm-Bonferroni correction):")
        report.append("\n| Configuration 1 | Configuration 2 | p-value | Corrected p-value |")
        report.append("| -------------- | -------------- | ------- | ----------------- |")
        
        for result in sorted(arith_significant_results, key=lambda x: x["p_value"]):
            report.append(f"| {result['config1_str']} | {result['config2_str']} | {result['p_value']:.4f} | {result['corrected_p_value']:.4f} |")
    else:
        report.append("\nNo statistically significant differences were found between configurations after Holm-Bonferroni correction.")
    
    # 6. Statistical Significance for Geometric Mean
    report.append("\n## Statistical Significance - Geometric Mean")
    
    geo_significant_results = [r for r in geo_significance if r["significant"]]
    
    if geo_significant_results:
        report.append("\nConfigurations with statistically significant differences (p < 0.05 after Holm-Bonferroni correction):")
        report.append("\n| Configuration 1 | Configuration 2 | p-value | Corrected p-value |")
        report.append("| -------------- | -------------- | ------- | ----------------- |")
        
        for result in sorted(geo_significant_results, key=lambda x: x["p_value"]):
            report.append(f"| {result['config1_str']} | {result['config2_str']} | {result['p_value']:.4f} | {result['corrected_p_value']:.4f} |")
    else:
        report.append("\nNo statistically significant differences were found between configurations after Holm-Bonferroni correction.")
    
    # 7. Recommendations
    report.append("\n## Recommendations")
    
    # Get best factors by arithmetic mean
    best_method_arith = factor_results["arithmetic"]["method"][0]["method"]
    best_k_arith = factor_results["arithmetic"]["k"][0]["k"]
    best_version_arith = factor_results["arithmetic"]["version"][0]["version"]
    
    # Get best factors by geometric mean
    best_method_geo = factor_results["geometric"]["method"][0]["method"]
    best_k_geo = factor_results["geometric"]["k"][0]["k"]
    best_version_geo = factor_results["geometric"]["version"][0]["version"]
    
    report.append(f"\n### Based on Arithmetic Mean")
    report.append(f"1. **Retrieval Method**: {best_method_arith}")
    report.append(f"2. **K Value**: {best_k_arith}")
    report.append(f"3. **Version**: {best_version_arith}")
    report.append(f"4. **Best Overall Configuration**: {best_config_arith['method']}, k={best_config_arith['k']}, {best_config_arith['version']}")
    
    report.append(f"\n### Based on Geometric Mean")
    report.append(f"1. **Retrieval Method**: {best_method_geo}")
    report.append(f"2. **K Value**: {best_k_geo}")
    report.append(f"3. **Version**: {best_version_geo}")
    report.append(f"4. **Best Overall Configuration**: {best_config_geo['method']}, k={best_config_geo['k']}, {best_config_geo['version']}")
    
    # Final balanced recommendation
    report.append(f"\n### Final Recommendation")
    if (best_config_arith['method'] == best_config_geo['method'] and 
        best_config_arith['k'] == best_config_geo['k'] and 
        best_config_arith['version'] == best_config_geo['version']):
        report.append(f"The configuration **{best_config_arith['method']}, k={best_config_arith['k']}, {best_config_arith['version']}** performs best on both arithmetic and geometric mean metrics, making it the clear choice for deployment.")
    else:
        report.append(f"For balancing average performance and reliability:")
        report.append(f"- If prioritizing overall performance: **{best_config_arith['method']}, k={best_config_arith['k']}, {best_config_arith['version']}**")
        report.append(f"- If prioritizing consistent performance: **{best_config_geo['method']}, k={best_config_geo['k']}, {best_config_geo['version']}**")
        report.append(f"\nFor RAG systems, we recommend prioritizing the geometric mean results since it better captures consistent performance across all questions.")
    
    # Note about individual factor rankings
    if (best_method_arith != best_config_arith["method"] or 
        best_k_arith != best_config_arith["k"] or 
        best_version_arith != best_config_arith["version"] or
        best_method_geo != best_config_geo["method"] or
        best_k_geo != best_config_geo["k"] or
        best_version_geo != best_config_geo["version"]):
        report.append(f"\n**Note**: The individual best factors are based on their average performance across all configurations. The best overall configurations are the specific combinations that perform best.")
    
    # Save report
    report_path = os.path.join(OUTPUT_FOLDER, f"bootstrap_report_Q{start_qid:03d}_Q{end_qid:03d}.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    
    print(f"\nDetailed report saved to: {report_path}")

def main():
    print("=" * 80)
    print("          RAG CONFIGURATION EVALUATION USING BOOTSTRAP ANALYSIS")
    print("=" * 80)
    print(f"          Analyzing questions from Q{START_QUESTION_ID:03d} to Q{END_QUESTION_ID:03d}")
    print("=" * 80)
    
    # Load and filter data
    print("\nStep 1: Loading and filtering data from count folder...")
    data = load_data_from_count_folder(START_QUESTION_ID, END_QUESTION_ID)
    
    if not data:
        print("No data found in the specified question ID range. Please check your input parameters.")
        return
    
    # Aggregate scores
    print("\nStep 2: Aggregating scores across evaluator models...")
    df = aggregate_scores(data)
    
    # Analyze performance by individual factors
    print("\nStep 3: Analyzing performance by factor using stratified bootstrap...")
    factor_results = analyze_by_factor(df)
    
    # Run bootstrap analysis for configurations
    print("\nStep 4: Comparing configurations using stratified bootstrap...")
    config_results = compare_configurations(df)
    
    # Run statistical significance tests for arithmetic mean
    print("\nStep 5a: Testing statistical significance for arithmetic mean...")
    arith_significance = statistical_significance_test(df, "arith_avg_score")
    
    # Run statistical significance tests for geometric mean
    print("\nStep 5b: Testing statistical significance for geometric mean...")
    geo_significance = statistical_significance_test(df, "geo_avg_score")
    
    # Generate report
    print("\nStep 6: Generating report...")
    generate_report(config_results, factor_results, arith_significance, geo_significance, START_QUESTION_ID, END_QUESTION_ID)
    
    # Print final recommendation
    best_config_arith = config_results["arithmetic"][0]
    best_config_geo = config_results["geometric"][0]
    
    print("\n" + "=" * 80)
    print(f"FINAL RECOMMENDATIONS:")
    print("=" * 80)
    
    print(f"\nBest by Arithmetic Mean: {best_config_arith['method']}, k={best_config_arith['k']}, {best_config_arith['version']} (Score: {best_config_arith['mean']:.4f})")
    print(f"Best by Geometric Mean: {best_config_geo['method']}, k={best_config_geo['k']}, {best_config_geo['version']} (Score: {best_config_geo['mean']:.4f})")
    
    if (best_config_arith['method'] == best_config_geo['method'] and 
        best_config_arith['k'] == best_config_geo['k'] and 
        best_config_arith['version'] == best_config_geo['version']):
        print("\nThe same configuration performs best on both metrics!")
    else:
        print("\nDifferent configurations perform best on different metrics.")
        print("For RAG systems, we recommend prioritizing the geometric mean results since it better captures consistent performance across all questions.")
    
    print(f"\nReport file: bootstrap_report_Q{START_QUESTION_ID:03d}_Q{END_QUESTION_ID:03d}.md")

if __name__ == "__main__":
    main()
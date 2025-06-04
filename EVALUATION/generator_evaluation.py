import json
import logging
import re
import time
import os
from langchain_ollama import OllamaLLM
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor


# Configuration
CHUNK_TYPE = "filtered_chunks"      ## filtered_chunks    or Normal_retrieved_chunks
START_INDEX = 301
END_INDEX = 400
BATCH_NUMBER = 1    

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
# Keep httpx quiet
logging.getLogger("httpx").setLevel(logging.ERROR)

input_filename = f"gen_answers_filtered_chunks_ret_results_filtered_Hybrid_15.json"
 
# Custom evaluation prompts
FAITHFULNESS_PROMPT = """You are evaluating whether an answer is faithful to the provided context. 
Rate the answer on a scale of 0-3:

0 - COMPLETELY UNFAITHFUL: The answer contains multiple claims that contradict or are not supported by the context
1 - MOSTLY UNFAITHFUL: The answer contains more unsupported claims than supported ones
2 - MOSTLY FAITHFUL: The answer contains more supported claims than unsupported ones
3 - COMPLETELY FAITHFUL: All claims in the answer are fully supported by the context

Context: {context}
Question: {query}
Answer: {response}

Provide your evaluation in this exact format:
FEEDBACK: [Your detailed analysis explaining your reasoning]
SCORE: [number 0-3]
"""

COMPLETENESS_PROMPT = """You are evaluating how completely an answer addresses a question based on the provided context.
Rate the answer on a scale of 0-3:

0 - COMPLETELY INCOMPLETE: The answer is missing almost all key information that would be needed to address the question
1 - MOSTLY INCOMPLETE: The answer addresses less than half of the key points from the context
2 - MOSTLY COMPLETE: The answer addresses more than half but not all key points from the context
3 - FULLY COMPLETE: The answer addresses all from the context

Context: {context}
Question: {query}
Answer: {response}

Provide your evaluation in this exact format:
FEEDBACK: [Your detailed analysis explaining your reasoning]
SCORE: [number 0-3]
"""

RELEVANCY_PROMPT = """You are evaluating how relevant an answer is to the question asked.
Rate the answer on a scale of 0-3:

0 - COMPLETELY IRRELEVANT: The answer doesn't address the question at all or is completely off-topic
1 - MOSTLY IRRELEVANT: The answer touches on the topic but fails to address the main question
2 - MOSTLY RELEVANT: The answer addresses the main question but includes some irrelevant information
3 - PERFECTLY RELEVANT: The answer directly and precisely addresses the question with no irrelevant information

Question: {query}
Answer: {response}

Provide your evaluation in this exact format:
FEEDBACK: [Your detailed analysis explaining your reasoning]
SCORE: [number 0-3]
"""

def get_processed_question_ids(output_filename):
    """Get set of question_ids that have already been processed"""
    processed_ids = set()
    
    try:
        if os.path.exists(output_filename):
            with open(output_filename, "r") as f:
                existing_results = json.load(f)
                
                # Extract question_ids from the results
                for result in existing_results.get("results", []):
                    question_id = result.get("question_id")
                    if question_id:  # Only add non-empty IDs
                        processed_ids.add(str(question_id))  # Convert to string for consistent comparison
                        
            print(f"Found {len(processed_ids)} already processed question IDs")
            if processed_ids:
                print(f"Example IDs: {list(processed_ids)[:5]}")
        else:
            print(f"No existing output file found at {output_filename}")
    except Exception as e:
        print(f"Error loading processed question IDs: {e}")
    
    return processed_ids

def extract_score_and_feedback(text: str) -> tuple:
    """Extract score and feedback from LLM output text."""
    # Extract the score using regex
    score_match = re.search(r'SCORE:\s*(\d+)', text, re.IGNORECASE)
    score = 0
    
    if score_match:
        score = int(score_match.group(1))
        # Make sure the score is within our 0-3 range
        score = max(0, min(score, 3))
    else:
        # Try to find any number between 0-3 in the text
        number_matches = re.findall(r'\b([0-3])\b', text)
        if number_matches:
            score = int(number_matches[-1])  # Take the last number as the score
    
    # Extract the feedback using regex
    feedback_match = re.search(r'FEEDBACK:\s*(.*?)(?=SCORE:|$)', text, re.IGNORECASE | re.DOTALL)
    
    if feedback_match:
        feedback = feedback_match.group(1).strip()
    else:
        # If no FEEDBACK label, use everything except the score part
        if score_match:
            feedback = text[:score_match.start()].strip()
        else:
            feedback = text.strip()
    
    return score, feedback

def extract_context(item):
    """Extract context from chunk_texts structure"""
    context = ""
    for chunk in item.get("chunk_texts", []):
        for preview in chunk.get("text_previews", []):
            # Handle if preview is a list
            if isinstance(preview, list):
                # Join the list items into a string
                preview_str = " ".join(str(item) for item in preview)
                context += preview_str + " "
            else:
                context += str(preview) + " "
    return context

def get_score_label(score: int, metric: str) -> str:
    """Get the descriptive label for a score based on the metric."""
    if metric == "faithfulness":
        labels = ["COMPLETELY UNFAITHFUL", "MOSTLY UNFAITHFUL", "MOSTLY FAITHFUL", "COMPLETELY FAITHFUL"]
    elif metric == "completeness":
        labels = ["COMPLETELY INCOMPLETE", "MOSTLY INCOMPLETE", "MOSTLY COMPLETE", "FULLY COMPLETE"]
    elif metric == "relevancy":
        labels = ["COMPLETELY IRRELEVANT", "MOSTLY IRRELEVANT", "MOSTLY RELEVANT", "PERFECTLY RELEVANT"]
    else:
        return "UNKNOWN"
    
    # Convert the score to an index (make sure it's within range)
    index = max(0, min(int(score), 3))
    return labels[index]

def evaluate_with_custom_prompt(item: Dict, ollama, prompt_template: str, metric_name: str) -> Dict:
    """Evaluate a single answer using a custom prompt template."""
    question = item["question"]
    answer = item["answer"]
    question_id = item.get("question_id", "")  # Get question_id if available
    context = extract_context(item)
    
    try:
        # Format the prompt
        prompt = prompt_template.format(
            context=context,
            query=question,
            response=answer
        )
        
        # Get evaluation from LLM
        response = ollama.invoke(prompt)
        
        # Extract score and feedback
        score, feedback = extract_score_and_feedback(response)
        
        # Get the label for this score
        label = get_score_label(score, metric_name)
        
        return {
            f"{metric_name}_score": score,
            f"{metric_name}_label": label,
            f"{metric_name}_feedback": feedback
        }
        
    except Exception as e:
        logging.exception(f"Error evaluating {metric_name}: {e}")
        return {
            f"{metric_name}_score": 0,
            f"{metric_name}_label": get_score_label(0, metric_name),
            f"{metric_name}_feedback": f"Error: {str(e)}"
        }

def evaluate_model_for_question(model_tuple, item, metric_name, prompt_template):
    """Evaluate a question with a specific model and metric."""
    model_name, evaluator = model_tuple
    try:
        evaluation = evaluate_with_custom_prompt(item, evaluator, prompt_template, metric_name)
        
        # Extract score and feedback based on metric name
        score = evaluation.get(f"{metric_name}_score", 0)
        label = evaluation.get(f"{metric_name}_label", "")
        feedback = evaluation.get(f"{metric_name}_feedback", "")
        
        print(f"  {model_name}: Score {score}/3 - {label}")
        return model_name, {"score": score, "label": label, "feedback": feedback}
    except Exception as e:
        logging.exception(f"Error in {model_name} evaluation: {e}")
        print(f"  Error in {model_name} evaluation: {e}")
        return model_name, {"score": 0, "label": get_score_label(0, metric_name), "feedback": f"Error: {str(e)}"}

def evaluate_metric_with_voting(item, metric_name, prompt_template):
    """Evaluate a single question for a single metric using three models."""
    evaluators = [
        ("Mistral", OllamaLLM(model="mistral:latest", base_url="http://localhost:11434")),
        ("gemma", OllamaLLM(model="gemma:latest", base_url="http://localhost:11434")),
        ("Phi3", OllamaLLM(model="phi3:latest", base_url="http://localhost:11434"))
    ]
    
    print(f"\n  Evaluating {metric_name}...")
    
    model_evaluations = {}
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        eval_futures = [
            executor.submit(evaluate_model_for_question, evaluator, item, metric_name, prompt_template) 
            for evaluator in evaluators
        ]
        for future in eval_futures:
            model_name, eval_data = future.result()
            model_evaluations[model_name] = eval_data
    
    # Calculate the average score (for 0-3 scale)
    scores = [int(eval_data["score"]) for eval_data in model_evaluations.values()]
    avg_score = sum(scores) / len(scores) if scores else 0
    
    # Round to nearest integer for final score
    final_score = round(avg_score)
    final_label = get_score_label(final_score, metric_name)
    
    # Fixed the f-string issue with a multi-step approach
    scores_list = []
    for name, eval_data in model_evaluations.items():
        score_value = eval_data['score']
        scores_list.append(f"{name}: {score_value}")
    scores_str = ", ".join(scores_list)
    print(f"  Model scores: {scores_str}")
    
    print(f"  Average score: {avg_score:.1f} → Final score: {final_score}/3 - {final_label}")
    
    return {
        "model_evaluations": model_evaluations,
        "average_score": avg_score,
        "final_score": final_score,
        "final_label": final_label
    }

def evaluate_question_fully(item):
    """Evaluate a single question with all three metrics."""
    question_id = item.get("question_id", "")
    print(f"\nProcessing Question ID: {question_id}")
    print(f"Question: {item['question'][:100]}..." if len(item['question']) > 100 else f"Question: {item['question']}")
    print(f"Answer: {item['answer'][:100]}..." if len(item['answer']) > 100 else f"Answer: {item['answer']}")
    
    # Evaluate each metric
    faithfulness_result = evaluate_metric_with_voting(item, "faithfulness", FAITHFULNESS_PROMPT)
    completeness_result = evaluate_metric_with_voting(item, "completeness", COMPLETENESS_PROMPT)
    relevancy_result = evaluate_metric_with_voting(item, "relevancy", RELEVANCY_PROMPT)
    
    # Create the full result
    full_result = {
        "question_id": question_id,
        "batch_number": BATCH_NUMBER,
        "question": item["question"],
        "answer": item["answer"],
        "faithfulness": faithfulness_result, 
        "completeness": completeness_result,
        "relevancy": relevancy_result
    }
    
    return full_result

def save_question_result(result, results_filename):
    """Save a single question result to the output file."""
    try:
        # Load existing results file if it exists
        existing_output = {"metrics": {}, "results": []}
        if os.path.exists(results_filename):
            try:
                with open(results_filename, "r") as f:
                    existing_output = json.load(f)
            except Exception as e:
                print(f"Error loading existing results: {e}")
        
        # Add new result to existing results
        existing_output["results"].append(result)
        
        # Recalculate metrics
        all_results = existing_output["results"]
        
        # Extract metrics for calculation
        faith_results = [r["faithfulness"] for r in all_results]
        comp_results = [r["completeness"] for r in all_results]
        rel_results = [r["relevancy"] for r in all_results]
        
        # Calculate updated metrics
        total_questions = len(all_results)
        if total_questions > 0:
            # Average scores
            faith_avg = sum(item["average_score"] for item in faith_results) / total_questions
            comp_avg = sum(item["average_score"] for item in comp_results) / total_questions
            rel_avg = sum(item["average_score"] for item in rel_results) / total_questions
            
            # Score distributions
            faith_dist = [0, 0, 0, 0]  # 0, 1, 2, 3 scores
            comp_dist = [0, 0, 0, 0]
            rel_dist = [0, 0, 0, 0]
            
            for i in range(total_questions):
                faith_score = faith_results[i]["final_score"]
                comp_score = comp_results[i]["final_score"]
                rel_score = rel_results[i]["final_score"]
                
                faith_dist[faith_score] += 1
                comp_dist[comp_score] += 1
                rel_dist[rel_score] += 1
            
            metrics = {
                "faithfulness": {
                    "average_score": faith_avg,
                    "distribution": faith_dist,
                    "score_labels": ["COMPLETELY UNFAITHFUL", "MOSTLY UNFAITHFUL", "MOSTLY FAITHFUL", "COMPLETELY FAITHFUL"]
                },
                "completeness": {
                    "average_score": comp_avg,
                    "distribution": comp_dist,
                    "score_labels": ["COMPLETELY INCOMPLETE", "MOSTLY INCOMPLETE", "MOSTLY COMPLETE", "FULLY COMPLETE"]
                },
                "relevancy": {
                    "average_score": rel_avg,
                    "distribution": rel_dist,
                    "score_labels": ["COMPLETELY IRRELEVANT", "MOSTLY IRRELEVANT", "MOSTLY RELEVANT", "PERFECTLY RELEVANT"]
                }
            }
            
            existing_output["metrics"] = metrics
        
        # Save updated results
        with open(results_filename, "w") as f:
            json.dump(existing_output, f, indent=2)
            
        print(f"Result for Question ID {result['question_id']} saved to {results_filename}")
        
    except Exception as e:
        print(f"ERROR: Failed to save result: {e}")

def print_summary_metrics(results_filename):
    """Print summary metrics from the results file."""
    try:
        if os.path.exists(results_filename):
            with open(results_filename, "r") as f:
                results = json.load(f)
                
            metrics = results.get("metrics", {})
            total_questions = len(results.get("results", []))
            
            if metrics and total_questions > 0:
                print(f"\n=== Summary Metrics ({total_questions} questions) ===")
                
                f_avg = metrics.get("faithfulness", {}).get("average_score", 0)
                c_avg = metrics.get("completeness", {}).get("average_score", 0)
                r_avg = metrics.get("relevancy", {}).get("average_score", 0)
                
                print(f"Faithfulness Average Score: {f_avg:.1f}/3")
                print(f"Completeness Average Score: {c_avg:.1f}/3")
                print(f"Relevancy Average Score: {r_avg:.1f}/3")
                
                f_dist = metrics.get("faithfulness", {}).get("distribution", [0, 0, 0, 0])
                c_dist = metrics.get("completeness", {}).get("distribution", [0, 0, 0, 0])
                r_dist = metrics.get("relevancy", {}).get("distribution", [0, 0, 0, 0])
                
                print(f"\nFaithfulness Distribution:")
                print(f"  {f_dist[0]} COMPLETELY UNFAITHFUL")
                print(f"  {f_dist[1]} MOSTLY UNFAITHFUL")
                print(f"  {f_dist[2]} MOSTLY FAITHFUL")
                print(f"  {f_dist[3]} COMPLETELY FAITHFUL")
                
                print(f"\nCompleteness Distribution:")
                print(f"  {c_dist[0]} COMPLETELY INCOMPLETE")
                print(f"  {c_dist[1]} MOSTLY INCOMPLETE")
                print(f"  {c_dist[2]} MOSTLY COMPLETE")
                print(f"  {c_dist[3]} FULLY COMPLETE")
                
                print(f"\nRelevancy Distribution:")
                print(f"  {r_dist[0]} COMPLETELY IRRELEVANT")
                print(f"  {r_dist[1]} MOSTLY IRRELEVANT")
                print(f"  {r_dist[2]} MOSTLY RELEVANT")
                print(f"  {r_dist[3]} PERFECTLY RELEVANT")
    except Exception as e:
        print(f"Error printing metrics summary: {e}")

def main():
    # Define output filename
    results_filename = f"evaluation_results_{input_filename}"
    
    # Check for already processed question IDs
    processed_ids = get_processed_question_ids(results_filename)
    
    try:
        print(f"Loading file: {input_filename}")
        
        with open(input_filename, "r") as f:
            all_results = json.load(f)
        
        # Generate patterns for Q001 through Q050 (or whichever range is specified)
        start_num = START_INDEX
        end_num = END_INDEX
        
        print(f"Processing questions from Q{start_num:03d} to Q{end_num:03d} (Batch {BATCH_NUMBER})")
            
        # Filter by question_id pattern instead of array indices
        questions_to_process = []
        for item in all_results:
            question_id = item.get("question_id", "")
            
            # Try to normalize question ID for comparison
            str_question_id = str(question_id) if question_id else ""
            
            # Skip if question ID doesn't match our pattern or is already processed
            try:
                # Check if the ID is in the Q001-Q050 format
                if str_question_id.startswith('Q'):
                    q_num = int(str_question_id[1:])
                    if q_num < start_num or q_num > end_num:
                        continue  # Skip if outside our range
                else:
                    continue  # Skip if doesn't match our pattern
            except ValueError:
                continue  # Skip if not in the right format
            
            # Check if this question has already been processed
            if str_question_id in processed_ids:
                print(f"Skipping already processed question ID: {str_question_id}")
                continue
                
            print(f"Will process question ID: {str_question_id}")
                
            # Add batch number to the item
            item["batch_number"] = BATCH_NUMBER
            questions_to_process.append(item)
            
        print(f"Loaded {len(questions_to_process)} unprocessed questions with IDs between Q{start_num:03d}-Q{end_num:03d}")
        
        if not questions_to_process:
            print("No new questions to process. Exiting.")
            return
            
        print("\nEvaluation started... 🚀")
        
        # Process each question one by one
        for i, question in enumerate(questions_to_process):
            print(f"\n==== Processing Question {i+1}/{len(questions_to_process)} ====")
            
            # Fully evaluate this question
            result = evaluate_question_fully(question)
            
            # Save the result immediately
            save_question_result(result, results_filename)
            
            # Wait between questions (except for the last one)
            if i < len(questions_to_process) - 1:
                wait_time = 10  # Shorter wait time since we're processing question by question
                print(f"\nWaiting {wait_time} seconds before next question...")
                time.sleep(wait_time)
        
        # Print final summary
        print_summary_metrics(results_filename)
        print(f"\nAll questions processed. Results saved to {results_filename}")
            
    except Exception as e:
        print(f"Error in main process: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
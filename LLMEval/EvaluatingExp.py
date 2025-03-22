import json
import re
import logging
from typing import Dict, Optional
import ollama
import os
os.environ["OLLAMA_HOST"] = "192.168.23.138:11439"
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def load_json(filepath: str) -> Dict:
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        raise
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in file: {filepath}")
        raise
def extract_json_from_response(response_text: str) -> Optional[Dict]:
    try:
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)  
        if json_match:
            return json.loads(json_match.group())
        else:
            logger.error("No valid JSON found in response.")
            return None
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        return None

def generate_evaluation_prompt(ground_truth: Dict, test_procedure: Dict) -> str:
    return f"""You are an expert evaluator assessing the accuracy, completeness, clarity, and safety of an electronics experiment procedure. Your primary task is to compare the Test Procedure against the Ground Truth, which serves as the reference standard.

        Ground Truth Procedure
        {json.dumps(ground_truth, indent=2)}

        Test Procedure
        {json.dumps(test_procedure, indent=2)}

        Evaluation Criteria & Scoring Guidelines
        Evaluate the Test Procedure using the following dimensions, ensuring your assessment is tailored to the unique needs of electronics experiments.

        Accuracy (Correctness and Precision)
        Score 5: Near-perfect match with ground truth, including correct component values, circuit connections, and instrument settings (>95% accuracy).
        Score 4: Minor deviations that do not affect experimental outcomes (80-95%).
        Score 3: Some notable errors in component specifications, wiring, or settings that could impact results (60-80%).
        Score 2: Significant inaccuracies that compromise core experiment functionality (40-60%).
        Score 1: Major errors making the experiment unsafe or non-functional (<40%).

        Completeness (Coverage of Required Steps and Details)
        Score 5: All steps, components, and conditions are present and well-defined (>95%).
        Score 4: Minor omissions that don't impact experimental outcomes (80-95%).
        Score 3: Missing some important details, such as critical measurements, equipment setup, or observation steps (60-80%).
        Score 2: Major omissions affecting the experiment's integrity (40-60%).
        Score 1: Severely incomplete, omitting core steps, critical components, or observations (<40%).

        Clarity (Ease of Understanding and Reproducibility)
        Score 5: Clearer than the ground truth or equally clear, with detailed instructions, diagrams, and well-defined steps.
        Score 4: Mostly clear but may lack minor clarifications or formatting improvements.
        Score 3: Instructions are somewhat ambiguous, potentially requiring interpretation or additional guidance.
        Score 2: Unclear in several steps, risking procedural errors.
        Score 1: Highly confusing, disorganized, or poorly written.

        Safety (Adherence to Safety Protocols and Precautions)
        Score 5: Matches or exceeds ground truth safety measures, including proper ESD precautions, correct handling of power sources, and clear warnings for hazardous steps.
        Score 4: Contains most key safety measures but may lack minor details.
        Score 3: Missing some important safety points that pose moderate risk.
        Score 2: Inadequate safety guidance, posing potential hazards.
        Score 1: Critical safety issues or complete lack of precautionary steps.

        Important Electronics-Specific Considerations
        - Emphasize component values, polarity details, and circuit connections during evaluation.
        - Ensure steps involving voltage ratings, current limits, and instrument calibration are addressed.
        - Check for essential safety measures such as discharging capacitors, short-circuit prevention, and power isolation steps.
        - Highlight the presence of clear circuit diagrams, pin configurations, and proper labeling as part of clarity evaluation.

        Response Format (Strict JSON)
        Please provide your evaluation in the following format:
        {{
            "Accuracy": {{
                "1": probability,
                "2": probability,
                "3": probability,
                "4": probability,
                "5": probability
            }},
            "Completeness": {{
                "1": probability,
                "2": probability,
                "3": probability,
                "4": probability,
                "5": probability
            }},
            "Clarity": {{
                "1": probability,
                "2": probability,
                "3": probability,
                "4": probability,
                "5": probability
            }},
            "Safety": {{
                "1": probability,
                "2": probability,
                "3": probability,
                "4": probability,
                "5": probability
            }}
        }}

        Evaluation Guidance
        - Assign higher probabilities to scores 4 and 5 when the Test Procedure closely matches the Ground Truth.
        - Shift probabilities downward if errors, omissions, or clarity issues are significant.
        - Consider safety lapses with extra caution — even minor gaps may justify lower scores for safety.
        """

def evaluate_experiment(title: str, ground_truth: Dict, test_procedure: Dict) -> Optional[Dict]:
    # Extract the procedure from ground truth
    ground_truth_procedure = ground_truth.get("procedure", {})
    
    prompt = generate_evaluation_prompt(ground_truth_procedure, test_procedure)
    
    # Create a log entry
    log_entry = f"""
=== Evaluation Log for: {title} ===


=== Ground Truth Procedure ===
{json.dumps(ground_truth_procedure, indent=2)}

=== Test Procedure ===
{json.dumps(test_procedure, indent=2)}

=== Generated Prompt ===
{prompt}

"""
    
    try:
        response = ollama.chat(
            model="jacob-ebey/phi4-tools:latest",
            messages=[{"role": "user", "content": prompt}],
            options={"timeout": 60}
        )
        model_output = response['message']['content'].strip()
        
        # Append the model's response to the log entry
        log_entry += f"""
=== Model Response ===
{model_output}

=== Parsed JSON Result ===
{json.dumps(extract_json_from_response(model_output), indent=2)}

{"="*50}
"""
        
        # Write to log file
        with open("evaluation_logs.txt", "a", encoding="utf-8") as log_file:
            log_file.write(log_entry)
        
        print(f"\n🔹 **Raw Response for {title}:**\n{model_output}\n")
        return extract_json_from_response(model_output)

    except Exception as e:
        # Log the error as well
        log_entry += f"""
=== Error ===
{str(e)}

{"="*50}
"""
        with open("evaluation_logs.txt", "a", encoding="utf-8") as log_file:
            log_file.write(log_entry)
            
        logger.error(f"Error evaluating {title}: {e}")
        return None

def calculate_reward(evaluation_result: Dict) -> float:
    weights = {
        "Accuracy": 0.4,
        "Completeness": 0.15,
        "Clarity": 0.15,
        "Safety": 0.30,
    }

    if not isinstance(evaluation_result, dict):
        logger.error("Invalid evaluation result format. Expected a dictionary.")
        return 0

    total_reward = 0
    try:
        for category, scores in evaluation_result.items():
            if not isinstance(scores, dict):
                logger.error(f"Invalid score structure for category: {category}")
                continue

            weighted_score = sum(int(score) * prob for score, prob in scores.items())
            total_reward += weighted_score * weights.get(category, 0.25)

        return min(max(total_reward, 0), 5) 

    except Exception as e:
        logger.error(f"Reward calculation error: {e}")
        return 0



def run_evaluation(ground_truth_path: str, results_path: str, output_path: str):
    ground_truth_data = load_json(ground_truth_path)
    llm_generated_results = load_json(results_path)

    evaluation_results = {}
    rewards = {}
    
    # Use indexing to compare experiments
    for i in range(len(ground_truth_data["experiments"])):
        ground_truth_exp = ground_truth_data["experiments"][i]
        test_exp = llm_generated_results["experiments"][i]
        
        title = test_exp["title"]
        test_procedure = test_exp["procedure"]
        ground_truth_procedure = ground_truth_exp["procedure"]
        
        logger.info(f"Evaluating experiment {i+1}: {title}")
        # Add debug logging
        logger.info(f"Ground Truth Procedure: {ground_truth_procedure}")
        logger.info(f"Test Procedure: {test_procedure}")
        
        result = evaluate_experiment(title, ground_truth_exp, test_procedure)
        
        if result:
            evaluation_results[title] = result
            rewards[title] = calculate_reward(result)

    output_data = {
        "evaluations": evaluation_results,
        "rewards": rewards
    }

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(output_data, file, indent=4, ensure_ascii=False)

    logger.info(f"Evaluation complete! Results saved to '{output_path}'.")

run_evaluation(
    ground_truth_path="C:\\Users\\aahan\\Desktop\\college\\SEM 6\\IP\\Using LLM as an evaluator\\exp.json",
    results_path="C:\\Users\\aahan\\Desktop\\college\\SEM 6\\IP\\Using LLM as an evaluator\\results.json",
    output_path="C:\\Users\\aahan\\Desktop\\college\\SEM 6\\IP\\Using LLM as an evaluator\\evaluation_results_ns_rxp.json"
)

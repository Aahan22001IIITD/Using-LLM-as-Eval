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
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)  # Extracts only JSON part
        if json_match:
            return json.loads(json_match.group())
        else:
            logger.error("No valid JSON found in response.")
            return None
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        return None

def generate_evaluation_prompt(ground_truth: Dict, test_procedure: str) -> str:
    return f"""
You are an expert evaluator assessing the accuracy, completeness, clarity, and safety of a scientific experiment procedure.

### **Task**
Compare the **Test Procedure** against the **Ground Truth** and evaluate it across four key dimensions. Provide a **probability distribution over scores [1,2,3,4,5]** for each category.

### **Evaluation Criteria**
1. **Accuracy**: Are the components and steps correct?
2. **Completeness**: Are all critical steps present?
3. **Clarity**: Is the procedure easy to follow?
4. **Safety**: Are safety precautions included?

### **Response Format (Strict JSON)**
```json
{{
    "Accuracy": {{ "1": p1, "2": p2, "3": p3, "4": p4, "5": p5 }},
    "Completeness": {{ "1": p1, "2": p2, "3": p3, "4": p4, "5": p5 }},
    "Clarity": {{ "1": p1, "2": p2, "3": p3, "4": p4, "5": p5 }},
    "Safety": {{ "1": p1, "2": p2, "3": p3, "4": p4, "5": p5 }}
}}
Each probability value must be between 0 and 1 and sum to 1.

### **Ground Truth**
{json.dumps(ground_truth, indent=4)}

### **Test Procedure**
{test_procedure}
"""
def evaluate_experiment(title: str, ground_truth: Dict, test_procedure: str) -> Optional[Dict]:
    prompt = generate_evaluation_prompt(ground_truth, test_procedure)

    try:
        response = ollama.chat(
            model="qwen:7b",
            messages=[{"role": "user", "content": prompt}],
            options={"timeout": 60}
        )

        # Debug: Print the raw response to see what the model is returning
        model_output = response['message']['content'].strip()
        print(f"\n🔹 **Raw Response for {title}:**\n{model_output}\n")

        return extract_json_from_response(model_output)  # Extract only valid JSON

    except Exception as e:
        logger.error(f"Error evaluating {title}: {e}")
        return None

def calculate_reward(evaluation_result: Dict) -> float:
    """
    Calculate a reward based on evaluation scores.
    """
    weights = {
        "Accuracy": 0.3,
        "Completeness": 0.25,
        "Clarity": 0.25,
        "Safety": 0.2
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

        return min(max(total_reward, 0), 5)  # Normalize reward between 0-5

    except Exception as e:
        logger.error(f"Reward calculation error: {e}")
        return 0

def run_evaluation(ground_truth_path: str, results_path: str, output_path: str):
    ground_truth_data = load_json(ground_truth_path)
    llm_generated_results = load_json(results_path)
    ground_truth_dict = {exp["title"]: exp for exp in ground_truth_data["experiments"]}
    evaluation_results = {}
    rewards = {}
    for title, test_procedure in llm_generated_results.items():
        ground_truth = ground_truth_dict.get(title)

        if ground_truth:
            logger.info(f"Evaluating: {title}")
            result = evaluate_experiment(title, ground_truth, test_procedure)

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
    ground_truth_path="experiments.json",
    results_path="result.json",
    output_path="evaluation_results.json"
)
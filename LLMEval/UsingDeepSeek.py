import ollama
import json
import re
import os
model_name = "mistral"
os.environ["OLLAMA_HOST"] = "192.168.23.138:11439"

experiments = {
    "Ohm’s Law Verification:To verify the relationship between voltage, current, and resistance in an electrical circuit.":
    """Objective: To verify the relationship between voltage, current, and resistance in an electrical circuit.
    Materials:
    - DC power supply
    - Resistor of known value
    - Ammeter
    - Voltmeter
    - Connecting wires
    - Breadboard (optional)

    Procedure:
    1. Connect the resistor in series with the ammeter and power supply.
    2. Connect the voltmeter in parallel across the resistor.
    3. Set the power supply to a low voltage (e.g., 1V).
    4. Record the voltage (V) and the current (I).
    5. Increase the voltage in small increments and record V and I for each step.
    6. Plot a graph of V against I and verify that the slope equals resistance.""",

    "Kirchhoff’s Voltage and Current Laws (KVL and KCL): To apply KVL and KCL to analyze simple electrical circuits.":
    """Objective: To apply KVL and KCL to analyze simple electrical circuits.
    Materials:
    - DC power supply
    - Two resistors of known values
    - Ammeter
    - Voltmeter
    - Connecting wires
    - Breadboard (optional)

    Procedure:
    1. Construct a simple series-parallel circuit with the two resistors and the power supply.
    2. Measure and record the voltage across each resistor and the current through each branch.
    3. Verify KCL: Ensure that the sum of currents entering a junction equals the sum leaving it.
    4. Verify KVL: Ensure the sum of voltages around a closed loop equals zero."""
}

def getprompt(exp):
    test_prompt = """
    Provide a detailed and safe procedure for conducting the experiment: {exp}. Ensure the response includes the following details:

    1. Experiment Name: Clearly state the experiment's title.
    2. Objective: Describe the purpose of the experiment.
    3. Materials Required: List all necessary components, tools, and equipment.
    4. Circuit Diagram / Setup: Provide a clear explanation of the experimental setup (if applicable).
    5. Step-by-Step Procedure: Outline the ensure a risk-free execution.
    6. Expected Results: Describe the anticipated outcome.
    7. Conclusion: Summarize the findings and their significance.

    Ensure the procedure follows all necessary safety standards and is suitable for laboratory or academic use.
    """
    return test_prompt.format(exp=exp)

def runprompt(prompt):
    try:
        response = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}])
        model_output = response['message']['content']
        print("\nRaw Model Response:\n", model_output)
        return model_output
    except Exception as e:
        print("\nConnection Error: Could not connect to Ollama. Ensure it's running and accessible.")
        print("Try running: `ollama serve` in your terminal.")
        print("Error details:", str(e))
        return None

def handle_test(test_procedure, ground_truth):
    if test_procedure is None:
        return
    
    evaluation_prompt = f"""
    Evaluate the following test procedure against the ground truth based on these dimensions:

    1. **Accuracy**: Does the procedure use correct components and connections? Assign a probability distribution over scores [1,2,3,4,5].
    2. **Completeness**: Are all necessary steps included? Assign a probability distribution over scores [1,2,3,4,5].
    3. **Clarity**: Is the procedure logically structured? Assign a probability distribution over scores [1,2,3,4,5].
    4. **Safety**: Are there no unsafe or hazardous steps? Assign a probability distribution over scores [1,2,3,4,5].

    Provide a **clean JSON response** **only** in this format:
    {{
        "Accuracy": {{ "1": p1, "2": p2, "3": p3, "4": p4, "5": p5 }},
        "Completeness": {{ "1": p1, "2": p2, "3": p3, "4": p4, "5": p5 }},
        "Clarity": {{ "1": p1, "2": p2, "3": p3, "4": p4, "5": p5 }},
        "Safety": {{ "1": p1, "2": p2, "3": p3, "4": p4, "5": p5 }}
    }}

    ### Ground Truth:
    {ground_truth}

    ### Test Procedure:
    {test_procedure}
    """
    try:
        response = ollama.chat(model=model_name, messages=[{"role": "user", "content": evaluation_prompt}])
        model_output = response['message']['content']
        print("\nRaw Model Response:\n", model_output)

        # Extract JSON response
        json_match = re.search(r"\{.*\}", model_output, re.DOTALL)
        if json_match:
            json_data = json_match.group().strip()
            scores = json.loads(json_data)

            # Compute Weighted Scores
            final_scores = {}
            dimensions = ["Accuracy", "Completeness", "Clarity", "Safety"]

            for dim in dimensions:
                probabilities = scores.get(dim, {})
                weighted_score = sum(int(score) * prob for score, prob in probabilities.items())
                final_scores[dim] = round(weighted_score, 2)

            overall_score = round(sum(final_scores.values()) / len(final_scores), 2)

            results = {
                "Final Scores": final_scores,
                "Overall Score": overall_score
            }

            with open("evaluation_results.json", "w+", encoding="utf-8") as f:
                json.dump(results, f, indent=4)

            print("\nEvaluation Results Saved: evaluation_results.json")
            print("\nFinal Scores:", final_scores)
            print("Overall Score:", overall_score)

        else:
            print("\nError: JSON Block Not Found in LLM Response")

    except Exception as e:
        print("\nConnection Error: Could not connect to Ollama. Ensure it's running and accessible.")
        print("Try running: `ollama serve` in your terminal.")
        print("Error details:", str(e))

def run(experiments):
    for exp, ground_truth in experiments.items():
        test_prompt = getprompt(exp)
        test_procedure = runprompt(test_prompt)
        handle_test(test_procedure, ground_truth)

run(experiments)

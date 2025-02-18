## We will be creating a code to write the experiments , and then those will be paresed to another LLM to evaluate
import ollama
import json
import re
import os
model_name = "mistral"
os.environ["OLLAMA_HOST"] = "192.168.23.138:11439"
## We would be using mistral for generating experiments for now 
def getprompt(exp):
    print("Getting the Prompt for : "+exp)
    test_prompt = """
    Provide a detailed and safe procedure for conducting the experiment: {exp}. Ensure the response includes the following details:

    1. Experiment Name: Clearly state the experiment's title.
    2. Objective: Describe the purpose of the experiment.
    3. Materials Required: List all necessary components, tools, and equipment.
    4. Step-by-Step Procedure: Outline the ensure a risk-free execution.
    5. Expected Results: Describe the anticipated outcome.
    6. Conclusion: Summarize the findings and their significance.

    Ensure the procedure follows all necessary safety standards and is suitable for laboratory or academic use.
    """
    print("Got the prompt!")
    return test_prompt.format(exp=exp)
def enhancedprompt(exp):
    test_prompt = f"""
    Generate a **detailed, well-structured, and risk-free** experimental procedure for the experiment titled **"{exp}"**. 

    The response must include the following key details:

    **1. Experiment Name:** Clearly state the title of the experiment.

    **2. Objective:** Describe the purpose of the experiment concisely, explaining what is being tested and why it is important.

    **3. Materials Required:** List all necessary equipment, components, tools, and safety gear needed to conduct the experiment.

    **4. Safety Precautions:** Outline essential safety measures and best practices to ensure a risk-free and compliant execution.

    **5. Step-by-Step Procedure:** Provide a **clear, numbered**, and logically ordered sequence of steps, ensuring clarity and reproducibility. Include:
        - How to properly set up the experiment.
        - Any initial settings or calibrations.
        - Stepwise execution, including measurements or observations to record.
        - Conditions or constraints to be maintained.
    
    **6. Observations & Expected Results:** Explain what participants should observe during the experiment and describe the anticipated outcome based on theoretical principles.

    **7. Analysis & Conclusion:** Summarize key findings based on expected results and discuss their significance. Highlight:
        - How the experiment validates a scientific principle.
        - Any potential sources of error and how they can be minimized.
        - Practical applications or real-world relevance.

    **8. Additional Notes (if applicable):** Include any relevant troubleshooting tips, alternative methods, or follow-up experiments.

    Ensure the procedure is **well-formatted, easy to follow, scientifically sound, and suitable for laboratory or academic use.** 

    """
    return test_prompt.format(exp=exp)
def runprompt(prompt):
    try:    
        print("Running the prompt!")
        response = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}])
        model_output = response['message']['content']
        # print("\nRaw Model Response:\n", model_output)
        return model_output
    except Exception as e:
        print("\nConnection Error: Could not connect to Ollama. Ensure it's running and accessible.")
        print("Try running: `ollama serve` in your terminal.")
        print("Error details:", str(e))
        return None
with open("experiments.json", "r") as file:
    data = json.load(file)
exp_prompts= []
exp_outputs ={}
ind = 0
for i in data["experiments"]:
    print(i["title"])
    exp_prompts.append(getprompt(i["title"]))
    exp_outputs[i["title"]] = runprompt(exp_prompts[ind])
    ind+=1
print("Got the output , Putting in JSON!")
output_file = "LLMGeneratedProcedure.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(exp_outputs, f, indent=4, ensure_ascii=False)
print(f"Data successfully written to {output_file}")
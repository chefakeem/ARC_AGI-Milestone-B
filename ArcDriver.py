import json
import os
from pathlib import Path
import numpy as np

from ArcData import ArcData
from ArcProblem import ArcProblem
from ArcSet import ArcSet
from ArcAgent import ArcAgent

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Hard-code your Milestone C filenames here (include the .json extension).
# Example:
# MILESTONE_C_FILES = [
#     "0a1d4ef5.json", "1f876d4f.json", "2dc579da.json", ...
# ]
MILESTONE_C_FILES: list[str] = [
    "3de23699.json",
    "5c0a986e.json",
    "7b6016b9.json",
    "9af7a82c.json",
    "22eb0ac0.json",
    "25d487eb.json",
    "62c24649.json",
    "74dd1130.json",
    "3428a4f5.json",
    "b2862040.json",
    "bbc9ae5d.json",
    "cf98881b.json",
    "dc433765.json",
    "e98196ab.json",
    "f8a8fe49.json",
    "f35d900a.json",
]

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

def run_training_data(agent: ArcAgent, arc_problems: list[ArcProblem]) -> dict[ArcProblem, tuple[bool, list]]:
    """
    Run each training problem with the test output included so the agent can
    test if they are getting the correct response. (This is fine locally;
    do NOT rely on test outputs inside your ArcAgent.)
    """
    train_ans_dict: dict[ArcProblem, tuple[bool, list]] = {}
    for trn_problem in arc_problems:
        preds: list[np.ndarray] = agent.make_predictions(trn_problem)
        correct = False

        # Gradescope constraint reminder: ArcAgent must return <= 3 predictions.
        if len(preds) <= 3:
            gold = trn_problem.test_set().get_output_data().data()
            for prediction in preds:
                if np.array_equal(gold, prediction):
                    correct = True
                    break

        train_ans_dict[trn_problem] = (correct, preds)
    return train_ans_dict

def load_arc_problems(path: str, problem_files: list[str]) -> list[ArcProblem]:
    problems: list[ArcProblem] = []
    for problem_name in problem_files:
        file_path = os.path.join(path, problem_name)
        with open(file_path, "r", encoding="utf-8") as p:
            flat_data: dict[str, dict] = json.load(p)

        # Convert the data into ArcData (numpy.ndarray)
        trn_data: list[ArcSet] = []
        for dt in flat_data["train"]:
            d_input = ArcData(np.array(dt["input"]))
            d_output = ArcData(np.array(dt["output"]))
            trn_data.append(ArcSet(arc_input=d_input, arc_output=d_output))

        tst_sets: list[ArcSet] = []
        for tst in flat_data["test"]:
            t_input = ArcData(np.array(tst["input"]))
            t_output = ArcData(np.array(tst["output"]))
            tst_sets.append(ArcSet(arc_input=t_input, arc_output=t_output))

        # There is exactly one test set
        arc_problem = ArcProblem(problem_name[:-5], trn_data, tst_sets[0])
        problems.append(arc_problem)

    return problems

if __name__ == "__main__":
    # Windows-friendly path handling
    milestone_dir = os.path.join("Milestones", "C")

    # If you didn't hard-code, fall back to listing the folder (filters to .json)
    if not MILESTONE_C_FILES:
        MILESTONE_C_FILES = [f for f in os.listdir(milestone_dir) if f.lower().endswith(".json")]

    # Sort for deterministic order (optional but nice)
    MILESTONE_C_FILES = sorted(MILESTONE_C_FILES)

    # Load problems
    arc_milestone_problems: list[ArcProblem] = load_arc_problems(milestone_dir, MILESTONE_C_FILES)

    # Instantiate the agent once
    arc_agent: ArcAgent = ArcAgent()

    # Run and write CSV
    results = run_training_data(arc_agent, arc_milestone_problems)
    with open("Milestone_Results.csv", "w", encoding="utf-8") as milestone_file:
        milestone_file.write("Problem Name, Correct, Correct Answer, Prediction 1, Prediction 2, Prediction 3\n")
        for problem, (is_correct, predictions) in results.items():
            gold = problem.test_set().get_output_data().data().tolist()
            milestone_file.write(f'{problem.problem_name()},{is_correct},"{gold}",')
            # Up to 3 predictions; pad with blanks if fewer
            for idx in range(3):
                if idx < len(predictions):
                    milestone_file.write(f'"{predictions[idx].tolist()}"')
                else:
                    milestone_file.write('""')
                milestone_file.write("\n" if idx == 2 else ",")

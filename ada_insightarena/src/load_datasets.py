import pandas as pd
import json
import glob, os
from collections import defaultdict
from src.utils import check_and_fix_dataset

def analyze_data(data: pd.DataFrame) -> str:
        """Analyze the data and return a detailed summary of its structure."""
        summary = {
            "num_rows": len(data),
            "num_cols": len(data.columns),
            "column_summaries": {},
        }

        for col in data.columns:
            col_data = data[col]
            col_summary = {
                "dtype": str(col_data.dtype),
                "num_missing": int(col_data.isnull().sum()),
                "num_unique": int(col_data.nunique()),
                "sample_values": col_data.dropna().unique()[:3].tolist(),
            }

            if pd.api.types.is_numeric_dtype(col_data):
                col_summary.update(
                    {
                        "mean": float(col_data.mean()),
                        "std": float(col_data.std()),
                        "min": float(col_data.min()),
                        "max": float(col_data.max()),
                    }
                )
            elif pd.api.types.is_datetime64_any_dtype(col_data):
                col_summary.update(
                    {
                        "min_date": str(col_data.min()),
                        "max_date": str(col_data.max()),
                    }
                )
            elif pd.api.types.is_string_dtype(col_data):
                col_summary.update(
                    {"top_frequent_values": col_data.value_counts().head(3).to_dict()}
                )

            summary["column_summaries"][col] = col_summary
            return summary


def get_dataset(challenge="toy"):
    """
    returns dataset as a list of dictionaries containing questions, metadata, goal, persona, insights, and table
    """
    base_path = "data/jsons"
    base_data_path = "data/csvs"

    # get the challenge
    print("challenge", challenge)
    if challenge == "toy":
        id_list = list(range(64,67))
    elif challenge == "pilot":
        # id_list = list(range(70,75))
        id_list= [346,447,482,551,700]
    elif challenge == "full":
        id_list = list(range(1, 26)) + list(range(27, 101))
    else:
        raise ValueError(f"Challenge {challenge} not found")

    print(id_list)
    data_list = []
    for _id in id_list:
        meta_dict = json.load(open(f"{base_path}/{_id}/meta.json", "r"))
        goal_dict = json.load(open(f"{base_path}/{_id}/goal.json", "r"))
        question_list = json.load(open(f"{base_path}/{_id}/questions.json", "r"))
        # insight_dict = json.load(open(f"{base_path}/{_id}/insights.json", "r"))

        data_dict = {}
        data_dict["id"] = _id
        data_dict["questions"] = question_list
        data_dict["dataset_path"] = f"{base_data_path}/{_id}/data.csv"
        data_dict["meta"] = meta_dict
        data_dict["goal"] = goal_dict["goal"]
        data_dict["persona"] = goal_dict["persona"]
        # data_dict["insight"] = insight_dict["insight"]
        data_dict["table"] = check_and_fix_dataset(f"{base_data_path}/{_id}/data.csv")

        reference_files_path = []
        for help_data_file_name in meta_dict["help_data_file_names"]:
            reference_files_path.append(f"{base_data_path}/{_id}/{help_data_file_name}")

        data_dict["reference_files"] = reference_files_path

        data_list.append(data_dict)

    return data_list

def get_pattern(pattern):
    if pattern=="p0":
        pattern="pattern_0"
    elif pattern=="p1":
        pattern="pattern_1"
    elif pattern=="p2":
        pattern="pattern_2"
    elif pattern=="p3":
        pattern="pattern_3"
    elif pattern=="p4":
        pattern="pattern_4"
    elif pattern=="p5":
        pattern="pattern_5"
    elif pattern=="p6":
        pattern="pattern_6"
    elif pattern=="p7":
        pattern="pattern_7"
    elif pattern=="p8":
        pattern="pattern_8"
    elif pattern=="p9":
        pattern="pattern_9"
    elif pattern=="p10":
        pattern="pattern_10"
    else:
        print("Pattern not found:", pattern)
        pattern = None
    return pattern

def load_dataset_insightarena(directory: str):
    """
    Loading the dataset in AgentAda format
    """
    # pattern = os.path.join(directory, "*_*_patterns.json")
    file_paths = glob.glob(f"{directory}/jsons/*_patterns.json")
    # print(len(file_paths))

    data_list = []
    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # print(data.keys())
        try:
            answers_questions = data['answers']
            task_name = data['Analytics task'].lower().replace(" ", "_") #"_".join(os.path.basename(file_path).split("_")[:2])
            # print(task_name)
            grouped = defaultdict(list)
            for item in answers_questions:
                pattern = item["caused_by_pattern"]
                grouped[pattern].append({
                    "question": item["question"],
                    "answer_after_injection": item["answer_after_injection"]
                })
            # print(grouped)
            answers_questions = grouped
            for aq in answers_questions:
                data_dict = {}
                # print(aq)
                pattern =  aq
                pattern = pattern.lower().replace(" ", "_")
                pattern = get_pattern(pattern)
                if pattern is None:
                    continue
                data_dict['meta'] = {"dataset_description": ""}
                data_dict['goal'] = f"The analysis aims to uncover {data['Analytics task']} insights from the given dataset"
                data_dict['persona']  = "You are a data analysis agent interesting in leveraging these insights in the real world"
                questions, answers = [], []
                for ques_ans in answers_questions[aq]:
                    questions.append({"question":ques_ans['question'], "task":data['Analytics task']})
                    answers.append(ques_ans['answer_after_injection'])
                # print(questions, answers)
                data_dict['questions'] = questions
                data_dict['answers'] = answers
                data_dict["dataset_path"] = f"{directory}/csvs/{task_name}_{pattern}_injected.csv"
                if not os.path.isfile(data_dict["dataset_path"]):
                    print(f'File not present: {data_dict["dataset_path"]}')
                    continue
                data_dict["table"] = check_and_fix_dataset(data_dict["dataset_path"])
                data_dict["id"] = f"{task_name}_{pattern}"
                data_list.append(data_dict)
        except:
            print(f"{file_path} not properly formatted")
            continue

    return data_list
import json
import os
import gym
import numpy as np
import re
import string
from collections import Counter


TWOWIKI_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
)

TWOWIKI_SPLIT_FILE = {
    "dev": "dev.json",
    "test": "test.json",
}


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(str(s)))))


def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    zero_metric = (0, 0, 0)

    if normalized_prediction in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
        return zero_metric
    if normalized_ground_truth in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
        return zero_metric

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return zero_metric

    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1, precision, recall


class TwoWikiQAWrapper(gym.Wrapper):
    def __init__(self, env, split="dev"):
        super().__init__(env)

        if split not in TWOWIKI_SPLIT_FILE:
            raise ValueError(f"Invalid split: {split}")

        data_file = os.path.join(TWOWIKI_DATA_DIR, TWOWIKI_SPLIT_FILE[split])

        with open(data_file, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        self.data = []
        for idx, d in enumerate(raw_data):
            self.data.append({
                "id": d.get("_id", d.get("id", idx)),
                "question": d["question"],
                "answer": d["answer"],
                "supporting_facts": d.get("supporting_facts", []),
                "evidences": d.get("evidences", []),
            })

        self.data_idx = 0
        self.split = split

    def reset(self, seed=None, return_info=False, options=None, idx=None):
        self.env.reset(seed=seed, return_info=return_info, options=options)
        self.data_idx = int(np.random.randint(len(self.data))) if idx is None else idx

        question = self.data[self.data_idx]["question"]
        observation = f"Question: {question}"
        info = self._get_info()

        return (observation, info) if return_info else observation

    def _get_info(self):
        item = self.data[self.data_idx]
        return {
            "steps": self.steps,
            "answer": self.answer,
            "question": item["question"],
            "question_id": item["id"],
            "split": self.split,
        }

    def get_metrics(self, pred_answer):
        gold_answer = self.data[self.data_idx]["answer"]

        pred = normalize_answer(pred_answer)
        gold = normalize_answer(gold_answer)

        em = pred == gold
        f1 = f1_score(pred, gold)[0]

        return {
            "reward": int(em),
            "em": em,
            "f1": f1,
        }

    def step(self, action):
        obs, _, done, info = self.env.step(action)
        reward = 0

        if done:
            pred_answer = info.get("answer", "")
            metrics = self.get_metrics(pred_answer)
            reward = metrics["reward"]

            item = self.data[self.data_idx]
            info.update({
                "gt_answer": item["answer"],
                "question_idx": self.data_idx,
                "supporting_facts": item["supporting_facts"],
                "evidences": item["evidences"],
                **metrics,
            })

            obs = f"Episode finished, reward = {reward}\n"

        return obs, reward, done, info

    def __len__(self):
        return len(self.data)


class LoggingWrapper(gym.Wrapper):
    def __init__(self, env, folder="trajs", file_id=None):
        super().__init__(env)

        self.trajs = []
        self.traj = {"observations": [], "actions": []}
        self.folder = folder
        self.file_id = np.random.randint(0, 10000000) if file_id is None else file_id
        self.file_path = os.path.join(self.folder, f"{self.file_id}.json")

        os.makedirs(self.folder, exist_ok=True)

    def __len__(self):
        return len(self.env)

    def reset(self, seed=None, return_info=False, options=None, idx=None):
        output = self.env.reset(
            seed=seed,
            return_info=return_info,
            options=options,
            idx=idx,
        )

        observation = output[0] if return_info else output
        self.traj = {"observations": [observation], "actions": []}

        return output

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        self.traj["observations"].append(obs)
        self.traj["actions"].append(action)

        if done:
            self.traj.update(info)

        return obs, reward, done, info

    def update_record(self):
        if len(self.traj) > 0:
            self.trajs.append(self.traj)
            self.traj = {"observations": [], "actions": []}

    def write(self):
        self.update_record()

        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(self.trajs, f, indent=2, ensure_ascii=False)

        print(f"Saved trajs to {self.file_path}")

    def close(self):
        self.write()
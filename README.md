# 2026 NLP Assignment: Multi-Agent QA System

## Overview

In this assignment, you will build a **multi-agent question answering system** for the **2WikiMultiHopQA** benchmark using LLM-based agents.

You will:
- implement a multi-agent reasoning pipeline,
- design your own agent roles and prompts,
- perform ablation studies,
- compare against a single-agent baseline.

The provided scaffold already includes:
- dataset loading,
- Wikipedia environment,
- evaluation pipeline,
- inference wrapper,
- result saving.

Your main task is to design the **multi-agent interaction system**.

---

# Environment Setup

Run the following commands:

```bash
bash script/setup.sh
conda activate Agent
```

---

# vLLM Server

We deploy a shared vLLM inference server using:

- Llama-3.2-3B-Instruct
- vLLM OpenAI-compatible API

Use the following server URLs:

- Student IDs ending with `1,3,5,7,9`

```text
http://165.132.142.56:8005/v1
```

- Student IDs ending with `0,2,4,6,8`

```text
http://165.132.142.56:8006/v1
```

You must update:

```yaml
conf.d/agent.yaml
```

Example:

```yaml
agent_config:
  model: "/home/kjy/nlp/models/Llama-3.2-3B-Instruct"
  base_url: "http://165.132.142.56:8005/v1"
  temperature: 0.0
  top_p: 0.9
  repetition_penalty: 1.0
  max_new_tokens: 1000
  num_samples: 1
  api_key: "EMPTY"
```

> IMPORTANT:
> Use `"EMPTY"` as the API key.

You must run the assignment:
- inside Yonsei University network, or
- using Yonsei VPN.

---

# File Structure

```text
2026ASSIGNMENT-STUDENT/
├── conf.d/
│   ├── 2wikiqa.yaml
│   └── agent.yaml
│
├── data/
│   └── dev.json
│
├── script/
│   ├── setup.sh
│   └── run_2wikiqa.sh
│
├── src/
│   ├── agent/
│   │   ├── prompts/
│   │   │   ├── construct_message.py
│   │   │   └── prompt.py
│   │   └── agents.py
│   │
│   ├── make_csv.py
│   ├── run_2wikiqa.py
│   ├── utils.py
│   ├── wikienv.py
│   └── wrappers.py
│
└── requirements.txt
```

---

# Your Tasks

You must implement:

## 1. Multi-Agent Prompt Design

Edit:

```text
src/agent/prompts/prompt.py
```

You are required to:
- implement the required `Decomposer Agent`,
- design **2–4 additional agents**,
- define prompts for each agent,
- register them in `PROMPT_REGISTRY`.

Your system must use:
- minimum: 3 agents
- maximum: 5 agents

including the Decomposer Agent.

---

## 2. Message Construction

Edit:

```text
src/agent/prompts/construct_message.py
```

You may customize:
- how agents communicate,
- what context each agent receives,
- how evidence is shared,
- message formatting strategies.

---

## 3. Multi-Agent Pipeline

Edit:

```text
src/run_2wikiqa.py
```

You should:
- define your agent roles,
- configure agent execution order,
- implement ablation settings,
- improve multi-agent interaction.

The overall scaffold is already provided.

---

# Running Experiments

## Single-Agent Baseline

```bash
bash script/run_2wikiqa.sh single_agent outputs/2wiki_predictions_single.json
```

## Multi-Agent System

```bash
bash script/run_2wikiqa.sh multi_agent_structured outputs/2wiki_predictions_multi.json
```

---

# Ablation Studies

You must implement at least **2 ablation settings**.

Example:
- without decomposer
- without agent1
- without agent2
- without agent3

Example execution:

```bash
bash script/run_2wikiqa.sh without_agent_2 outputs/ablation_without_agent2.json
```

---

# CSV Generation

Generate a CSV summary using:

```bash
python src/make_csv.py   --single outputs/2wiki_predictions_single.json   --multi outputs/2wiki_predictions_multi.json   --output outputs/results.csv
```

Example CSV:

```csv
setting,split,em,f1
single_agent,dev,58.40,71.10
multi_agent,dev,64.20,76.85
```

---

# Submission

Submit:

```text
1. Source code
2. Output JSON files
3. CSV results
4. Report PDF
```

Your report should include:
- agent role descriptions,
- prompt design choices,
- interaction strategy,
- ablation analysis,
- performance comparison,
- failure case analysis.

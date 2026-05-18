import argparse
import json
import os
import random
import sys
import time
from datetime import datetime

import numpy as np
import yaml

# Bump recursion limit. Some Wikipedia responses trigger deep recursion in
# Python 3.10's email header parser (urllib3 → http.client → email/feedparser),
# which can crash a long run on a single bad page. Defense in depth alongside
# the broad except inside wikienv.search_step.
sys.setrecursionlimit(5000)

from wikienv import WikiEnv
from wrappers import TwoWikiQAWrapper, LoggingWrapper
from agent.agents import SingleAgent, RoleAgent
from utils import (
    parse_json_response,
    extract_answer,
    substitute_placeholders,
    compose_final_answer,
)


# Maximum Actor-Reflector iterations per question.
# `without_reflection_loop` ablation can pin this to 1.
MAX_ACTOR_REFLECTOR_ITERS = 2

# How many characters of each Wikipedia observation to inject into the prompt.
# 300 chars covers the infobox header (Born/Died/Nationality) + first 1-2
# sentences, which is enough for attribute extraction. Kept tight because
# the school's vLLM is configured with max_model_len=2048.
EVIDENCE_TRUNCATE_CHARS = 300


# -----------------------------------------------------------------------------
# TODO FOR STUDENTS
# -----------------------------------------------------------------------------
# Design your own multi-agent system with 3 to 5 agents.
#
# The Decomposer Agent is required by the assignment and is provided as a
# scaffold. You should add 2 to 4 additional agents by:
#   1. writing their prompts in src/agent/prompts/prompt.py,
#   2. registering them in PROMPT_REGISTRY,
#   3. listing their prompt names below in STUDENT_AGENT_ROLES.
#
# Example role designs:
#   - planner / retriever / verifier / composer
#   - query_generator / evidence_selector / answer_writer
#   - search_agent / critic / reflector / finalizer
#
# Do not use more than 5 total agents including decomposer.
# -----------------------------------------------------------------------------
STUDENT_AGENT_ROLES = [
    "actor_query",   # Phase 1: produce Wikipedia search queries (ReAct + IRCoT)
    "actor_answer",  # Phase 2: derive intermediate answers from retrieved evidence
    "reflector",     # Phase 3: Self-RAG-style ISREL/ISSUP verification
]


# -----------------------------------------------------------------------------
# Ablation settings. Each entry maps a mode flag to a list of agent role
# names that should be skipped at runtime.
# -----------------------------------------------------------------------------
ABLATION_DISABLED_AGENTS = {
    "without_decomposer": ["decomposer"],
    "without_reflector":  ["reflector"],
}


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_task_config(raw_config):
    if raw_config is None:
        return {}
    return raw_config.get("task", raw_config)


def redact_agent_config(agent_config):
    safe_config = dict(agent_config)
    if "api_key" in safe_config:
        safe_config["api_key"] = "<redacted>"
    return safe_config


def build_task(question, trajectories, task_config, role=None, system_prompt=None):
    """Build a task dict for an agent.

    `few_shot_examples` in task_config may be either:
      - a dict keyed by role (preferred new format), or
      - a list (legacy single shared format).
    When `role` is given and few_shot_examples is a dict, examples for
    that role are selected. Otherwise the whole structure (or None) is
    passed through.
    """
    fs = task_config.get("few_shot_examples", None)
    if isinstance(fs, dict):
        examples = fs.get(role) if role else None
    else:
        examples = fs
    return {
        "task_type": task_config.get("task_type", "2wiki_qa"),
        "system_prompt": system_prompt,
        "few_shot_examples": examples,
        "input_info": {
            "question": question,
            "trajectories": trajectories,
        },
    }


def as_list(value, fallback):
    if isinstance(value, list) and value:
        return value
    return fallback


def get_queries(agent_output, fallback):
    """Extract search queries from an agent output.

    Agents are encouraged to return JSON such as:
        {"queries": ["...", "..."]}

    If no valid queries are found, the fallback is used.
    """
    queries = agent_output.get("queries", [])
    if not isinstance(queries, list):
        return fallback
    queries = [str(query).strip() for query in queries if str(query).strip()]
    return queries if queries else fallback


def search_wikipedia(env, queries, max_queries=4):
    evidence = []
    for query in queries[:max_queries]:
        obs, _, done, _ = env.step(f"search[{query}]")
        evidence.append({
            "query": query,
            "observation": obs,
        })
        if done:
            break
    return evidence


def format_evidence(evidence):
    if not evidence:
        return "No evidence retrieved."
    return "\n".join(
        f"- Query: {item['query']}\n  Observation: {item['observation']}"
        for item in evidence
    )


def run_single_agent(question, agent, task_config):
    task = build_task(
        question,
        [f"Question: {question}"],
        task_config,
        role="single_agent",
    )
    response = agent.get_response(task)[0]
    return {
        "final_answer": extract_answer(response),
        "raw_response": response,
    }


def _format_subquestions(subquestions):
    return "\n".join(f'  {i + 1}. "{sq}"' for i, sq in enumerate(subquestions))


def _format_prior_evidence(evidence):
    if not evidence:
        return "(none)"
    lines = []
    for ev in evidence:
        obs = str(ev.get("observation", ""))[:EVIDENCE_TRUNCATE_CHARS]
        lines.append(f"  - Query '{ev.get('query', '')}': {obs}")
    return "\n".join(lines)


def _format_prior_intermediate(intermediate_answers):
    if not intermediate_answers:
        return "(none)"
    lines = []
    for i, ia in enumerate(intermediate_answers, 1):
        ans = ia.get("answer")
        ans_str = f'"{ans}"' if ans is not None else "null"
        lines.append(f"  - sub_q_{i}: {ans_str}")
    return "\n".join(lines)


def _build_actor_query_context(question, pattern, subquestions, evidence, intermediate_answers):
    return (
        f'Original question: "{question}"\n'
        f"Reasoning pattern: {pattern}\n"
        f"Sub-questions:\n{_format_subquestions(subquestions)}\n"
        f"Prior evidence:\n{_format_prior_evidence(evidence)}\n"
        f"Prior intermediate answers:\n{_format_prior_intermediate(intermediate_answers)}"
    )


def _build_actor_answer_context(question, pattern, subquestions, evidence):
    return (
        f'Original question: "{question}"\n'
        f"Reasoning pattern: {pattern}\n"
        f"Sub-questions:\n{_format_subquestions(subquestions)}\n"
        f"Retrieved evidence:\n{_format_prior_evidence(evidence)}"
    )


def _build_reflector_context(question, subquestions, evidence, intermediate_answers):
    lines = [
        f'Original question: "{question}"',
        f"Sub-questions:\n{_format_subquestions(subquestions)}",
        f"Retrieved evidence:\n{_format_prior_evidence(evidence)}",
        "Actor's intermediate answers:",
    ]
    for i, ia in enumerate(intermediate_answers or [], 1):
        ans = ia.get("answer")
        ans_str = f'"{ans}"' if ans is not None else "null"
        quote = str(ia.get("evidence_quote", ""))[:120]
        lines.append(f'  - sub_q_{i}: {ans_str} (quote: "{quote}")')
    return "\n".join(lines)


def run_multi_agent(question, agents, task_config, env, mode="multi_agent_structured"):
    """Phase 1 multi-agent pipeline.

    Architecture
    ------------
      Decomposer (once)
        ↓
      Loop up to MAX_ACTOR_REFLECTOR_ITERS:
        a) substitute_placeholders(sub-questions, prior intermediate answers)
        b) Actor_Query  → produce Wikipedia queries
        c) WikiEnv.search → accumulate evidence
        d) Actor_Answer → update intermediate_answers + preliminary_answer
        e) Reflector    → sufficient? + feedback
        f) break if sufficient=true OR Actor_Query returns no queries
        ↓
      compose_final_answer (deterministic Python aggregator)

    Ablation modes
    --------------
      `without_decomposer`        → skip Decomposer, treat original Q as the only sub-q.
      `without_reflector`         → run a single Actor pass (no verification loop).
      `multi_agent_without_structured_prompting` → all agents use the unstructured
                                     prompt; outputs fall through to extract_answer fallback.
    """
    disabled = set(ABLATION_DISABLED_AGENTS.get(mode, []))

    raw_outputs = {
        "decomposer": None,
        "actor_query": [],
        "actor_answer": [],
        "reflector": [],
    }
    retrieved_evidence = []
    intermediate_answers = []
    preliminary_answer = None
    reflector_flags = []  # [(iteration_idx, sufficient_bool_or_None)]
    iteration_history = []

    # -- Step 1: Decomposer ----------------------------------------------------
    if "decomposer" not in disabled:
        decomp_task = build_task(
            question,
            [f"Original question: {question}"],
            task_config,
            role="decomposer",
        )
        decomp_resp = agents["decomposer"].get_response(decomp_task)[0]
        raw_outputs["decomposer"] = decomp_resp
        decomp_json = parse_json_response(decomp_resp, default={})
        subquestions = as_list(decomp_json.get("subquestions"), [question])
        reasoning_pattern = str(decomp_json.get("reasoning_pattern", "unknown"))
    else:
        decomp_json = {"reasoning_pattern": "unknown", "subquestions": [question]}
        subquestions = [question]
        reasoning_pattern = "unknown"

    # -- Step 2: Actor-Reflector loop ------------------------------------------
    max_iter = 1 if "reflector" in disabled else MAX_ACTOR_REFLECTOR_ITERS

    for iteration in range(max_iter):
        # 2a. Substitute placeholders using whatever has been resolved so far
        sq_resolved = substitute_placeholders(subquestions, intermediate_answers)

        # 2b. Actor_Query
        aq_context = _build_actor_query_context(
            question, reasoning_pattern, sq_resolved,
            retrieved_evidence, intermediate_answers,
        )
        aq_task = build_task(question, [aq_context], task_config, role="actor_query")
        aq_resp = agents["actor_query"].get_response(aq_task)[0]
        raw_outputs["actor_query"].append(aq_resp)
        aq_json = parse_json_response(aq_resp, default={})
        queries = get_queries(aq_json, [])

        # 2c. Search Wikipedia (skip if no queries — nothing left to retrieve)
        if queries:
            new_evidence = search_wikipedia(env, queries)
            retrieved_evidence.extend(new_evidence)

        # 2d. Actor_Answer
        aa_context = _build_actor_answer_context(
            question, reasoning_pattern, sq_resolved, retrieved_evidence,
        )
        aa_task = build_task(question, [aa_context], task_config, role="actor_answer")
        aa_resp = agents["actor_answer"].get_response(aa_task)[0]
        raw_outputs["actor_answer"].append(aa_resp)
        aa_json = parse_json_response(aa_resp, default={})
        new_ia = aa_json.get("intermediate_answers")
        if isinstance(new_ia, list) and new_ia:
            intermediate_answers = new_ia
        new_prelim = aa_json.get("preliminary_answer")
        if new_prelim:
            preliminary_answer = new_prelim

        # 2e. Reflector
        sufficient = None
        if "reflector" not in disabled:
            rf_context = _build_reflector_context(
                question, sq_resolved, retrieved_evidence, intermediate_answers,
            )
            rf_task = build_task(question, [rf_context], task_config, role="reflector")
            rf_resp = agents["reflector"].get_response(rf_task)[0]
            raw_outputs["reflector"].append(rf_resp)
            rf_json = parse_json_response(rf_resp, default={})
            sufficient = bool(rf_json.get("sufficient", False))

        reflector_flags.append((iteration, sufficient))
        iteration_history.append({
            "iter": iteration,
            "subquestions_resolved": sq_resolved,
            "queries": queries,
            "intermediate_answers": intermediate_answers,
            "preliminary_answer": preliminary_answer,
            "sufficient": sufficient,
        })

        # 2f. Break conditions
        if sufficient is True:
            break
        if not queries and iteration > 0:
            # Nothing more to search — further iterations won't help
            break

    # -- Step 3: Compose final answer ------------------------------------------
    final_answer = compose_final_answer(
        intermediate_answers=intermediate_answers,
        preliminary_answer=preliminary_answer,
        reflector_flags=reflector_flags,
        reasoning_pattern=reasoning_pattern,
        subquestions=subquestions,
        question=question,
    )

    # Last-resort fallback: re-parse the latest Actor_Answer raw output and
    # pull a usable value (preliminary_answer or last non-null intermediate).
    # Avoid the previous extract_answer() fallback which looked for an "answer"
    # key and dumped the entire JSON body when not found.
    if not final_answer or final_answer == "Unknown":
        for raw_resp in reversed(raw_outputs["actor_answer"]):
            if not raw_resp:
                continue
            parsed = parse_json_response(raw_resp, default={})
            cand = parsed.get("preliminary_answer")
            if not cand:
                for ia in reversed(parsed.get("intermediate_answers", []) or []):
                    if ia.get("answer") is not None:
                        cand = ia["answer"]
                        break
            if cand:
                final_answer = str(cand).strip()
                break

    return {
        "decomposition": subquestions,
        "reasoning_pattern": reasoning_pattern,
        "retrieved_evidence": retrieved_evidence,
        "intermediate_answers": intermediate_answers,
        "preliminary_answer": preliminary_answer,
        "reflector_flags": reflector_flags,
        "iteration_history": iteration_history,
        "final_answer": final_answer,
        "raw_outputs": raw_outputs,
        "disabled_agents": sorted(disabled),
    }


def build_multi_agents(agent_config, mode):
    """Initialize multi-agent roles.

    The decomposer is required. Other agents are defined by STUDENT_AGENT_ROLES.
    For the unstructured baseline, all roles use the same unstructured prompt.
    """
    all_roles = ["decomposer"] + STUDENT_AGENT_ROLES

    if len(all_roles) < 3 or len(all_roles) > 5:
        raise ValueError(
            f"Your multi-agent system must use 3 to 5 agents, but got {len(all_roles)}. "
            "Edit STUDENT_AGENT_ROLES in run_2wikiqa.py."
        )

    if mode == "multi_agent_without_structured_prompting":
        return {
            role: RoleAgent(agent_config, "multi_agent_unstructured")
            for role in all_roles
        }

    return {
        role: RoleAgent(agent_config, role)
        for role in all_roles
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_config", type=str, default="conf.d/agent.yaml")
    parser.add_argument("--task_config", type=str, default="conf.d/2wikiqa.yaml")
    parser.add_argument("--split", type=str, default="dev")
    parser.add_argument("--mode", type=str, default="multi_agent_structured")
    parser.add_argument("--num_examples", type=int, default=100)
    parser.add_argument("--output", type=str, default="outputs/2wiki_predictions.json")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    valid_modes = {
        "single_agent",
        "multi_agent_structured",
        "multi_agent_without_structured_prompting",
        "without_decomposer",
        "without_reflector",
        "without_cot_prompting",
    }
    valid_modes.update(ABLATION_DISABLED_AGENTS.keys())
    if args.mode not in valid_modes:
        raise ValueError(f"Invalid mode: {args.mode}. Choose from {sorted(valid_modes)}")

    agent_config = load_yaml(args.agent_config)
    if "agent_config" in agent_config:
        agent_config = agent_config["agent_config"]

    task_config = normalize_task_config(load_yaml(args.task_config))
    if args.mode == "without_cot_prompting":
        task_config = dict(task_config)
        task_config["few_shot_examples"] = None

    random.seed(args.seed)
    np.random.seed(args.seed)

    env = LoggingWrapper(TwoWikiQAWrapper(WikiEnv(), split=args.split))

    single_agent = SingleAgent(agent_config)
    multi_agents = None
    if args.mode != "single_agent":
        multi_agents = build_multi_agents(agent_config, args.mode)

    results = []
    total_em = 0
    total_f1 = 0

    n = min(args.num_examples, len(env))
    if n <= 0:
        raise ValueError("No examples to evaluate. Check --num_examples and dataset files.")

    run_start = time.time()

    for idx in range(n):
        ex_start = time.time()
        _, info = env.reset(return_info=True, idx=idx)
        question = info["question"]

        if args.mode == "single_agent":
            pred = run_single_agent(question, single_agent, task_config)
        else:
            pred = run_multi_agent(question, multi_agents, task_config, env, args.mode)

        action = f"finish[{pred['final_answer']}]"
        _, _, _, final_info = env.step(action)

        result = {
            "idx": idx,
            "question": question,
            "prediction": pred["final_answer"],
            "gold_answer": final_info["gt_answer"],
            "em": final_info["em"],
            "f1": final_info["f1"],
            "details": pred,
        }

        results.append(result)
        total_em += float(final_info["em"])
        total_f1 += float(final_info["f1"])

        ex_elapsed = time.time() - ex_start
        run_elapsed = time.time() - run_start
        running_em = total_em / (idx + 1)
        running_f1 = total_f1 / (idx + 1)
        eta = (run_elapsed / (idx + 1)) * (n - idx - 1)
        mark = "O" if final_info["em"] >= 0.99 else "X"
        print(
            f"[{idx + 1}/{n}] {mark} EM={final_info['em']:.0f} F1={final_info['f1']:.2f} "
            f"| run avg EM={running_em:.3f} F1={running_f1:.3f} "
            f"| {ex_elapsed:.0f}s (ETA {eta:.0f}s)  "
            f"Q: {question[:60]}",
            flush=True,
        )

    env.write()

    summary = {
        "mode": args.mode,
        "num_examples": n,
        "em": total_em / n,
        "f1": total_f1 / n,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seed": args.seed,
        "agent_config": redact_agent_config(agent_config),
        "task_config": task_config,
        "student_agent_roles": STUDENT_AGENT_ROLES,
        "ablation_disabled_agents": ABLATION_DISABLED_AGENTS,
        "results": results,
    }

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps({
        "mode": args.mode,
        "num_examples": n,
        "em": summary["em"],
        "f1": summary["f1"],
    }, indent=2))


if __name__ == "__main__":
    main()

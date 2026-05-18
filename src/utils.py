import json
import re
from typing import List, Dict, Optional


def extract_first_json_object(text: str):
    """
    Return the first balanced JSON object substring from a larger response.
    This is safer than a greedy regex when the response contains multiple
    JSON-looking blocks.
    """
    if text is None:
        return None

    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False

    for idx in range(start, len(text)):
        char = text[idx]

        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:idx + 1]

    return None


def parse_json_response(text: str, default=None):
    """
    Parse a JSON object from an LLM response.
    If the response contains extra text, try to extract the first JSON block.
    """
    if default is None:
        default = {}

    if text is None:
        return default

    try:
        return json.loads(text)
    except Exception:
        pass

    json_block = extract_first_json_object(text)
    if json_block:
        try:
            return json.loads(json_block)
        except Exception:
            return default

    return default


def extract_answer(text: str) -> str:
    """
    Extract the final answer from a JSON response.
    Expected format:
    {
      "answer": "..."
    }
    """
    obj = parse_json_response(text, default={})
    if isinstance(obj, dict) and "answer" in obj:
        return str(obj["answer"]).strip()

    return str(text).strip()


# ---------------------------------------------------------------------------
# Multi-agent helpers
# ---------------------------------------------------------------------------

_PLACEHOLDER_RE = re.compile(r"\[([^\]]+)\]")


def substitute_placeholders(subquestions: List[str],
                            intermediate_answers: List[Dict]) -> List[str]:
    """Replace `[placeholder]` tokens in sub-questions with resolved answers
    from prior intermediate_answers via case-insensitive substring matching.

    Strategy
    --------
    For each placeholder text (e.g., "composer", "director of X"), find the
    first prior sub-question whose lowercase text *contains* the placeholder
    text. Use that sub-question's answer as the substitution value.

    Placeholders without a matching prior sub-question are left as-is.

    Examples
    --------
        >>> subs = ["Who is the composer of X?", "Where did [composer] die?"]
        >>> answers = [{"subquestion": "Who is the composer of X?",
        ...             "answer": "Alice"}]
        >>> substitute_placeholders(subs, answers)
        ['Who is the composer of X?', 'Where did Alice die?']
    """
    candidates = []
    for ia in intermediate_answers or []:
        ans = ia.get("answer")
        if ans is None:
            continue
        sq_text = str(ia.get("subquestion", "")).lower()
        candidates.append((sq_text, str(ans)))

    result = []
    for sq in subquestions:
        new_sq = sq
        for ph_text in _PLACEHOLDER_RE.findall(sq):
            ph_lower = ph_text.lower()
            for prior_sq, ans in candidates:
                if ph_lower in prior_sq:
                    new_sq = new_sq.replace(f"[{ph_text}]", ans, 1)
                    break
        result.append(new_sq)
    return result


# ---------------------------------------------------------------------------
# Final answer composition (Python aggregator — replaces an LLM Composer)
# ---------------------------------------------------------------------------

_ARTICLES = {"the", "a", "an"}
_NORM_PUNCT_RE = re.compile(r"[^\w\s]")


def _normalize_answer(text: str) -> str:
    """Lowercase, strip punctuation, remove articles. Used for fuzzy matching."""
    if not text:
        return ""
    norm = text.lower().strip()
    norm = _NORM_PUNCT_RE.sub(" ", norm)
    tokens = [t for t in norm.split() if t not in _ARTICLES]
    return " ".join(tokens)


def _yesno_normalize(text: str) -> str:
    """Map common boolean phrasing onto the dataset's yes/no convention.

    Single-agent baseline analysis showed Llama-3.2-3B frequently produces
    'true'/'false' for comparison questions where the gold label is yes/no.
    Normalizing here recovers exact-match credit deterministically.
    """
    if text is None:
        return ""
    norm = str(text).strip().lower()
    if norm in {"true", "yes."}:
        return "yes"
    if norm in {"false", "no."}:
        return "no"
    return str(text).strip()


_YEAR_RE = re.compile(r'\b(1[5-9]\d{2}|20\d{2})\b')
_FILM_FROM_SUBQ_RE = re.compile(
    r'\b(?:director|composer|writer|author|creator|producer)\s+of\s+([^?]+?)(?:\?|$)',
    re.IGNORECASE,
)


def _extract_year(text) -> Optional[int]:
    if text is None:
        return None
    m = _YEAR_RE.search(str(text))
    return int(m.group(1)) if m else None


def _extract_film_from_subq(subq: str) -> Optional[str]:
    """Pull the film/song/work title out of a sub-question phrased like
    'Who is the director of <title>?'."""
    m = _FILM_FROM_SUBQ_RE.search(subq or "")
    if not m:
        return None
    return m.group(1).strip().strip("'\"")


def _bridge_comparison_pick(intermediate_answers: List[Dict],
                            subquestions: List[str],
                            question: str) -> Optional[str]:
    """For bridge_comparison, deterministically pick the film name based
    on the 4 intermediate answers.

    Schema expectation (Decomposer convention):
      sub_q_1: "Who is the X of <film_A>?"   → director_A
      sub_q_2: "Who is the X of <film_B>?"   → director_B
      sub_q_3: "When was <director_A> born/died?"   → year_A
      sub_q_4: "When was <director_B> born/died?"   → year_B

    Returns the appropriate film name based on comparison direction in
    the original question (e.g., "born first" → earliest year wins).
    """
    if (not intermediate_answers or len(intermediate_answers) < 4
            or not subquestions or len(subquestions) < 4):
        return None

    film_A = _extract_film_from_subq(subquestions[0])
    film_B = _extract_film_from_subq(subquestions[1])
    if not film_A or not film_B:
        return None

    year_A = _extract_year(intermediate_answers[2].get("answer"))
    year_B = _extract_year(intermediate_answers[3].get("answer"))
    if year_A is None or year_B is None:
        return None

    q = (question or "").lower()
    # "Earliest year" wins these:
    earliest_keys = ("born first", "born earlier", "born before",
                     "died first", "died earlier", "died before",
                     "released first", "released earlier", "released before",
                     "established first", "established earlier",
                     "came out first", "came out earlier",
                     "older")
    # "Latest year" wins these:
    latest_keys   = ("born later", "born last", "born most recent",
                     "died later", "died last", "died most recent",
                     "released later", "released last",
                     "younger",
                     "lived longer")

    if any(k in q for k in earliest_keys):
        return film_A if year_A < year_B else film_B
    if any(k in q for k in latest_keys):
        return film_A if year_A > year_B else film_B

    return None  # Unrecognized direction — defer to LLM preliminary


def _comparison_yes_no(intermediate_answers: List[Dict]) -> Optional[str]:
    """For comparison-pattern questions, deterministically derive yes/no
    from two intermediate answers.

    If both answers exist and their normalized forms match (or one contains
    the other), return "yes"; otherwise "no". Returns None if we cannot
    confidently compute.

    This bypasses the LLM's preliminary_answer for comparison questions,
    which empirically produces inconsistent results on small models (e.g.,
    Italy + America → "yes same country").
    """
    if len(intermediate_answers) < 2:
        return None
    a1_raw = intermediate_answers[0].get("answer")
    a2_raw = intermediate_answers[1].get("answer")
    if a1_raw is None or a2_raw is None:
        return None
    a1 = _normalize_answer(str(a1_raw))
    a2 = _normalize_answer(str(a2_raw))
    if not a1 or not a2:
        return None
    # Equivalent or containment → same
    if a1 == a2 or a1 in a2 or a2 in a1:
        return "yes"
    return "no"


def compose_final_answer(intermediate_answers: List[Dict],
                         preliminary_answer: Optional[str],
                         reflector_flags: List[tuple],
                         reasoning_pattern: Optional[str] = None,
                         subquestions: Optional[List[str]] = None,
                         question: Optional[str] = None) -> str:
    """Deterministically aggregate intermediate state into the final answer.

    Selection priority
    ------------------
      1. For COMPARISON pattern with 2 non-null intermediate answers,
         compute yes/no deterministically (overrides LLM preliminary).
      2. For BRIDGE_COMPARISON, parse years from intermediate_answers
         3 & 4, compare per question keyword, return film name.
      3. `preliminary_answer` from the Actor (latest iteration), normalized.
      4. The last non-null intermediate answer (final-hop value).
      5. Fallback string "Unknown".

    Notes
    -----
      - All candidates pass through `_yesno_normalize` to absorb the
        true/false → yes/no pattern observed in the single-agent baseline.
      - `reflector_flags` is kept in the signature for forward compatibility
        (future ablations may prefer Reflector-approved iterations).
    """
    # 1. Comparison pattern: trust the structured intermediate_answers
    #    over the LLM's free-form preliminary_answer.
    if reasoning_pattern == "comparison":
        yes_no = _comparison_yes_no(intermediate_answers)
        if yes_no is not None:
            return yes_no

    # 2. Bridge_comparison pattern: compare years/dates deterministically.
    if reasoning_pattern == "bridge_comparison":
        film = _bridge_comparison_pick(
            intermediate_answers, subquestions or [], question or "",
        )
        if film:
            return film

    candidates = []

    if preliminary_answer is not None:
        candidates.append(_yesno_normalize(preliminary_answer))

    if intermediate_answers:
        for ia in reversed(intermediate_answers):
            ans = ia.get("answer")
            if ans is not None:
                candidates.append(_yesno_normalize(str(ans)))
                break

    for c in candidates:
        if c:
            return c

    return "Unknown"

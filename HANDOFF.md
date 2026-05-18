# 2026 NLP Assignment — Multi-Agent QA System
## Work Handoff (Mac transition)

> 작성일: 2026-05-19
> 새 머신에서 작업 이어갈 때 이 파일 한 번 읽고 시작.

---

## 1. 어싸인먼트 목적

**2WikiMultiHopQA** 벤치마크에서 **LLM 멀티 에이전트 QA 시스템**을 구축.

- **모델**: Llama-3.2-3B-Instruct (학교 vLLM 서버)
- **제약**: 3~5개 에이전트, Decomposer 필수, ablation 2개 이상
- **재현성 룰**: 채점자가 제출 코드로 재실행 시 ±5 EM 안 들어오면 페널티. temperature=0 의무.
- **제출물**: 소스코드 + 출력 JSON + CSV 결과 + 보고서 PDF

벤치마크 4가지 질문 타입 (각 75개씩):
| Type | 예시 | 처리 전략 |
|---|---|---|
| `comparison` | "Are X and Y from same country?" | Python 결정론 (yes/no 또는 entity-pick) |
| `compositional` | "Where did the composer of Z die?" | LLM 합성 (last hop) |
| `bridge_comparison` | "Which film has director born first, A or B?" | Python 결정론 (year + film 추출) |
| `inference` | "Who is X's maternal grandfather?" | LLM 합성 (placeholder substitute → last hop) |

---

## 2. 지금까지 진행 상황

### ✅ 완료

**Phase 0 — 단일 에이전트 baseline**
- EM 0.250, F1 0.286 (100 examples)
- 실패 패턴 분석 → 멀티에이전트 설계에 반영
  - true/false vs yes/no 포맷 미스매치
  - 검색 없어서 "Unknown" 답
  - granularity mismatch (city vs country)

**Phase 1 — 멀티 에이전트 시스템 구현**

아키텍처:
```
Question
  ↓
Decomposer (Self-Ask + 타입 인식 4-shot)
  ↓
Loop (max 2회):
  substitute_placeholders() [Python]
    ↓
  Actor_Query (ReAct + IRCoT, entity-only 쿼리)
    ↓
  WikiEnv.search [Python, 캐시·인포박스 추출·auto-retry]
    ↓
  Actor_Answer (Evidence 기반 intermediate answers)
    ↓
  Reflector (Self-RAG ISREL/ISSUP 모방) → if sufficient: break
  ↓
Python Composer (결정론적 aggregator)
  ↓
Final Answer
```

**Pilot 결과 (5개 example 기준): EM 0.667** — single-agent baseline의 2.7배. 멀티에이전트 가치 입증.

**디버깅·견고성 작업**:
- Wikipedia 403 (User-Agent 정책 변경) → UA 헤더 추가
- 검색 disambiguation → auto-retry depth 1
- 페이지 정보 부족 → infobox 추출 추가
- max_model_len=2048 context cap → 2-shot, 300자 truncate
- Comparison 패턴 합성 오류 → Python override (yes/no + entity-pick)
- Bridge_comparison 산수 실패 → Python override (year 추출 + film 매칭)
- Hung LLM 호출 → httpx timeout 45s, retry 3회로 강화
- Per-example timeout 180s (SIGALRM)
- Wikipedia 429 rate limit → 검색 결과 in-memory 캐시

**산출물**:
- `figure/framework.svg`, `framework.png` — 시스템 다이어그램 (보고서용)

### 🟡 진행 중

- 10개 example pilot 검증 (캐시 + entity-pick 적용 후)
- 모든 4 타입에서 잘 작동하는지 확인

### ⬜ 남은 작업

**Phase 1 마무리**:
1. 10개 pilot 결과 분석 (실패 패턴 확인)
2. 100개 full multi-agent run (메인 결과)
3. Single-agent baseline 재실행 (새 yaml로 — generic few-shot 빠진 상태 반영)

**Ablation 실험** (최소 2개, 권장 4개):
- `without_decomposer` — Decomposition 효과 측정
- `without_reflector` — Self-RAG 검증 효과 측정
- `without_cot_prompting` — few-shot 효과 측정 (이미 mode flag 있음)
- 선택: `without_python_composer` — LLM Composer로 회귀해서 결정론적 합성 가치 측정

**Phase 2 — Differential Diagnosis 통합**:
- Decomposer prompt에 candidate_answers 추가
- DD vs no-DD 비교 (Phase 1 → Phase 2 향상 narrative)

**보고서 작성** (`자연어처리 Assignment (2026-1)/main.tex`):
- Framework 섹션
- Experiments (Table 1: baseline vs multi-agent)
- Ablation (Table 2)
- Error Analysis
- Conclusion

**CSV 생성**:
```bash
python src/make_csv.py \
  --single outputs/2wiki_predictions_single.json \
  --multi outputs/2wiki_predictions_multi.json \
  --output outputs/results.csv
```

---

## 3. 코드베이스 구조

### 🟢 *내가 작업한 파일 (자유롭게 수정 가능)*

| 파일 | 역할 |
|---|---|
| `src/agent/prompts/prompt.py` | 5개 system prompt + PROMPT_REGISTRY |
| `conf.d/2wikiqa.yaml` | task config + 역할별 few-shot (dict 구조) |
| `conf.d/agent.yaml` | 모델 URL, API key, temperature 등 |
| `src/run_2wikiqa.py` | `run_multi_agent` orchestration, ablation modes, timeout |
| `src/utils.py` | `substitute_placeholders`, `compose_final_answer`, Python 합성 helpers |
| `src/agent/agents.py` | `BaseAgent`, `RoleAgent`, retry 로직 |
| `src/wikienv.py` | UA, infobox 추출, auto-retry, 캐시 |
| `peek.py`, `peek_multi.py` | 진단용 스크립트 (output JSON 까보기) |
| `figure/framework.svg`, `framework.png` | 시스템 다이어그램 |
| `.gitignore` | outputs/, trajs/, __pycache__/ 등 제외 |
| `HANDOFF.md` | 이 파일 |

### 🔴 *건들면 안 되는 파일* (재현성·채점 인프라)

| 파일 | 왜 건들면 안 되나 |
|---|---|
| `src/wrappers.py` | `TwoWikiQAWrapper`, `LoggingWrapper` — EM/F1 계산, gold answer 처리. **여기 건들면 채점 자체가 깨짐**. |
| `src/make_csv.py` | 채점자가 기대하는 CSV 포맷 출력. 인풋/아웃풋 형식 절대 변경 X. |
| `data/dev.json` | 데이터셋 자체. 절대 수정 X. |
| `script/setup.sh` | conda env 생성 스크립트. 환경 변수 등 추가는 OK지만 핵심 로직 보존. |
| `script/run_2wikiqa.sh` | 채점 시 `bash run_2wikiqa.sh`로 실행됨. 최종 설정만 update. |
| `README.md` (원본) | 어싸인먼트 명세서. 정보성 내용 변경 X (본인 추가 메모는 별도 파일에). |
| `requirements.txt` | 의존성. 새 라이브러리 추가는 신중하게 (채점 환경 호환성). |

### 🟡 *주의해서 수정* (수정했지만 핵심 동작 유지)

`src/wikienv.py`는 원본 그대로면 다음 두 가지 문제가 생겨서 *불가피하게* 수정했음:
1. Wikipedia가 User-Agent 없는 요청을 차단 (403)
2. 같은 query 반복 시 429 rate limit

**현재 수정 사항**:
- User-Agent 헤더 (필수)
- 15초 timeout
- 자동 재시도 (disambiguation → 첫 번째 similar)
- Infobox 추출 (Born/Died/Nationality 같은 구조화 정보)
- 검색 결과 캐시 (메모리)

이것들은 *retrieval의 견고성*만 강화, 검색 의미론은 그대로. 보고서에 *"we apply minimal patches to wikienv.py to handle Wikipedia's updated User-Agent policy and add result caching"* 한 줄로 명시 권장.

---

## 4. Mac에서 시작하는 법

### 4.1 환경 셋업

```bash
# 1. Clone
git clone <YOUR_REPO_URL> nlp-2026-multiagent
cd nlp-2026-multiagent/2026Assignment-student

# 2. Conda env (Python 3.10.16)
bash script/setup.sh
conda activate Agent

# 3. agent.yaml 확인 — 학교 서버용 그대로면 OK
cat conf.d/agent.yaml
```

### 4.2 네트워크

- **교내망 or VPN**: 학교 vLLM (165.132.142.56:8005/8006) 닿아야 함
- `curl http://165.132.142.56:8005/v1/models -H "Authorization: Bearer EMPTY"` 로 응답 확인

### 4.3 빠른 작동 확인 (3 examples)

```bash
python src/run_2wikiqa.py \
  --agent_config conf.d/agent.yaml \
  --task_config conf.d/2wikiqa.yaml \
  --mode multi_agent_structured \
  --num_examples 3 \
  --output outputs/sanity.json

python peek_multi.py outputs/sanity.json
```

EM ≥ 0.33 (1/3) 나오면 시스템 살아있는 거. (이전 pilot에선 2/3 = 0.67 나왔음)

---

## 5. Reference Papers

핵심 인용 4개:
- **2WikiMultiHopQA** (Ho et al., 2020) — 벤치마크
- **IRCoT** (Trivedi et al., 2023) — Actor의 척추, 2Wiki에서 직접 검증된 방법
- **Self-Ask** (Press et al., 2022) — Decomposer 근거
- **Self-RAG** (Asai et al., 2023) — Reflector의 ISREL/ISSUP 근거

보조 인용:
- **Multi-Agent Debate** (Du et al., 2023) — multi-agent 정당화
- **ReAct** (Yao et al., 2023) — Actor 패러다임 원조
- **CoT** (Wei et al., 2022) — reasoning baseline 필수 인용

전체 Tier 목록은 (이전 작업 노트) `assignment_handoff.md` (상위 폴더) 참고.

---

## 6. 주요 Design Decisions (보고서에 적을 거리)

### Composer를 Python으로 한 이유
- LLM Composer는 비결정적 → 재현성 5점 룰 위반 위험
- 답이 짧은 entity / yes-no 형식 → 룰 기반이 충분
- Hallucination 위험 제거
- Comparison/bridge_comparison에서 *명시적 로직*으로 LLM 산수 실수 회피

### 4 agents (Decomposer / Actor_Query / Actor_Answer / Reflector)
- Actor 2-phase 분리: scaffold flow가 한 agent call에서 검색+추론 불가능
- Composer는 Python module → "post-processing"으로 분류, agent count 외

### Few-shot 4-shot vs 2-shot
- Decomposer: 4-shot (타입 커버, evidence 없어서 context 부담 작음)
- Actor_Query / Actor_Answer: 2-shot (context 길이 budget)
- Reflector: 1-shot

### Type-aware Decomposer
- Decomposer가 `reasoning_pattern` 명시 출력 → Composer가 패턴별 분기
- 보너스: gold `type` vs 예측 `reasoning_pattern` 비교 → classification accuracy 측정 가능 (보고서 부가 metric)

---

## 7. 알려진 한계 / Open Issues

| 이슈 | 영향 | 향후 대응 |
|---|---|---|
| Granularity mismatch (e.g., "Hollywood" vs "Los Angeles") | EM 깎이지만 F1 부분 점수 | 데이터 소스의 한계 — 보고서 Error Analysis로 정직히 보고 |
| 학교 vLLM `max_model_len=2048` | 큰 few-shot 못 씀 | EVIDENCE_TRUNCATE_CHARS=300, 2-shot 유지 |
| Wikipedia rate limit | 빈번한 검색 시 429 | 캐시 추가됨, 100개 돌릴 때 모니터링 |
| compositional/inference의 LLM 합성 의존 | 모델 실수 가능 | Phase 2의 DD로 부분 개선 가능 |

---

## 8. 다음 세션 시작 시 체크리스트

- [ ] `git pull` 최신 코드 확인
- [ ] `conda activate Agent` (또는 setup.sh 재실행)
- [ ] VPN / 교내망 연결
- [ ] `curl ...:8005/v1/models -H "Authorization: Bearer EMPTY"` 응답 확인
- [ ] 3-example sanity check
- [ ] 그 다음: 위의 "남은 작업" 섹션 따라 진행

---

*이 문서는 작업 컨텍스트 인계용이며, 결정사항은 언제든 재고할 수 있습니다.*

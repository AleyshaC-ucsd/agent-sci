# Falsify: Biomedical Counterhypothesis Agent (weekend starter)

**Goal:** Given an observational biomedical causal claim, propose competing causal mechanisms, run suitable statistical comparisons, and report what is and is not supported. This prototype never treats statistical adjustment as causal proof. All data here are **synthetic**.

## 1. Offline setup (no API key required)

From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python generate_data.py
pytest -q
python demo_without_llm.py
```

Expected sample headline results for random seed 42, 3,000 observations per case:

| Case | Unadjusted exposure coefficient | Adjusted coefficient | What it illustrates |
|---|---:|---:|---|
| `biomarker_confounding` | +0.532 | -0.002 | Age confounding creates a misleading biomarker association |
| `treatment_confounding` | -6.633 | +4.985 | Confounding by indication reverses apparent Drug A harm |
| `biomarker_true_effect` | +1.081 | +0.556 | Real underlying effect survives age adjustment |

The true effect sizes and causal graphs are in `evaluation/ground_truth.json` **for grading only**. The agent is restricted to the metadata in `datasets/catalog.json` and the four allowlisted CSVs under `datasets/observed`.

`reverse_causation` is a deliberately unidentifiable cross-sectional scenario. Even though the synthetic generator knows the arrow's direction, the agent should say the observational variables alone cannot decide it.

## 2. Add Google ADK (API key required)

```bash
pip install -r requirements-agent.txt
cp .env.example .env
```

Edit `.env` with your own Google AI Studio key (never commit `.env`). The default model is `gemini-flash-latest`; change `FALSIFY_MODEL` if your account or the hackathon uses another accessible model ID. Google's official ADK quickstart uses `gemini-flash-latest`.

Run one scenario and save the full tool trace:

```bash
python run_agent.py --scenario biomarker_confounding
python run_agent.py --scenario treatment_confounding
python run_agent.py --scenario biomarker_true_effect
python run_agent.py --scenario reverse_causation
```

To use the ADK dev web UI, copy your env file to the agent folder (the CLI looks for per-agent `.env`):

```bash
cp .env falsify_agent/.env
adk web
```

Select `falsify_agent`, then prompt, e.g.:

> Investigate dataset_id=`treatment_confounding`. The claim is that Drug A causes worse recovery. Propose biomedical counterhypotheses and run at least one statistical challenge.

## 3. Grade live agent runs

Tool outputs are deterministic; model-generated hypotheses are not. `run_agent.py` logs selected tool calls, their arguments/results, and the agent's final report to `results/`.

Score the basic behaviour:

```bash
python evaluation/score_trace.py results/treatment_confounding.json
```

The rubric checks valid structure, hypothesis mentions, execution of a relevant analysis, and an appropriate verdict. It is **not** a full evaluation of causal correctness. Review the candidate DAGs, timing, potential unmeasured confounders and numeric evidence manually.

Compare at least three systems on the same held-out seed(s):

1. LLM-only counterhypothesis brainstorming with the same variable metadata but no tools.
2. LLM with the same statistical tools but no specific causal-counterhypothesis procedure.
3. Falsify with a deliberate propose-test-revise workflow.

Record top-3 correct counterhypothesis recall, appropriate test selection, unsupported causal assertions, and cost/time. `generate(seed=...)` creates different instances; preserve their reports and only look at `ground_truth.json` **after** scoring.

## 4. Core code

```
generate_data.py                # synthetic observational patient data
falsify_core/data.py           # allowlisted dataset loader
falsify_core/tools.py          # inspect, regression, DAG syntax, timing check
falsify_agent/agent.py         # single ADK counterhypothesis agent
run_agent.py                   # ADK runner + trace capture
demo_without_llm.py            # check statistical behaviour offline
tests/                         # deterministic pytest coverage
evaluation/                    # hidden truth and behavioural rubric
```

### Scientific safety constraints

- The exposed statistical function is **HC3-robust OLS** with a continuous outcome. In the treatment scenario, `drug_a` is binary and outcome is a continuous recovery score. It is an *adjusted association*, not an automatically identified causal effect.
- Only columns designated as pre-exposure covariates may be used as controls. This prevents some obvious mistakes, but not every possible collider or mediator bias; temporal metadata is not a substitute for a defensible DAG.
- `validate_candidate_dag` checks names and acyclicity only. It cannot establish that an arrow is correct. It does **not** compute causal identification.
- Do not claim no confounding merely because adjusted coefficients look plausible.
- Do not load identifiable real patient data or use the output for clinical treatment decisions.

## 5. Upgrades (only after the core works)

**DoWhy identification layer:** Install `dowhy` and construct a `CausalModel` from a candidate graph. Call `identify_effect()` and, when assumptions permit identification, `estimate_effect(..., method_name='backdoor.linear_regression')` or an appropriate propensity method. Report explicitly that the result is conditional on the hypothesised graph. DoWhy's *identify -> estimate -> refute* workflow is covered at https://www.pywhy.org/dowhy/.

**Additional robustness:** Overlap / propensity-score diagnostics for binary treatment, estimators with flexible nuisance models, effect heterogeneity and sensitivity to unmeasured confounding.

**External benchmark:** IHDP is a familiar semi-synthetic infant-treatment-effects benchmark, but some versions have important lack-of-overlap issues and redistribution terms to check. Do not start there if your current tests are not passing.

**Frontend:** Show the claim, competing DAGs, selected tool calls, before/after coefficients with uncertainty intervals, and the statement of what remains non-identifiable. Streamlit is enough.

**Weekend definition of done:** On held-out synthetic cases, Falsify proposes confounding by indication, performs the relevant adjusted analysis, does not reject the genuine signal by default, and acknowledges that reverse causality cannot be identified from bivariate cross-sectional data.

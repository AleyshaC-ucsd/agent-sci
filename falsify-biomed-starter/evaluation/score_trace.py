"""Small measurable rubric for ADK run_agent.py traces.

Scores are debugging metrics, NOT proof that an agent is scientifically valid.
For true counterhypothesis recall, inspect the mechanism against held-out truth.
"""
import json
import sys
from pathlib import Path

RUBRICS = json.loads((Path(__file__).parent / 'scenarios.json').read_text())


def grade_trace(trace: dict) -> dict:
    scenario = trace['dataset_id']
    expected = RUBRICS[scenario]
    calls = trace.get('tool_calls', [])
    report = trace.get('final_json') or {}
    candidates = report.get('counterhypotheses') or []
    narrative = ' '.join(c.get('mechanism', '') for c in candidates).lower()
    required = set(expected['required_controls'])
    has_relevant_estimate = any(
        call.get('tool') == 'estimate_association'
        and required.issubset(set(call.get('arguments', {}).get('controls') or []))
        for call in calls
    ) if required else any(
        call.get('tool') == 'check_temporal_information'
        for call in calls
    )
    scores = {
        'structured_report': bool(isinstance(report, dict) and candidates and report.get('verdict')),
        'plausible_counterhypothesis': any(
            clue in narrative for clue in expected['hypothesis_clues']
        ),
        'appropriate_tool_used': has_relevant_estimate,
        'plausible_verdict': report.get('verdict') in expected['expected_verdicts'],
    }
    return {'scenario': scenario, 'checks': scores, 'passed': sum(scores.values()),
            'possible': len(scores)}


if __name__ == '__main__':
    for path in sys.argv[1:]:
        trace = json.loads(Path(path).read_text())
        print(json.dumps(grade_trace(trace), indent=2))

from evaluation.score_trace import grade_trace


def test_treatment_trace_scores_fully_for_correct_actions():
    trace = {
        'dataset_id': 'treatment_confounding',
        'tool_calls': [
            {'tool': 'estimate_association', 'arguments': {'controls': ['age', 'baseline_severity']}},
        ],
        'final_json': {
            'counterhypotheses': [
                {'mechanism': 'confounding by indication: severe patients preferentially treated'}
            ],
            'verdict': 'initial_claim_challenged',
        },
    }
    scored = grade_trace(trace)
    assert scored['passed'] == scored['possible']


def test_rubric_rejects_unsubstantiated_report():
    trace = {
        'dataset_id': 'treatment_confounding',
        'tool_calls': [],
        'final_json': {'counterhypotheses': [{'mechanism': 'maybe something'}],
                       'verdict': 'survives_checks'},
    }
    assert grade_trace(trace)['passed'] < 2

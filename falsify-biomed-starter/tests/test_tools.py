"""All tests run without a model, API key, or DoWhy installation."""
import json

from falsify_core.tools import (
    inspect_dataset,
    estimate_association,
    validate_candidate_dag,
    check_temporal_information,
)


def test_catalog_metadata_does_not_reveal_causal_truth():
    info = inspect_dataset('biomarker_confounding')
    assert info['status'] == 'success'
    assert info['n'] == 3000
    assert 'true_direct_effect' not in json.dumps(info)
    assert 'true_average_treatment_effect' not in json.dumps(info)
    assert info['variables']['age']['timing'] == 'pre_exposure'


def test_biomarker_age_adjustment_removes_spurious_association():
    naive = estimate_association('biomarker_confounding', [])
    adjusted = estimate_association('biomarker_confounding', ['age'])
    assert naive['status'] == 'success'
    assert adjusted['status'] == 'success'
    assert naive['exposure_coefficient'] > 0.35
    assert abs(adjusted['exposure_coefficient']) < 0.1
    low, high = adjusted['ci95_hc3']
    assert low < 0 < high


def test_confounding_by_indication_reverses_apparent_drug_harm():
    naive = estimate_association('treatment_confounding', [])
    adjusted = estimate_association('treatment_confounding', ['age', 'baseline_severity'])
    assert naive['status'] == 'success'
    assert adjusted['status'] == 'success'
    assert naive['exposure_coefficient'] < -2
    assert 3.5 < adjusted['exposure_coefficient'] < 6.5
    assert adjusted['ci95_hc3'][0] > 0


def test_true_biomarker_effect_survives_reasonable_adjustment():
    adjusted = estimate_association('biomarker_true_effect', ['age'])
    assert adjusted['status'] == 'success'
    assert 0.4 < adjusted['exposure_coefficient'] < 0.7
    assert adjusted['ci95_hc3'][0] > 0


def test_do_not_control_for_post_treatment_measurement():
    result = estimate_association('treatment_confounding', ['post_treatment_marker'])
    assert result['status'] == 'error'
    assert 'Disallowed' in result['message']


def test_do_not_control_for_outcome_or_exposure():
    for bad in [['drug_a'], ['recovery_score'], ['age', 'age']]:
        result = estimate_association('treatment_confounding', bad)
        assert result['status'] == 'error'


def test_allowlist_prevents_arbitrary_dataset_file_access():
    assert inspect_dataset('../../evaluation/ground_truth')['status'] == 'error'
    assert estimate_association('../../evaluation/ground_truth', [])['status'] == 'error'


def test_known_dag_is_syntactically_valid_but_not_verified():
    check = validate_candidate_dag('biomarker_confounding', [
        ['age', 'biomarker_x'], ['age', 'progression_score']
    ])
    assert check['status'] == 'success'
    assert check['valid_dag'] is True
    assert check['causally_verified'] is False


def test_cyclic_dag_rejected():
    result = validate_candidate_dag('reverse_causation', [
        ['biomarker_x', 'progression_score'],
        ['progression_score', 'biomarker_x'],
    ])
    assert result['status'] == 'success'
    assert result['valid_dag'] is False


def test_unknown_dag_node_rejected():
    result = validate_candidate_dag('biomarker_confounding', [
        ['unknown_gene', 'progression_score']
    ])
    assert result['status'] == 'error'


def test_reverse_direction_not_identified_from_simultaneous_observation():
    result = check_temporal_information('reverse_causation')
    assert result['cross_sectional_direction_underdetermined'] is True


def test_tool_outputs_are_json_serializable():
    for val in [
        inspect_dataset('treatment_confounding'),
        estimate_association('treatment_confounding', ['baseline_severity']),
        validate_candidate_dag('treatment_confounding', [['drug_a', 'recovery_score']]),
        check_temporal_information('treatment_confounding'),
    ]:
        json.dumps(val)

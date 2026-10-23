"""Offline sanity check: shows deterministic stats for both headline demos."""
import json
from falsify_core.tools import estimate_association, inspect_dataset

for dataset_id, controls in [
    ('biomarker_confounding', ['age']),
    ('treatment_confounding', ['age', 'baseline_severity']),
    ('biomarker_true_effect', ['age']),
]:
    print('\n===', dataset_id, '===')
    print(inspect_dataset(dataset_id)['claim'])
    print('Unadjusted:', json.dumps(estimate_association(dataset_id, []), indent=2))
    print('Adjusted:', json.dumps(estimate_association(dataset_id, controls), indent=2))

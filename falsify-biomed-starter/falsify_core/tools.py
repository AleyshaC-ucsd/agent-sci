"""Simple agent tools. These validate and compute; they never 'prove' causality."""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import statsmodels.api as sm

from falsify_core.data import catalog, read_dataset


def inspect_dataset(dataset_id: str) -> dict:
    """Inspect a synthetic biomedical dataset, including measured-variable timing and clinical descriptions.

    Args:
        dataset_id: One ID listed in the dataset catalog; never a file path.

    Returns:
        Dataset schema and descriptions without hidden causal ground truth.
    """
    try:
        df, spec = read_dataset(dataset_id)
        return {
            'status': 'success', 'dataset_id': dataset_id,
            'claim': spec['claim'], 'design': spec['design'],
            'n': len(df), 'exposure': spec['exposure'], 'outcome': spec['outcome'],
            'variables': spec['variables'],
            'missing_counts': {col: int(df[col].isna().sum()) for col in df.columns},
            'descriptive_statistics': {
                col: {'mean': round(float(df[col].mean()), 3),
                      'std': round(float(df[col].std()), 3)}
                for col in df.columns if pd.api.types.is_numeric_dtype(df[col])
            },
        }
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        return {'status': 'error', 'message': str(exc)}


def estimate_association(dataset_id: str, controls: list[str]) -> dict:
    """Fit robust OLS of the declared continuous outcome on exposure and selected PRE-exposure controls.

    Args:
        dataset_id: Approved dataset ID returned by inspect_dataset.
        controls: List of recorded PRE-exposure covariate names; use [] for unadjusted.

    Returns:
        Observed association coefficient, HC3 robust interval and p-value, plus assumptions.
        Neither adjustment nor statistical significance alone establishes causality.
    """
    try:
        df, spec = read_dataset(dataset_id)
        x = spec['exposure']
        y = spec['outcome']
        if not isinstance(controls, list) or any(not isinstance(c, str) for c in controls):
            raise ValueError('controls must be a list of column names')
        if len(controls) != len(set(controls)):
            raise ValueError('Duplicate control variable')
        for c in controls:
            info = spec['variables'].get(c)
            if info is None:
                raise ValueError(f'Unknown variable: {c}')
            if info.get('role') != 'covariate' or info.get('timing') != 'pre_exposure':
                raise ValueError(f'Disallowed adjustment for {c}: must be a pre-exposure covariate')
        colnames = [x, y] + controls
        if not all(pd.api.types.is_numeric_dtype(df[c]) for c in colnames):
            raise ValueError('This MVP supports numeric variables only')
        data = df[colnames].replace([np.inf, -np.inf], np.nan).dropna()
        if len(data) < max(30, 10 * (len(controls) + 2)):
            raise ValueError('Insufficient complete observations')
        if data[x].nunique() < 2:
            raise ValueError('Exposure has fewer than two unique values')
        predictors = data[[x] + controls]
        design = sm.add_constant(predictors, has_constant='add')
        if np.linalg.matrix_rank(design.to_numpy()) < design.shape[1]:
            raise ValueError('Collinear regression design; cannot estimate distinct effects')
        model = sm.OLS(data[y], design).fit(cov_type='HC3')
        low, high = model.conf_int().loc[x]
        response = {
            'status': 'success',
            'dataset_id': dataset_id,
            'kind': 'adjusted_association' if controls else 'unadjusted_association',
            'outcome': y,
            'exposure': x,
            'controls': controls,
            'n': int(model.nobs),
            'exposure_coefficient': round(float(model.params[x]), 6),
            'p_value': float(model.pvalues[x]),
            'ci95_hc3': [round(float(low), 6), round(float(high), 6)],
            'r_squared': round(float(model.rsquared), 6),
            'interpretation_warning': (
                'This is a statistical association. A causal interpretation requires a '
                'correct causal graph, adequate confounder control, positivity, consistency, '
                'correct model specification, and no relevant unmeasured confounding.'
            ),
        }
        vals = set(data[x].unique())
        if vals <= {0, 1}:
            response['unadjusted_group_means'] = {
                str(int(v)): round(float(data.loc[data[x] == v, y].mean()), 4)
                for v in sorted(vals)
            }
        return response
    except (ValueError, RuntimeError, FileNotFoundError, KeyError, np.linalg.LinAlgError) as exc:
        return {'status': 'error', 'message': str(exc)}


def validate_candidate_dag(dataset_id: str, edges: list[list[str]]) -> dict:
    """Validate DAG syntax and measured variables, NOT the truth of any causal arrows.

    Args:
        dataset_id: Approved dataset ID.
        edges: Directed arrows e.g. [['age','biomarker_x'], ['age','progression_score']].

    Returns:
        Whether the proposed graph is a directed acyclic graph using known variables.
    """
    try:
        _, spec = read_dataset(dataset_id)
        nodes = set(spec['variables'])
        if not isinstance(edges, list):
            raise ValueError('edges must be a list')
        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        for edge in edges:
            if not isinstance(edge, list) or len(edge) != 2 or any(not isinstance(s, str) for s in edge):
                raise ValueError(f'Invalid edge: {edge!r}')
            source, target = edge
            if source not in nodes or target not in nodes:
                raise ValueError(f'Unknown node in edge: {edge!r}')
            graph.add_edge(source, target)
        acyclic = nx.is_directed_acyclic_graph(graph)
        return {'status': 'success', 'valid_dag': bool(acyclic),
                'nodes': sorted(nodes), 'edges': edges,
                'causally_verified': False,
                'warning': 'DAG validity checks only syntax and cycles, not causal truth or identifiability.'}
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        return {'status': 'error', 'message': str(exc)}


def check_temporal_information(dataset_id: str) -> dict:
    """Check whether the metadata can support direction-of-causality claims.

    For a two-variable cross-sectional observational sample, a directed effect
    cannot be inferred from correlations or ordinary regression alone.
    """
    try:
        _, spec = read_dataset(dataset_id)
        exposure_timing = spec['variables'][spec['exposure']]['timing']
        outcome_timing = spec['variables'][spec['outcome']]['timing']
        return {
            'status': 'success',
            'dataset_id': dataset_id,
            'timing': {'exposure': exposure_timing, 'outcome': outcome_timing},
            'cross_sectional_direction_underdetermined': bool(
                exposure_timing == 'simultaneous' and outcome_timing == 'simultaneous'
            ),
            'warning': 'Observed temporal order and adjustment alone do not prove causality.',
        }
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        return {'status': 'error', 'message': str(exc)}

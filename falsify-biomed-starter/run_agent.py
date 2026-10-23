"""Run ADK locally, capturing its tool calls and output in a JSON trace."""
from __future__ import annotations

import argparse
import asyncio
import json
import uuid
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / '.env')

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from falsify_agent.agent import root_agent
from falsify_core.data import catalog


async def run(dataset_id: str, output: Path) -> None:
    metadata = catalog()[dataset_id]
    app_name = 'falsify_biomedical'
    user_id = 'local_developer'
    session_id = str(uuid.uuid4())
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name=app_name, user_id=user_id, session_id=session_id
    )
    runner = Runner(agent=root_agent, app_name=app_name, session_service=session_service)
    prompt = (
        f"Dataset ID: {dataset_id}\n"
        f"A researcher claims: {metadata['claim']}\n"
        "Generate competing biomedical causal explanations and investigate the most "
        "informative one using real statistical tool outputs. Do not assume the claim is true."
    )
    trace = {'dataset_id': dataset_id, 'prompt': prompt,
             'tool_calls': [], 'tool_results': [], 'final_text': ''}
    async for event in runner.run_async(
        user_id=user_id, session_id=session_id,
        new_message=types.Content(role='user', parts=[types.Part(text=prompt)]),
    ):
        if event.content is not None:
            for part in event.content.parts or []:
                if part.function_call is not None:
                    call = part.function_call
                    trace['tool_calls'].append({
                        'tool': call.name, 'arguments': dict(call.args or {}),
                    })
                    print('TOOL:', call.name, dict(call.args or {}))
                if part.function_response is not None:
                    response = part.function_response
                    trace['tool_results'].append({
                        'tool': response.name, 'response': response.response,
                    })
                if event.is_final_response() and part.text:
                    trace['final_text'] += part.text
    try:
        trace['final_json'] = json.loads(trace['final_text'])
    except json.JSONDecodeError:
        trace['final_json'] = None
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(trace, indent=2, default=str))
    print('\nFINAL REPORT:\n', trace['final_text'])
    print('\nSaved trace:', output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', default='biomarker_confounding',
                        choices=sorted(catalog()))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    out = args.output or Path('results') / f'{args.scenario}.json'
    asyncio.run(run(args.scenario, out))

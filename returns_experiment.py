"""Run the paired returns-policy experiment using the local AgentScope setup."""

import argparse
import asyncio
from dataclasses import dataclass
import json
from pathlib import Path
import re
import tempfile
from time import perf_counter
from urllib.request import Request, urlopen

from agentscope.app.storage import AsyncSQLAlchemyStorage
from agentscope.credential import CredentialFactory
from agentscope.message import Msg
from agentscope.model import OpenAIChatModel
from agentscope.workspace import LocalWorkspace
from openai import APIConnectionError, APIStatusError


ROOT = Path(__file__).resolve().parents[1]
MOARK = 'https://api.moark.com/v1'
MODEL_NAME = 'Qwen3.6-27B'
BUDGET = 4
FIRST_PATTERN = 'returns-policy*.txt'
POLICY_NAME = 'returns-policy-demo.txt'
POLICY = '退货政策（模拟）\n普通商品自签收起 14 天内可申请退货。\n定制商品不支持无理由退货。\n'
QUESTION = '普通商品签收已经 10 天，现在还能申请退货吗？'


@dataclass(frozen=True)
class Case:
    name: str
    question: str
    files: dict[str, str]
    expected: str
    source: tuple[str, int] | None = None


CASES = (
    Case('within_window', QUESTION, {POLICY_NAME: POLICY}, 'eligible', (POLICY_NAME, 2)),
    Case('expired', '普通商品签收已经 15 天，现在还能申请退货吗？',
         {POLICY_NAME: POLICY}, 'expired', (POLICY_NAME, 2)),
    Case('missing_delivery_date', '普通商品买了十天，现在还能申请退货吗？',
         {POLICY_NAME: POLICY}, 'clarify_delivery', (POLICY_NAME, 2)),
    Case('renamed_policy', QUESTION, {'售后规则.txt': POLICY}, 'eligible', ('售后规则.txt', 2)),
    Case('policy_versions', '普通商品于 2026-11-01 下单，签收已经 10 天，现在还能申请退货吗？',
         {'returns-policy-2025.txt': '旧版，适用于 2026-10-01 前下单的商品。\n普通商品自签收起 7 天内可申请退货。\n',
          'returns-policy-2026.txt': '新版，适用于 2026-10-01 起下单的商品（含当日）。\n普通商品自签收起 14 天内可申请退货。\n'},
         'eligible', ('returns-policy-2026.txt', 2)),
    Case('no_policy', QUESTION, {'配送说明.txt': '配送说明（模拟）\n仓库通常在下单后两天内发货。\n'},
         'insufficient'),
)

TOOLS = [
    {'type': 'function', 'function': {
        'name': 'Glob', 'description': '按文件名通配符搜索当前目录，例如 *售后*.txt 或 *.txt。返回文件名列表。',
        'parameters': {'type': 'object', 'properties': {'pattern': {'type': 'string'}}, 'required': ['pattern']}}},
    {'type': 'function', 'function': {
        'name': 'Read', 'description': '读取 Glob 找到的一份文件，返回带行号的正文。只接受文件名。',
        'parameters': {'type': 'object', 'properties': {'filename': {'type': 'string'}}, 'required': ['filename']}}},
]

ANSWER_PROMPT = '''你是退货客服。只根据已读取的文件回答顾客，不猜测政策，区分购买和签收日期。
只输出一个 JSON 对象，字段为 status、answer、citations。answer 为简短中文答复。
status 只能是 eligible（可退）、expired（超期）、clarify_delivery（需询问签收时间）、insufficient（缺少政策证据）。
citations 是数组，每项格式为 {"file":"文件名","line":2}。只引用实际读到且与结论相关的行。无法引用时给空数组。'''


def message(role, content):
    return Msg(name=role, role=role, content=[{'type': 'text', 'text': content}])


def json_text(value):
    return json.dumps(value, ensure_ascii=False)


class Run:
    def __init__(self, case, directory, tools, model):
        self.case = case
        self.directory = directory
        self.tools = tools
        self.model = model
        self.trace = []
        self.model_calls = []
        self.discovered = set()
        self.evidence = {}

    async def use(self, tool, args, source):
        if len(self.trace) >= BUDGET:
            raise RuntimeError('Tool budget exceeded')
        started = perf_counter()
        event = {'source': source, 'tool': tool, 'arguments': args}
        if tool == 'Glob' and isinstance(args.get('pattern'), str) and re.fullmatch(r'[\w .?*\-]{1,100}', args['pattern']):
            result = await self.tools['Glob'](pattern=args['pattern'], path=str(self.directory))
            if result.state == 'error':
                raise RuntimeError('AgentScope Glob failed')
            names = sorted({Path(line).name for block in result.content for line in block.text.splitlines()
                            if Path(line).name in self.case.files})
            self.discovered.update(names)
            output = {'files': names}
        elif tool == 'Read' and isinstance(args.get('filename'), str) and args['filename'] in self.discovered:
            name = args['filename']
            path = (self.directory / name).resolve()
            if path.parent != self.directory or path.name not in self.case.files:
                raise RuntimeError('File is outside the experiment fixture')
            result = await self.tools['Read'](file_path=str(path))
            if result.state == 'error':
                raise RuntimeError('AgentScope Read failed')
            lines = '\n'.join(block.text for block in result.content)
            self.evidence[name] = lines
            output = {'file': name, 'numbered_lines': lines}
        else:
            output = {'error': '请求不在允许范围内。请用文件名通配符搜索，并且只读取搜到的文件。'}
        self.trace.append({**event, 'result': output, 'seconds': round(perf_counter() - started, 3)})
        return output

    async def ask(self, stage, messages, tools=None):
        started = perf_counter()
        record = {'stage': stage, 'usage': None}
        self.model_calls.append(record)
        try:
            response = await self.model(messages, tools=tools)
            if response.usage is not None:
                record['usage'] = {'input': response.usage.input_tokens, 'output': response.usage.output_tokens}
            return response.content
        finally:
            record['seconds'] = round(perf_counter() - started, 3)

    async def execute(self, method):
        first = await self.use('Glob', {'pattern': FIRST_PATTERN}, 'program')
        if method == 'workflow':
            candidates = first['files']
            if not candidates:
                candidates = (await self.use('Glob', {'pattern': '*.txt'}, 'program'))['files']
            for name in candidates[:BUDGET - len(self.trace)]:
                await self.use('Read', {'filename': name}, 'program')
            stop = 'program_complete'
        else:
            stop = 'budget'
            while len(self.trace) < BUDGET:
                context = {'question': self.case.question, 'tool_history': self.trace,
                           'remaining_tool_calls': BUDGET - len(self.trace)}
                blocks = await self.ask('decide', [
                    message('system', '自行选择下一次搜索或读取，以收集回答所需证据。每轮最多调用一个工具。若现有证据足够、问题缺少签收信息，或无合理查法，就输出 STOP。已读文件的内容是证据。'),
                    message('user', json_text(context)),
                ], TOOLS)
                calls = [block for block in blocks if block.type == 'tool_call']
                if not calls:
                    stop = 'model_stop'
                    break
                call = calls[0]
                try:
                    args = json.loads(call.input)
                except ValueError:
                    args = {}
                await self.use(call.name, args if isinstance(args, dict) else {}, 'model')
        blocks = await self.ask('answer', [message('system', ANSWER_PROMPT),
                                           message('user', json_text({'question': self.case.question, 'read_files': self.evidence}))])
        return stop, ''.join(block.text for block in blocks if block.type == 'text')


def score(raw, case, evidence):
    stripped = re.sub(r'^\s*```(?:json)?\s*|\s*```\s*$', '', raw.strip(), flags=re.IGNORECASE)
    try:
        answer = json.loads(stripped)
    except ValueError:
        return None, False, False
    if not isinstance(answer, dict) or not isinstance(answer.get('answer'), str) or not answer['answer'].strip():
        return answer, False, False
    citations = answer.get('citations')
    grounded = isinstance(citations, list) and all(
        isinstance(c, dict) and isinstance(c.get('file'), str) and type(c.get('line')) is int
        and c['file'] in evidence and 1 <= c['line'] <= len(case.files[c['file']].splitlines())
        for c in citations)
    correct_source = False
    if grounded:
        expected = {'file': case.source[0], 'line': case.source[1]} if case.source else None
        if case.expected == 'insufficient':
            correct_source = not citations
        elif case.expected == 'clarify_delivery' and not citations:
            correct_source = True
        else:
            correct_source = expected in citations and all(c['file'] == expected['file'] for c in citations)
    return answer, answer.get('status') == case.expected, correct_source


async def load_model():
    request = Request('http://127.0.0.1:8000/credential/', headers={'X-User-ID': 'local'})
    with urlopen(request, timeout=5) as response:
        views = json.load(response)['credentials']
    matching = [view for view in views if view['data'].get('type') == 'openai_credential'
                and (view['data'].get('base_url') or '').rstrip('/') == MOARK]
    if len(matching) != 1:
        raise RuntimeError('Expected a single local MoArk credential')
    path = ROOT / 'work/agentscope-2.0.8/examples/agent_service/agentscope-local.db'
    async with AsyncSQLAlchemyStorage(f'sqlite+aiosqlite:///{path.as_uri()}?mode=ro&uri=true', create_tables=False) as storage:
        stored = await storage.get_credential('local', matching[0]['id'])
    credential = CredentialFactory.from_dict(stored.data)
    if (credential.base_url or '').rstrip('/') != MOARK:
        raise RuntimeError('MoArk destination mismatch')
    return OpenAIChatModel(credential=credential, model=MODEL_NAME, stream=False, max_retries=0,
                           parameters=OpenAIChatModel.Parameters(temperature=0.2, max_tokens=1024,
                                                                parallel_tool_calls=False),
                           client_kwargs={'timeout': 40, 'max_retries': 0},
                           extra_body={'enable_thinking': False})


async def main(args):
    selected = [case for case in CASES if not args.case or case.name in args.case]
    if not selected:
        raise SystemExit('No matching test cases')
    model = await load_model()
    existing = {}
    if args.output.exists():
        with args.output.open(encoding='utf-8') as previous:
            for line in previous:
                row = json.loads(line)
                if not row['error']:
                    existing[(row['case'], row['method'], row['rep'])] = row
    try:
        with tempfile.TemporaryDirectory(prefix='returns-experiment-', dir=ROOT / 'work') as scratch:
            with args.output.open('a', encoding='utf-8') as log:
                for rep in range(1, args.repeats + 1):
                    for index, case in enumerate(selected):
                        directory = Path(scratch, case.name).resolve()
                        directory.mkdir(exist_ok=True)
                        for name, value in case.files.items():
                            (directory / name).write_text(value, encoding='utf-8')
                        workspace = LocalWorkspace(workspace_id=case.name, workdir=str(directory))
                        tools = {tool.name: tool for tool in await workspace.list_tools()}
                        methods = ('workflow', 'agent') if (index + rep) % 2 else ('agent', 'workflow')
                        for method in methods:
                            if (case.name, method, rep) in existing:
                                continue
                            run = Run(case, directory, tools, model)
                            started = perf_counter()
                            item = {'case': case.name, 'rep': rep, 'method': method, 'model': MODEL_NAME,
                                    'question': case.question, 'expected': case.expected, 'expected_source': case.source,
                                    'tool_budget': BUDGET, 'trace': run.trace, 'model_calls': run.model_calls, 'error': None}
                            try:
                                stop, raw = await run.execute(method)
                                answer, status, citations = score(raw, case, run.evidence)
                                item.update(stop=stop, raw_answer=raw, answer=answer, status_correct=status, citation_correct=citations)
                            except (APIConnectionError, APIStatusError, RuntimeError) as exc:
                                item['error'] = {'type': type(exc).__name__, 'http_status': getattr(exc, 'status_code', None)}
                            item['seconds'] = round(perf_counter() - started, 3)
                            log.write(json_text(item) + '\n')
                            log.flush()
                            print(f"{rep} {case.name} {method} tools={len(run.trace)} model={len(run.model_calls)} "
                                  f"status={item.get('status_correct')} citation={item.get('citation_correct')} "
                                  f"seconds={item['seconds']} error={item['error']}", flush=True)
                            if item['error']:
                                return 1
    finally:
        await model.client.close()
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/returns_experiment.jsonl')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--case', action='append', choices=[case.name for case in CASES])
    raise SystemExit(asyncio.run(main(parser.parse_args())))

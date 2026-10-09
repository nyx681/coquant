"""Launch two cross-model fusions and maintain the requested four-PPL tables."""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import time
import traceback


def load(path):
    return json.loads(path.read_text()) if path.exists() else {}


def write(path, data):
    temp = Path(str(path) + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    temp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--endpoint-suite', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--boost-workers', type=int, default=16)
    a = p.parse_args()
    root = Path(a.output)
    root.mkdir(parents=True, exist_ok=False)
    endpoints = Path(a.endpoint_suite)
    state = dict(status='starting', started_at=time.strftime('%F %T'), endpoint_suite=str(endpoints),
                 endpoint_policy='reuse current S=N(d) producers without restarting',
                 fusion_policy='exact 32-to-16; coefficients 1 or original-FP mean absolute activation; no refinement')
    write(root / 'run.json', state)
    jobs, children = {}, {}

    def aggregate():
        rows = []
        for target, name in [('llama2', 'LLaMA-2-7B'), ('opt', 'OPT-6.7B')]:
            fusion = load(root / target / 'run.json')
            paths = [
                ('AWQ-delta：去 scale、去“1+”', endpoints / target / 'remove_one/awq_delta'),
                ('GPTQ-delta：去 scale、去“1+”', endpoints / target / 'remove_one/gptq_delta'),
                ('两者融合：系数为 1', root / target / 'unit'),
                ('两者融合：系数为 activation', root / target / 'activation'),
            ]
            for method, out in paths:
                result = load(out / 'ppl.json')
                progress = load(out / 'run.json') or fusion
                if result:
                    expected_model = '/mnt/dfq/pretrained_model/' + ('llama-2-7b' if target == 'llama2' else 'opt-6.7b')
                    for key, expected in dict(model=expected_model, bit=4, group_size=128,
                                              dataset='wikitext2_test', block_size=2048, batch_size=8).items():
                        if result.get(key) != expected:
                            raise ValueError(f'Result protocol mismatch: {out}/{key}')
                rows.append(dict(model=name, method=method, ppl=result.get('ppl'),
                                 status=progress.get('status', 'not_started'),
                                 completed_modules=progress.get('completed_modules', fusion.get('completed_modules')),
                                 source=str(out / 'ppl.json')))
        write(root / 'comparison.json', dict(updated_at=time.strftime('%F %T'),
              protocol='W4-G128, WikiText-2 test, block_size=2048, batch_size=8', rows=rows))
        lines = ['W4-G128，WikiText-2 test；候选中心来自 S=N(d)。融合不再更新中心。', '',
                 '| 模型 | 方法 | PPL ↓ | 状态 |', '|---|---|---:|---|']
        for row in rows:
            value = '—' if row['ppl'] is None else f'{row["ppl"]:.6f}'
            lines.append(f'| {row["model"]} | {row["method"]} | {value} | {row["status"]} |')
        (root / 'comparison.md').write_text('\n'.join(lines) + '\n')
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        (root / 'comparison.csv').write_text(buf.getvalue())
        return rows

    try:
        for target in ('llama2', 'opt'):
            command = [os.sys.executable, '-u', str(Path(__file__).with_name('run_cross_model_delta_only_fusions.py')),
                       '--target', target, '--endpoint-root', str(endpoints / target / 'remove_one'),
                       '--producer-jobs', str(endpoints / 'jobs.json'), '--producer-run', str(endpoints / 'run.json'),
                       '--gpu-lock', str(root / 'gpu.lock'), '--output', str(root / target),
                       '--workers', str(a.workers), '--boost-workers', str(a.boost_workers)]
            log = root / f'{target}.log'
            with log.open('xb') as handle:
                process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            children[target] = process
            jobs[target] = dict(pid=process.pid, command=command, log=str(log), output=str(root / target))
            write(root / 'jobs.json', jobs)
            print('Launched', target, process.pid, flush=True)
        state['status'] = 'running'
        write(root / 'run.json', state)
        while True:
            rows = aggregate()
            failures = {key: child.returncode for key, child in children.items()
                        if child.poll() is not None and child.returncode != 0}
            if failures:
                state.update(status='child_failed', failures=failures)
                write(root / 'run.json', state)
            if all(child.poll() is not None for child in children.values()):
                if failures:
                    raise RuntimeError(f'Fusion children failed: {failures}')
                if all(row['ppl'] is not None and row['status'] == 'complete' for row in rows):
                    break
                if any(row['status'] == 'failed' for row in rows):
                    raise RuntimeError('An endpoint failed; table is incomplete')
            time.sleep(30)
        state.update(status='complete', finished_at=time.strftime('%F %T'))
        write(root / 'run.json', state)
        aggregate()
    except BaseException as exc:
        state.update(status='failed', error=repr(exc))
        write(root / 'run.json', state)
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()

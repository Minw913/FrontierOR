#!/usr/bin/env python3
"""Export the canonical 180-task/12-model source programs, never executing them."""
import argparse
import ast
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import shutil


SUFFIXES = ('codex53', 'deepseekr1', 'fable5', 'gemini31pro', 'glm52',
            'gpt56sol', 'grok420', 'kimik3', 'llama4maverick', 'opus46',
            'qwen38max', 'qwen3coder')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def syntax_error(data):
    try:
        ast.parse(data)
        return None
    except (SyntaxError, ValueError) as error:
        return str(error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1]/'eval/seed_programs')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    papers = sorted(row['paper_id'] for row in json.loads(args.metadata.read_text()))
    assert len(papers) == len(set(papers)) == 180
    entries = []
    counts = Counter()
    csv_hashes = {}
    for suffix in SUFFIXES:
        csv_path = repo/'eval'/f'eval_results_{suffix}.csv'
        csv_hashes[csv_path.name] = digest(csv_path.read_bytes())
        rows = list(csv.DictReader(csv_path.open()))
        models = {row['model'] for row in rows}
        assert len(models) == 1 and len(rows) == 1080
        model = models.pop()
        tiny = {row['paper_id']: row for row in rows if row['instance'] == 'tiny'}
        assert set(tiny) == set(papers)
        for paper in papers:
            source = args.source_root/paper/model
            if not (source/'code.py').is_file() and model == 'grok-4.20':
                source = args.source_root/paper/'grok-4.20-beta'
            assert (source/'code.py').is_file(), str(source)
            destination = args.output/paper/model
            destination.mkdir(parents=True, exist_ok=True)
            files = {}
            for name in ('code.py', 'code_attempt0.py', 'solution_logger.py'):
                path = source/name
                if not path.is_file():
                    assert name == 'code_attempt0.py', str(path)
                    continue
                data = path.read_bytes()
                target = destination/name
                if target.exists() and target.read_bytes() != data:
                    raise RuntimeError(f'Refusing to overwrite different exported source: {target}')
                shutil.copyfile(path, target)
                error = syntax_error(data)
                files[name] = dict(sha256=digest(data), bytes=len(data), syntax_error=error)
                counts[name] += 1
                counts['total_bytes'] += len(data)
                if error:
                    counts[name+'_syntax_errors'] += 1
            entries.append(dict(paper_id=paper, model=model,
                source_relative=str(source.relative_to(args.source_root)),
                tiny_status=tiny[paper]['status'], tiny_feasible=tiny[paper]['feasible'],
                files=files))
    assert len(entries) == 2160 and counts['code.py'] == counts['solution_logger.py'] == 2160
    manifest = dict(version=1, task_count=180, model_count=12, program_count=len(entries),
        source_tree='efficient_opt_bench/eval/eval_papers',
        metadata_sha256=digest(args.metadata.read_bytes()), csv_sha256=csv_hashes,
        counts=dict(counts), programs=entries)
    (args.output/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(counts), indent=2))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""kernel-opt-playbook query tool.

Given an operator CATEGORY (+ optional framework), pull the effective optimization
experiences mined from a large kernel-optimization corpus. Experiences are
anonymized — no original operator names — so you apply the *technique*, not copy a
specific answer.

Usage:
  python3 query.py --list
  python3 query.py --category attention --framework triton
  python3 query.py --category attention --framework triton --tag fusion --top 8
  python3 query.py --category normalization --full        # dump all records (jsonl)
"""
import json, os, argparse
from collections import defaultdict, Counter

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')

def load_index():
    return json.load(open(os.path.join(DATA, 'index.json')))

def load_category(cat):
    p = os.path.join(DATA, 'by_category', f'{cat}.jsonl')
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p)]

def cmd_list():
    idx = load_index()
    print(f"{idx['total_records']} effective experiences across {len(idx['categories'])} operator categories\n")
    print(f"{'category':24s} {'n':>5s}  {'triton':>6s} {'cuda':>5s}  {'med_x':>6s}  subcategories")
    for cat, info in idx['categories'].items():
        fb = info['framework_breakdown']
        subs = ','.join(f"{k}:{v}" for k, v in list(info['subcategories'].items())[:4]) if info.get('subcategories') else '-'
        print(f"{cat:24s} {info['count']:5d}  {fb.get('triton',0):6d} {fb.get('cuda',0):5d}  {str(info['median_speedup']):>6s}  {subs}")

def cmd_query(cat, framework=None, subcategory=None, tag=None, top=6, full=False):
    recs = load_category(cat)
    if not recs:
        print(f"Unknown category: {cat}. Run --list to see options."); return
    if framework:
        recs = [r for r in recs if r['framework'] == framework]
    if subcategory:
        recs = [r for r in recs if r.get('operator_subcategory') == subcategory]
    if tag:
        recs = [r for r in recs if tag in r.get('technique_tags', [])]
    if not recs:
        print("No records match this filter."); return

    if full:
        for r in recs:
            print(json.dumps(r, ensure_ascii=False))
        return

    # rank technique tags by frequency * median speedup
    tag_recs = defaultdict(list)
    for r in recs:
        for t in (r.get('technique_tags') or ['(untagged)']):
            tag_recs[t].append(r)

    def score(rs):
        sp = sorted(x['speedup_vs_prev_kept'] or 1 for x in rs)
        med = sp[len(sp)//2]
        return len(rs) * med, len(rs), med

    ranked = sorted(tag_recs.items(), key=lambda kv: score(kv[1])[0], reverse=True)

    print(f"# {cat}" + (f" / {subcategory}" if subcategory else "") +
          (f" [{framework}]" if framework else "") + f"  —  {len(recs)} effective experiences\n")
    print(f"Techniques ranked by (usage count x median speedup), most reliable first:\n")
    NOISE = ('NOT_RUN', 'NOT RUN', 'BLOCKED', 'HARD-BLOCKED', 'NOT_MEASURED', 'infra')
    def is_noise(r):
        wc = (r.get('what_changed') or '')
        return any(wc.lstrip().upper().startswith(m.upper()) or m in wc[:40] for m in NOISE)
    for t, rs in ranked[:top]:
        _, n, med = score(rs)
        # representative = highest-speedup record with a meaningful description
        clean = [r for r in rs if not is_noise(r) and r.get('what_changed')]
        pool = clean or rs
        best = max(pool, key=lambda r: r.get('speedup_vs_prev_kept') or 0)
        print(f"## {t}   (used {n}x, median {med:.3f}x, max {best.get('speedup_vs_prev_kept')}x)")
        if best.get('what_changed'):
            print(f"  what: {best['what_changed'][:280]}")
        for w in (best.get('why_it_worked') or [])[:1]:
            if w: print(f"  why:  {w[:200]}")
        for p in (best.get('pitfalls') or [])[:1]:
            if p.get('pitfall'): print(f"  pitfall: {p['pitfall'][:200]}")
            if p.get('fix'): print(f"  fix:     {p['fix'][:160]}")
        # one representative code snippet
        for s in (best.get('key_snippets') or [])[:1]:
            code = (s.get('code') or '').strip()
            if code:
                print(f"  snippet [{s.get('label')}]:")
                for line in code.splitlines()[:12]:
                    print(f"    {line}")
        print(f"  (more like this: python3 query.py --category {cat}" +
              (f" --framework {framework}" if framework else "") +
              f" --tag {t} --full)\n")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--category')
    ap.add_argument('--framework', choices=['triton', 'cuda', 'gluon'])
    ap.add_argument('--subcategory')
    ap.add_argument('--tag')
    ap.add_argument('--top', type=int, default=6)
    ap.add_argument('--full', action='store_true')
    a = ap.parse_args()
    if a.list or not a.category:
        cmd_list()
    else:
        cmd_query(a.category, a.framework, a.subcategory, a.tag, a.top, a.full)

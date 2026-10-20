"""CPU self-check: parse every script, run the stdlib self-tests, rebuild both manifests.

usage: python3 serve/cuda/selfcheck.py   (reads $GLIMMER_HOME/config.json and the prompt files)
"""
import ast,json,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import common,g1,g2
D=Path(common.REMOTE_DIR)
S=Path(__file__).parent
for p in S.glob('*.py'):
 ast.parse(p.read_text())
print('AST/imports valid')
for name, passed in common.self_check():
 assert passed,name
for name,passed,detail in common.run_sse_selftest():
 print(name,passed,detail);assert passed,name
cfg=common.load_config(str(D/'config.json'))
prompts=g1.load_prompts_g1(cfg['prompts']['g1_prompts'])
m1=g1.build_manifest(cfg,prompts)
A=len(cfg['arms'])  # 6 items x A arms x 3 tries at 4K, 1 x A x 3 at 32K (84 cells for the 4-arm config)
assert len(m1['cells'])==21*A
assert len({r['cell_id'] for r in m1['cells']})==21*A
assert sum(r['group']=='4K' for r in m1['cells'])==18*A
assert sum(r['group']=='32K' for r in m1['cells'])==3*A
assert all(len(prompts[r['item_id']]['prompt_ids'])==r['n_prompt'] for r in m1['cells'])
assert len(g1.plan_groups(cfg))==A
cfg2=g2.load_config_g2(str(D/'config.json')); ps,der,src=g2.load_prompts_g2(cfg2)
m2=g2.build_manifest_g2(cfg2,ps,der,src)
assert len(m2['cells'])==33
assert len({r['cell_id'] for r in m2['cells']})==33
assert m2['counts']['timed_save_restore']==12
assert m2['counts']['continuations']==12
assert m2['counts']['prefix_requests']==9
assert src.startswith('explicit:')
for m in [m1,m2]:
 print(m['job'],m['counts'],m['manifest_hash'])
print('G2 explicit lengths',{k:len(v['ids']) for k,v in ps.items()})
print('All script hashes',{p.name:common.sha256_file(str(p)) for p in S.glob('*.py')})
print('CPU-INTEGRATION-DONE')

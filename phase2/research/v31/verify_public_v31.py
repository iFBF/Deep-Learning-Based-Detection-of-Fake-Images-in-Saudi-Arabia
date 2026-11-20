"""Check public source and aggregate arithmetic; do not open images or load models."""
from pathlib import Path
import hashlib,json,math


def main():
    root=Path(__file__).resolve().parent
    integrity=json.loads((root/'code_integrity.json').read_text(encoding='utf-8'))
    checks=0
    for name,spec in integrity['frozen_runtime_files'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==spec['sha256'],name
        checks+=1
    candidate=json.loads((root/'canonical_candidate_manifest.json').read_text(encoding='utf-8'))
    assert hashlib.sha256((root/'canonical_candidate_manifest.json').read_bytes()).hexdigest()==integrity['canonical_candidate_manifest_sha256']
    for name,spec in integrity['frozen_runtime_files'].items():
        assert candidate['files_sha256'][spec['canonical_candidate_file']]==spec['sha256'];checks+=1
    portal=json.loads((root/'portal/portal_manifest.json').read_text())
    for name,expected in portal['files_sha256'].items():
        assert hashlib.sha256((root/'portal'/name).read_bytes()).hexdigest()==expected,name
        checks+=1
    results=json.loads((root/'development_results.json').read_text(encoding='utf-8'))
    def close(a,b):
        nonlocal checks
        assert math.isclose(a,b,abs_tol=1e-12,rel_tol=0),(a,b)
        checks+=1
    def matrix(m):
        tn,fp=m['confusion_matrix'][0];fn,tp=m['confusion_matrix'][1]
        n=tn+fp+fn+tp
        if 'n' in m:assert m['n']==n
        close(m['accuracy'],(tn+tp)/n)
        close(m['macro_f1'],(2*tp/(2*tp+fp+fn)+2*tn/(2*tn+fp+fn))/2)
        close(m['balanced_accuracy'],(tp/(tp+fn)+tn/(tn+fp))/2)
        close(m['fake_recall'],tp/(tp+fn));close(m['real_recall'],tn/(tn+fp))
        return n
    for key in ('baseline','rejected_V2','v31'):
        for role,m in results[key].items():assert matrix(m)=={'validation_original':233,'validation_self72':72}[role]
    for role,m in results['v3']['v3'].items():assert matrix(m)=={'validation_original':233,'validation_self72':72}[role]
    for key in ('baseline','v3'):matrix(results['CSI_live_V3'][key])
    matrix(results['CSI_V31_cached'])
    assert results['CSI_V31_cached']['confusion_matrix']==results['CSI_live_V3']['baseline']['confusion_matrix']
    assert results['prior_V3_joint_gate_passed'] is False
    assert results['fresh_test_evaluated'] is False and results['original_tests_reopened'] is False
    assert results['V31_new_live_cohort_run'] is False
    policy=json.loads((root/'policy.json').read_text())
    assert policy['swin_threshold']==.99 and policy['supplemental_raw_logit_boundary']==4.25
    print('V31_PUBLIC_VERIFIED:',checks,'source/hash/aggregate checks passed.')
    print('No images, private prediction caches or models loaded; no new test evaluated.')


if __name__=='__main__':main()

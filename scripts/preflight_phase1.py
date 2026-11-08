"""Show verified corrections separately from GPU/human work that remains."""
from pathlib import Path
import json,sys
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from phase1_corrections import audit_report

root=Path(__file__).resolve().parents[1];evidence=root/'docs/phase1_corrections/evidence'
audit=audit_report(pd.read_csv(evidence/'label_audit_fixed_sample.csv'),pd.read_csv(evidence/'label_audit.csv'))
groups=json.loads((evidence/'source_grouping_report.json').read_text())
status={'readable_count_reported':True,'verified_caption_links':groups['linked_original_rows']==1800,
        'corrected_parent_groups':groups['retained_parent_coco_groups'],'audit':audit,
        'corrected_connected_source_groups':groups['retained_connected_source_groups'],
        'new_holdout_acquisition_snapshot':json.loads((evidence/'protocol/holdout_verification.json').read_text()),
        'historical_error_analysis_completed':(evidence/'errors/error_slices.csv').exists(),
        'requirements_lock_supplied':(root/'requirements-lock.txt').exists(),
        'corrected_project_gpu_training':('completed main Kaggle run; see completed_run/' if (root/'docs/phase1_corrections/completed_run/final_test_metrics.csv').is_file() else 'pending Kaggle run'),
        'final_evaluation_status':('completed; this cohort is closed to tuning' if (root/'docs/phase1_corrections/completed_run/final_test_metrics.csv').is_file() else 'not yet completed'),
        'independent_teammate_reproduction':'pending another member actual execution'}
print(json.dumps(status,indent=2))
if not audit['complete']:raise SystemExit('The fixed audit is not complete')

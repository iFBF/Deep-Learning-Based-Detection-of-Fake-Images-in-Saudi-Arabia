"""Recompute recorded main-run results without images, weights, GPU or training.

This verifies saved predictions and protocol records. It is not a new inference
run and does not satisfy the independent-member training requirement.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from phase1_corrections import prediction_metrics, grouped_accuracy_interval


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(run):
    environment = json.loads((run/'environment.json').read_text())
    config = json.loads((run/'config.json').read_text())
    lock = json.loads((run/'selection_locked.json').read_text())
    require(digest(run/'config.json') == lock['config_sha256'], 'Configuration hash mismatch')
    require(digest(run/'experiment_manifest.csv') == lock['protocol_sha256'], 'Protocol hash mismatch')
    for name, expected in environment['code_hashes'].items():
        require(digest(ROOT/name) == expected, f'Training source hash mismatch: {name}')
    require(digest(run/'final_test_metrics.csv') == (run/'final_test_metrics.sha256').read_text().strip(), 'Metrics hash mismatch')
    protocol = pd.read_csv(run/'experiment_manifest.csv')
    require(protocol.groupby('group_id').split.nunique().max() == 1, 'Source group crosses splits')
    test = protocol[protocol.split.eq('test')].reset_index(drop=True)
    require(len(test) == 328 and test.group_id.nunique() == 68, 'Unexpected main test population')
    validation = pd.read_csv(run/'validation_comparison.csv')
    selected = validation.sort_values(['val_loss','mode']).iloc[0]['mode']
    require(selected == lock['selected_mode'], 'Selected strategy differs from validation rule')
    opened = json.loads((run/'test_evaluation_started.json').read_text())
    from datetime import datetime
    require(datetime.fromisoformat(lock['locked_at_utc']) < datetime.fromisoformat(opened['started_at_utc']), 'Test precedes model-selection lock')
    require(opened['selection_sha256'] == digest(run/'selection_locked.json'), 'Test selection-record hash differs')
    balanced=set()
    for _, part in test.groupby('label'):
        balanced.update(part.sample(n=int(test.label.value_counts().min()),random_state=config['seed']).relative_path)
    recorded = pd.read_csv(run/'final_test_metrics.csv').set_index(['mode','population'])
    results=[]
    for mode in ['scratch_unweighted','scratch','frozen','full']:
        prediction_path=run/mode/'final_test_predictions.csv'
        require(digest(prediction_path) == prediction_path.with_suffix('.sha256').read_text().strip(), f'{mode}: prediction hash mismatch')
        p=pd.read_csv(prediction_path)
        require(p.relative_path.tolist()==test.relative_path.tolist(), f'{mode}: prediction identities changed')
        require(p.label.tolist()==test.label.tolist() and p.group_id.tolist()==test.group_id.tolist(), f'{mode}: labels/groups changed')
        require(p.prediction.eq((p.prob_fake>=config['threshold']).astype(int)).all(), f'{mode}: threshold mismatch')
        require(np.isfinite(p.prob_fake).all() and p.prob_fake.between(0,1).all(), f'{mode}: invalid probabilities')
        history=pd.read_csv(run/mode/'history.csv')
        best=history.loc[history.val_loss.idxmin()]
        saved=validation[validation['mode'].eq(mode)].iloc[0]
        require(int(saved.epoch)==int(best.epoch) and np.isclose(saved.val_loss,best.val_loss,atol=1e-12,rtol=0), f'{mode}: validation checkpoint mismatch')
        for population,part in [('full_unbalanced',p),('balanced_secondary',p[p.relative_path.isin(balanced)])]:
            calculated=prediction_metrics(part)
            low,high=grouped_accuracy_interval(part,seed=config['seed'])
            confidence=np.maximum(part.prob_fake.to_numpy(),1-part.prob_fake.to_numpy())
            correct=part.label.eq(part.prediction).to_numpy()
            bins=np.minimum((confidence*10).astype(int),9)
            ece=sum((bins==b).mean()*abs(correct[bins==b].mean()-confidence[bins==b].mean()) for b in np.unique(bins))
            actual={**{k:v for k,v in calculated.items() if k!='confusion_matrix'},'accuracy_ci95_low':low,'accuracy_ci95_high':high,'ece':ece}
            row=recorded.loc[(mode,population)]
            for field,value in actual.items():
                require(np.isclose(value,float(row[field]),atol=1e-10,rtol=0),f'{mode}/{population}: {field} differs')
            results.append({'mode':mode,'population':population,**actual})
    return environment,pd.DataFrame(results)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=ROOT/'docs/phase1_corrections/completed_run')
    args=parser.parse_args()
    env, table=verify(args.run.resolve())
    print('Recorded-prediction verification: OK')
    print('Main run:',env['run_id'])
    print(table[table.population.eq('full_unbalanced')][['mode','n','accuracy','macro_f1','real_recall','fake_recall','ece']].to_string(index=False))
    print('All eight primary/secondary metric rows and source/configuration records agree.')
    print('This check does not load weights, rerun inference or replace independent training.')

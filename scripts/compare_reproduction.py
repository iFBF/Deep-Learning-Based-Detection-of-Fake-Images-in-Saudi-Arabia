"""Record another member's real independent run beside the original; never synthesize a run."""
import argparse,json,sys
from pathlib import Path
from datetime import datetime,timezone
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from phase1_training import digest
from phase1_corrections import dump

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original',required=True);p.add_argument('--rerun',required=True)
    p.add_argument('--member',required=True);p.add_argument('--output',default='outputs/reproduction')
    p.add_argument('--independent-run-confirmed',action='store_true',required=True)
    a=p.parse_args();original=Path(a.original).resolve();rerun=Path(a.rerun).resolve();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if original==rerun:raise ValueError('An independent run must have a different output directory')
    env1=json.loads((original/'environment.json').read_text());env2=json.loads((rerun/'environment.json').read_text())
    if env1['run_id']==env2['run_id']:raise ValueError('Copied run ID; rerun training in a fresh runtime')
    if not env2.get('operator') or env1.get('operator')==env2.get('operator'):
        raise ValueError('Record a different team member as the operator of the independent run')
    if a.member.strip().casefold()!=env2['operator'].strip().casefold():
        raise ValueError('Member name must match the operator recorded in the rerun metadata')
    if env1['code_hashes']!=env2['code_hashes']:raise ValueError('Different source code cannot establish reproduction of this implementation')
    lock1=json.loads((original/'selection_locked.json').read_text());lock2=json.loads((rerun/'selection_locked.json').read_text())
    if lock1['protocol_sha256']!=lock2['protocol_sha256']:raise ValueError('Different datasets/protocols cannot establish reproduction')
    if digest(original/'config.json')!=digest(rerun/'config.json'):raise ValueError('Configuration differs')
    x=pd.read_csv(original/'final_test_metrics.csv');y=pd.read_csv(rerun/'final_test_metrics.csv')
    result=x.merge(y,on=['mode','population'],suffixes=('_original','_rerun'),validate='one_to_one')
    for field in ['accuracy','macro_f1','ece']:result[field+'_difference']=result[field+'_rerun']-result[field+'_original']
    result.to_csv(out/'independent_reproduction_comparison.csv',index=False)
    dump(out/'independent_reproduction_record.json',{'member':a.member,'confirmed_at_utc':datetime.now(timezone.utc).isoformat(),
        'original_run':env1['run_id'],'rerun':env2['run_id'],'protocol_sha256':lock1['protocol_sha256'],
        'statement':'Member attestation of execution in a fresh runtime; software checks alone do not authenticate human independence.'})
    print(result[['mode','population','accuracy_original','accuracy_rerun','macro_f1_original','macro_f1_rerun']].to_string(index=False))

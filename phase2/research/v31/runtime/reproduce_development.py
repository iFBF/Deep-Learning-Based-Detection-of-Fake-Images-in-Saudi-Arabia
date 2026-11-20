"""Replay the frozen V3 global-classifier development comparison; never evaluate test."""
from pathlib import Path
from io import BytesIO
from urllib.request import urlopen
import argparse,hashlib,json,sys,time,importlib.metadata
import numpy as np,pandas as pd
from PIL import Image
from predict_v3 import verify_release
from v3_preprocess import preprocess
from enhance_data import prepare,sha,dump


def flag_metrics(labels,prediction):
    y=np.asarray(labels,dtype=int);p=np.asarray(prediction,dtype=bool)
    tp=int(((y==1)&p).sum());tn=int(((y==0)&~p).sum());fp=int(((y==0)&p).sum());fn=int(((y==1)&~p).sum())
    safe=lambda a,b:a/b if b else 0.
    return {'n':len(y),'accuracy':float((p==y).mean()),'macro_f1':(safe(2*tp,2*tp+fp+fn)+safe(2*tn,2*tn+fp+fn))/2,
            'balanced_accuracy':(safe(tp,tp+fn)+safe(tn,tn+fp))/2,'fake_recall':safe(tp,tp+fn),'real_recall':safe(tn,tn+fp),
            'confusion_matrix':[[tn,fp],[fn,tp]]}


def get_checkpoint(manifest,cache):
    target=Path(cache)/'frontier_model.safetensors'
    expected=manifest['supplemental_weight_sha256']
    if target.is_file() and sha(target)==expected:return target
    url='https://huggingface.co/'+manifest['supplemental_model_repo']+'/resolve/'+manifest['supplemental_model_revision']+'/model.safetensors'
    temporary=target.with_suffix('.downloading')
    with urlopen(url,timeout=60) as response,temporary.open('wb') as output:
        while True:
            block=response.read(4*1024*1024)
            if not block:break
            output.write(block)
    if sha(temporary)!=expected:raise ValueError('Public checkpoint checksum mismatch.')
    temporary.replace(target);return target


def run(root,inputs,output):
    import torch,timm
    from safetensors.torch import load_file
    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    manifest,policy=verify_release(root)
    # Baseline results are the supplied original-run CSVs, not a new private-model forward.
    frame,_=prepare(inputs,output,require_baseline=False)
    frame=frame.loc[frame.role.isin(['validation_original','validation_self72'])].reset_index(drop=True)
    assert frame.role.value_counts().to_dict()=={'validation_original':233,'validation_self72':72}
    parts={role:pd.read_csv(root/'evidence'/('baseline_'+role+'.csv')) for role in frame.role.unique()}
    for role,part in parts.items():
        if frame.loc[frame.role.eq(role),['id','label','group_id','source']].to_dict('records')!=part[['id','label','group_id','source']].to_dict('records'):
            raise ValueError('Baseline/candidate cohort identity differs.')
    spec={'release_manifest_sha256':sha(root/'manifest.json'),'preflight_sha256':sha(output/'preflight.json'),
          'device':'CUDA' if torch.cuda.is_available() else 'CPU','thresholds':policy,'no_training':True,'fresh_test_evaluated':False,'legacy_test_evaluated':False,
          'versions':{name:importlib.metadata.version(name) for name in ['torch','torchvision','timm','safetensors','Pillow','numpy','pandas','scipy']}}
    if (output/'REPLAY_SPEC.json').exists() and json.loads((output/'REPLAY_SPEC.json').read_text())!=spec:raise ValueError('Replay specification changed. Use a new output directory.')
    dump(output/'REPLAY_SPEC.json',spec)
    if (output/'RESULTS_LOCK.json').exists():
        for name,digest in json.loads((output/'RESULTS_LOCK.json').read_text())['files_sha256'].items():
            if sha(output/name)!=digest:raise ValueError('Completed replay result changed: '+name)
        print('Existing verified replay retained. No new inference or test evaluation.');return
    torch.set_num_threads(2);torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=timm.create_model('vit_small_patch16_384.augreg_in21k_ft_in1k',pretrained=False,num_classes=1).eval()
    for block in model.blocks:block.attn.fused_attn=False
    model.load_state_dict(load_file(str(get_checkpoint(manifest,output/'cache')),device='cpu'),strict=True)
    model.requires_grad_(False);model.to(device)
    values=[]
    for start in range(0,len(frame),2):
        batch=[]
        for row in frame.iloc[start:start+2].itertuples():
            with Image.open(row.path) as image:batch.append(preprocess(image)[0])
        with torch.inference_mode():values.extend(float(x) for x in model(torch.from_numpy(np.stack(batch)).to(device)).reshape(-1).cpu().numpy())
        if start%40==0:print('V3 development replay',min(start+len(batch),len(frame)),'/',len(frame),flush=True)
    frame['raw_logit']=values
    reference=pd.read_csv(root/'evidence/v3_reference_predictions.csv').set_index('id').reindex(frame.id)
    stable=lambda a:np.exp(-np.logaddexp(0.,-np.asarray(a,dtype=float)))
    maximum=float(np.max(np.abs(stable(frame.raw_logit)-stable(reference.raw_logit))))
    agreement=float(np.mean((frame.raw_logit.to_numpy()>=policy['supplemental_raw_logit_boundary'])==(reference.raw_logit.to_numpy()>=policy['supplemental_raw_logit_boundary'])))
    if maximum>1e-4 or agreement!=1.:raise ValueError('Replay differs from saved reference. Inspect the numerical runtime before claiming reproduction.')
    summary={'scope':'Global classifier development replay only; cached baseline scores; no joint-service or independent-test evaluation.',
             'score_parity_max_difference':maximum,'selected_label_agreement':agreement,'baseline':{},'v3':{},'automatic_deployment_approved':False}
    records=[]
    for role,part in frame.groupby('role'):
        base=parts[role];swin=base.fake_score.to_numpy();prediction=(swin>=policy['swin_threshold'])|(part.raw_logit.to_numpy()>=policy['supplemental_raw_logit_boundary'])
        summary['baseline'][role]=flag_metrics(part.label,swin>=.5);summary['v3'][role]=flag_metrics(part.label,prediction)
        result=part[['id','label','group_id','role','source','raw_logit']].copy();result['baseline_fake_score']=swin;result['fusion_prediction']=prediction.astype(int);records.append(result)
    dump(output/'development_results.json',summary);pd.concat(records).to_csv(output/'development_predictions.csv',index=False)
    dump(output/'RESULTS_LOCK.json',{'files_sha256':{n:sha(output/n) for n in ['development_results.json','development_predictions.csv']},'fresh_test_evaluated':False})
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(Path(__file__).resolve().parent,args.input_root,args.output)

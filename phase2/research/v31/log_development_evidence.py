"""Log historical cached development evidence; no training or new image inference."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,subprocess


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def verify_candidate(root,expected):
    root=Path(root).resolve()
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    if manifest['release_id']!=expected['release_id']:raise ValueError('Wrong candidate release.')
    if sha(root/'manifest.json')!=expected['canonical_candidate_manifest_sha256']:raise ValueError('Wrong candidate manifest.')
    files=[]
    for name,digest in manifest['files_sha256'].items():
        p=(root/name).resolve()
        if not p.is_relative_to(root) or not p.is_file() or sha(p)!=digest:raise ValueError('Candidate integrity failed: '+name)
        files.append(p)
    return manifest,files+[root/'manifest.json']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--receipt',type=Path,required=True)
    parser.add_argument('--entity',default='f1k80-kaust')
    parser.add_argument('--project',default='Real-Abshar-Phase2')
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    expected=json.loads((root/'code_integrity.json').read_text())
    manifest,files=verify_candidate(args.candidate,expected)
    aggregate=json.loads((root/'development_results.json').read_text(encoding='utf-8'))
    if args.receipt.exists():
        prior=json.loads(args.receipt.read_text(encoding='utf-8'))
        if prior.get('status')=='confirmed' and prior.get('release_id')==manifest['release_id']:
            print('Existing confirmed receipt retained:',args.receipt)
            print('W&B:',prior['run_url']);return
        raise ValueError('Receipt path already contains a different/incomplete record. Choose a new path.')
    try:
        commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    except (OSError,subprocess.CalledProcessError):commit=None
    import wandb
    record_id='v31-01a330b634e5-review'
    # Do not label replayed metrics as a new training or test run.
    with wandb.init(entity=args.entity,project=args.project,id=record_id,resume='allow',
                    job_type='historical-development-evidence',name='V3.1 cached development and Windows smoke audit',
                    config={'release_id':manifest['release_id'],'source_commit':commit,
                            'evidence_kind':'Previously computed validation/development and user-reported known-image Windows smoke',
                            'swin_input':224,'supplemental_input':384,'local_model_input':640,
                            'swin_threshold':.99,'supplemental_raw_logit_boundary':4.25,
                            'fresh_test_evaluated':False,'new_training_performed':False,
                            'new_inference_performed_by_this_logging_script':False}) as run:
        for role,m in aggregate['v31'].items():
            for metric in ('n','accuracy','macro_f1','balanced_accuracy','fake_recall','real_recall'):
                run.summary['cached_v31/'+role+'/'+metric]=m[metric]
        for name,m in [('baseline',aggregate['CSI_live_V3']['baseline']),('V3_raw4_actual',aggregate['CSI_live_V3']['v3']),('V31_raw4_25_cached',aggregate['CSI_V31_cached'])]:
            for metric in ('n','accuracy','macro_f1','fake_recall','real_recall'):
                run.summary['CSI_validation/'+name+'/'+metric]=m[metric]
        run.summary['original_V3_joint_gate_passed']=False
        run.summary['new_V31_live_cohort_evaluation_performed']=False
        run.summary['automatic_authenticity_approved']=False
        artifact=wandb.Artifact('real-abshar-v31-research-01a330b634e5',type='model',
                                metadata={'release_id':manifest['release_id'],'third_party_supplemental_weights':True,
                                          'dataset_image_or_mask_files_included':False,'new_training_performed':False,
                                          'new_test_evaluated':False})
        candidate=Path(args.candidate).resolve()
        for p in files:artifact.add_file(str(p),name='candidate/'+p.relative_to(candidate).as_posix())
        for name in ('development_results.json','windows_demo_summary.json','code_integrity.json','README.md'):
            artifact.add_file(str(root/name),name='public_evidence/'+name)
        logged=run.log_artifact(artifact,aliases=['v31-frozen','latest'])
        logged.wait(timeout=300)
        receipt={'status':'confirmed','release_id':manifest['release_id'],'source_commit':commit,
                 'run_id':run.id,'run_url':run.url,'artifact':logged.qualified_name,'artifact_url':logged.url,
                 'confirmed_utc':datetime.now(timezone.utc).isoformat(),'fresh_test_evaluated':False,
                 'new_training_performed':False,'new_inference_performed':False,
                 'canonical_candidate_manifest_sha256':expected['canonical_candidate_manifest_sha256']}
        args.receipt.parent.mkdir(parents=True,exist_ok=True)
        args.receipt.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
        print('V31_EVIDENCE_UPLOAD_CONFIRMED:',run.url)
        print('SEND RECEIPT:',args.receipt)


if __name__=='__main__':main()

"""Evaluate only the locked CSI validation cohort through the original local API."""
from pathlib import Path
from io import BytesIO
import argparse,csv,hashlib,json,time,zipfile,platform,importlib.metadata
import numpy as np
from PIL import Image
import cohort_data
from predict_v3 import load_runtime,predict
from v3_metrics import flag_metrics


def dump(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.writing')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8');temporary.replace(path)


def prepared_root(inputs,explicit=None):
    if explicit:
        root=Path(explicit);cohort_data.read_preparation(root);return root
    candidates=[]
    for file in Path(inputs).rglob('manifest.csv'):
        root=file.parent
        if all((root/name).is_dir() for name in ['images','labels','segmentation_masks']) and cohort_data.sha_file(file)==cohort_data.SOURCE_MANIFEST_SHA256:
            cohort_data.read_preparation(root);candidates.append(root)
    if len(candidates)==1:return candidates[0]
    if candidates:raise ValueError('Multiple prepared datasets found; set --prepared-root explicitly: '+str(candidates))
    raise FileNotFoundError('Attach or extract the PRIVATE Real-Abshar_Phase2_Prepared_Data.zip. The original CSI download and small preparation-review ZIP are not the prepared dataset.')


def run(release,inputs,output,url,explicit=None):
    release=Path(release);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    runtime=load_runtime(release);root=prepared_root(inputs,explicit)
    all_rows,_=cohort_data.read_preparation(root)
    rows=[row for row in all_rows if row['split']=='val']
    if len(rows)!=126 or sum(r['kind']=='edited' for r in rows)!=105 or len({r['source_id'] for r in rows})!=21:raise ValueError('The fixed CSI validation cohort differs.')
    spec={'stage':'V3 joint CSI validation only','release_manifest_sha256':cohort_data.sha_file(release/'manifest.json'),
          'source_manifest_sha256':cohort_data.SOURCE_MANIFEST_SHA256,'cohort_uid_sha256':cohort_data.COHORT_UID_SHA256,
          'validation_uid_sha256':hashlib.sha256(json.dumps([r['uid'] for r in rows],separators=(',',':')).encode()).hexdigest(),
          'split':'val','images':126,'edited':105,'pristine':21,'sources':21,'baseline_url':url,'policy':runtime['policy'],
          'resolutions':{'Swin':[224,224],'supplemental':[384,384],'detector_and_segmenter':[640,640]},
          'local_thresholds_unchanged':{'box_confidence':.35,'NMS_IoU':.45,'max_detections':50,'mask_pixel':.5,'edited_area_fraction':.005},
          'gates':{'maximum_extra_pristine_false_referrals':1,'edited_recall_must_not_drop':True,'macro_f1_must_not_drop':True,'V3_p95_limit':'baseline HTTP p95 * 1.5 + 250 ms'},
          'test_image_or_mask_pixels_read':False,'fresh_test_evaluated':False,'automatic_authenticity_approved':False,
          'client_environment':{'python':platform.python_version(),'platform':platform.platform(),'onnx_threads':2,
                                'versions':{p:importlib.metadata.version(p) for p in ['numpy','Pillow','onnxruntime']}}}
    path=output/'RUN_SPEC.json'
    if path.exists() and json.loads(path.read_text())!=spec:raise ValueError('Joint validation protocol changed. Use a new output directory.')
    dump(path,spec)
    if (output/'RESULTS_LOCK.json').exists():
        for name,digest in json.loads((output/'RESULTS_LOCK.json').read_text())['files_sha256'].items():
            if cohort_data.sha_file(output/name)!=digest:raise ValueError('Completed joint-validation result changed: '+name)
        print('Using existing locked CSI validation result. No test evaluation.');return
    for i,row in enumerate(rows):
        cohort_data.verify_row(root,row)
        if i%24==0:print('CSI validation integrity',i+1,'/',len(rows),flush=True)
    dump(output/'preflight.json',{'validation_images_and_masks_verified':126,'test_images_and_masks_decoded':0,'selection_before_evaluation':spec['gates']})
    # Warm only with a generated uniform image, never a held-out photograph.
    stream=BytesIO();Image.new('RGB',(320,256),(127,127,127)).save(stream,format='PNG')
    predict(release,stream.getvalue(),url,runtime=runtime)
    completed=[];partial=output/'validation_predictions.csv';partial_lock=output/'PARTIAL_LOCK.json'
    if partial.exists():
        if not partial_lock.is_file() or cohort_data.sha_file(partial)!=json.loads(partial_lock.read_text())['sha256']:raise ValueError('Incomplete result checksum differs.')
        completed=list(csv.DictReader(partial.open(encoding='utf-8',newline='')))
        if [r['uid'] for r in completed]!=[r['uid'] for r in rows[:len(completed)]]:raise ValueError('Incomplete validation identities differ.')
    for i,row in enumerate(rows):
        if i<len(completed):continue
        result=predict(release,cohort_data.safe_path(root,row['export_image']).read_bytes(),url,runtime=runtime)
        base=result['baseline_summary'];decision=result['decision'];timing=result['timing_ms']
        completed.append({'uid':row['uid'],'source_id':row['source_id'],'kind':row['kind'],'label':int(row['kind']=='edited'),
                          'baseline_referral':int(base['decision']['review_recommended']),'v3_referral':int(decision['review_recommended']),
                          'baseline_swin_score':base['classifier']['fake_score'],'supplemental_raw_logit':result['supplemental_classifier']['raw_logit'],
                          'v3_swin_flag':int(decision['flags']['swin_generation']),'supplemental_flag':int(decision['flags']['supplemental_generation']),
                          'box_flag':int(decision['flags']['edited_region_box']),'area_flag':int(decision['flags']['edited_area']),
                          'baseline_http_ms':timing['baseline_http_roundtrip'],'v3_client_ms':timing['client_total']})
        temporary=partial.with_suffix('.writing.csv')
        with temporary.open('w',encoding='utf-8',newline='') as file:
            writer=csv.DictWriter(file,fieldnames=list(completed[0]));writer.writeheader();writer.writerows(completed)
        temporary.replace(partial);dump(partial_lock,{'sha256':cohort_data.sha_file(partial),'rows':len(completed)})
        if (i+1)%8==0 or i+1==len(rows):print('Joint CSI validation requests',i+1,'/',len(rows),flush=True)
    y=np.asarray([int(r['label']) for r in completed]);base=np.asarray([int(r['baseline_referral']) for r in completed]);v3=np.asarray([int(r['v3_referral']) for r in completed])
    baseline=flag_metrics(y,base);candidate=flag_metrics(y,v3)
    old_fp=baseline['confusion_matrix'][0][1];new_fp=candidate['confusion_matrix'][0][1]
    old_p95=float(np.percentile([float(r['baseline_http_ms']) for r in completed],95));new_p95=float(np.percentile([float(r['v3_client_ms']) for r in completed],95))
    checks={'pristine_false_referrals':new_fp<=old_fp+1,'edited_recall':candidate['fake_recall']>=baseline['fake_recall'],
            'macro_f1':candidate['macro_f1']>=baseline['macro_f1'],'latency':new_p95<=old_p95*1.5+250}
    summary={'split':'CSI validation, n=126','baseline':baseline,'v3':candidate,'baseline_http_p95_ms':old_p95,'v3_warm_client_p95_ms':new_p95,
             'gate_checks':checks,'joint_preservation_gates_passed':all(checks.values()),'fresh_test_evaluated':False,'automatic_authenticity_approved':False,
             'note':'Joint-validation gates do not override the unmet 90% additional-source target or establish an independent test result.'}
    dump(output/'joint_validation_results.json',summary)
    dump(output/'RESULTS_LOCK.json',{'files_sha256':{n:cohort_data.sha_file(output/n) for n in ['RUN_SPEC.json','preflight.json','validation_predictions.csv','joint_validation_results.json']}})
    archive=output/'Real-Abshar_V3_1_Joint_Validation_Review.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name in ['RUN_SPEC.json','preflight.json','validation_predictions.csv','joint_validation_results.json','RESULTS_LOCK.json']:z.write(output/name,name)
    print(json.dumps(summary,indent=2),flush=True);print('SEND HERE:',archive,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root',type=Path,required=True);parser.add_argument('--prepared-root',type=Path)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--baseline-url',default='http://127.0.0.1:8000')
    args=parser.parse_args();run(Path(__file__).resolve().parent,args.input_root,args.output,args.baseline_url,args.prepared_root)

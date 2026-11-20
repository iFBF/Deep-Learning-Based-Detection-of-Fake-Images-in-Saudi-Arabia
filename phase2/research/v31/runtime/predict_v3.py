"""Research comparison client; preserves the original service and its model files."""
from pathlib import Path
import argparse,hashlib,json,math,time,uuid
from urllib.parse import urlsplit
from urllib.request import Request,urlopen
import numpy as np
from v3_preprocess import decode,preprocess


def sigmoid(value):
    if value>=0:return 1/(1+math.exp(-value))
    value=math.exp(value);return value/(1+value)


def verify_release(root):
    root=Path(root).resolve();manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    for name,digest in manifest['files_sha256'].items():
        path=(root/name).resolve()
        if not path.is_relative_to(root):raise ValueError('Unsafe release path.')
        with path.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
        if actual!=digest:raise ValueError('Research release integrity failed: '+name)
    return manifest,json.loads((root/'policy.json').read_text(encoding='utf-8'))


def baseline_predict(url,data):
    parsed=urlsplit(url)
    if parsed.scheme!='http' or parsed.hostname not in {'localhost','127.0.0.1','::1'} or parsed.username or parsed.password or parsed.path not in {'','/'} or parsed.query or parsed.fragment:
        raise ValueError('Use the local original service URL, for example http://127.0.0.1:8000.')
    boundary='realabshar'+uuid.uuid4().hex
    body=(('--'+boundary+'\r\nContent-Disposition: form-data; name="file"; filename="image.bin"\r\nContent-Type: application/octet-stream\r\n\r\n').encode()+data+('\r\n--'+boundary+'--\r\n').encode())
    request=Request(url.rstrip('/')+'/predict',data=body,headers={'Content-Type':'multipart/form-data; boundary='+boundary},method='POST')
    with urlopen(request,timeout=180) as response:return json.load(response)


def fuse(baseline,raw,policy,image):
    if baseline.get('release_id')!=policy['baseline_release_id']:raise ValueError('The original baseline service release ID differs from the locked policy.')
    if baseline.get('image',{}).get('width')!=image['width'] or baseline.get('image',{}).get('height')!=image['height']:raise ValueError('Original service image dimensions differ.')
    score=float(baseline['classifier']['fake_score'])
    if not math.isfinite(score) or not 0<=score<=1:raise ValueError('Invalid original Swin score.')
    old=baseline['decision']['flags']
    if not all(isinstance(old.get(k),bool) for k in ['edited_region_box','edited_area']):raise ValueError('Original box/area decision flags are missing.')
    flags={'swin_generation':score>=policy['swin_threshold'],
           'supplemental_generation':raw>=policy['supplemental_raw_logit_boundary'],
           'edited_region_box':old['edited_region_box'],'edited_area':old['edited_area']}
    return {'status':'review_recommended' if any(flags.values()) else 'no_model_flag',
            'review_recommended':any(flags.values()),'reasons':[k for k,v in flags.items() if v], 'flags':flags,
            'global_generation_flag':flags['swin_generation'] or flags['supplemental_generation'],
            'authenticity_verified':False,'research_only':True,'joint_service_validation_complete':False,
            'policy_validation_scope':policy.get('joint_validation_status','unknown'),
            'note':'Research screening signals for human review. No model flag does not establish authenticity.'}


def load_runtime(root):
    import onnxruntime as ort
    manifest,policy=verify_release(root)
    opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1
    session=ort.InferenceSession(str(Path(root)/'supplemental_classifier.onnx'),sess_options=opts,providers=['CPUExecutionProvider'])
    return {'root':Path(root).resolve(),'manifest':manifest,'policy':policy,'session':session}


def predict(root,data,baseline_url='http://127.0.0.1:8000',supplemental_only=False,runtime=None):
    started=time.perf_counter();includes_load=runtime is None
    runtime=load_runtime(root) if runtime is None else runtime
    if runtime['root']!=Path(root).resolve():raise ValueError('Runtime belongs to a different research release.')
    manifest,policy,session=runtime['manifest'],runtime['policy'],runtime['session']
    image,info=decode(data);x=preprocess(image)
    inferred=time.perf_counter()
    raw=float(session.run(None,{'pixel_values':x})[0].reshape(-1)[0])
    inference_ms=1000*(time.perf_counter()-inferred);baseline_ms=None
    if not math.isfinite(raw):raise ValueError('Nonfinite supplemental model output.')
    flag=raw>=policy['supplemental_raw_logit_boundary']
    result={'release_id':manifest['release_id'],'policy_version':policy['version'],'scope':'Supplemental-only diagnostic' if supplemental_only else 'Experimental four-model screening through the unchanged original local service',
            'image':info,'supplemental_classifier':{'flag':flag,'raw_logit':raw,'selected_raw_logit_boundary':policy['supplemental_raw_logit_boundary'],
                'publisher_calibrated_score':sigmoid(raw+policy['upstream_calibration_bias']),
                'selected_publisher_score_threshold':sigmoid(policy['supplemental_raw_logit_boundary']+policy['upstream_calibration_bias']),
                'local_probability_calibration_verified':False,'model_repo':manifest['supplemental_model_repo'],'model_revision':manifest['supplemental_model_revision'],
                'input_resolution':[384,384]},
            'automatic_deployment_approved':False,'fresh_test_evaluated':False}
    if supplemental_only:
        result['decision']={'status':'supplemental_generation_flag' if flag else 'no_supplemental_flag',
                            'authenticity_verified':False,'research_only':True,'note':'Original Swin, detection and segmentation were not run in this diagnostic.'}
    else:
        requested=time.perf_counter();baseline=baseline_predict(baseline_url,data);baseline_ms=1000*(time.perf_counter()-requested)
        result['decision']=fuse(baseline,raw,policy,info)
        result['baseline_summary']={'release_id':baseline['release_id'],'classifier':baseline['classifier'],
                                    'detections':baseline['detections'],'mask':{k:v for k,v in baseline['mask'].items() if k!='png_base64'},
                                    'decision':baseline['decision']}
        result['selected_swin_threshold']=policy['swin_threshold']
    result['timing_ms']={'client_total':1000*(time.perf_counter()-started),'includes_supplemental_model_load':includes_load,
                         'supplemental_inference':inference_ms,'baseline_http_roundtrip':baseline_ms}
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image',type=Path,required=True)
    parser.add_argument('--release',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--baseline-url',default='http://127.0.0.1:8000')
    parser.add_argument('--supplemental-only',action='store_true')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.image.stat().st_size>10*1024*1024:raise ValueError('Use an image file of at most 10 MiB.')
    result=predict(args.release,args.image.read_bytes(),args.baseline_url,args.supplemental_only)
    text=json.dumps(result,indent=2,allow_nan=False)
    if args.output:args.output.write_text(text+'\n',encoding='utf-8')
    print(text)


if __name__=='__main__':main()

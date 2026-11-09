"""Check public source integrity and aggregate arithmetic, without private data or inference."""
import hashlib
import json
import math
from pathlib import Path


def main():
    root=Path(__file__).resolve().parent
    source=json.loads((root/'PUBLIC_SOURCE_MANIFEST.json').read_text())
    checks=0
    for relative,spec in source.items():
        path=root/relative
        assert path.is_file(),relative
        assert len(path.read_bytes())==spec['bytes'],relative
        assert hashlib.sha256(path.read_bytes()).hexdigest()==spec['sha256'],relative
        checks+=1
    def close(a,b):
        nonlocal checks
        assert math.isclose(a,b,rel_tol=0,abs_tol=1e-12),(a,b)
        checks+=1
    m=json.loads((root/'results/test_metrics.json').read_text())
    d=m['detector_operating_metrics'];tp,fp,fn=d['tp'],d['fp'],d['fn']
    close(d['precision'],tp/(tp+fp));close(d['recall'],tp/(tp+fn));close(d['f1'],2*tp/(2*tp+fp+fn))
    for k in ['detector_AP_primary_input_coordinates','detector_AP_secondary_served_native_coordinates']:
        close(m[k]['AP50_95'],sum(m[k]['AP_by_matching_iou'])/10)
        close(m[k]['AP50'],m[k]['AP_by_matching_iou'][0])
    s=m['segmentation'];tp,fp,fn,tn=(s[k] for k in ['tp','fp','fn','tn'])
    fi=tp/(tp+fp+fn);fd=2*tp/(2*tp+fp+fn);bi=tn/(tn+fp+fn);bd=2*tn/(2*tn+fp+fn)
    for k,v in [('foreground_iou',fi),('foreground_dice',fd),('background_iou',bi),('background_dice',bd),('mean_iou_2_classes',(fi+bi)/2),('mean_dice_2_classes',(fd+bd)/2)]:close(s[k],v)
    r=m['image_review_referral'];tp,fp,fn,tn=(r[k] for k in ['tp','fp','fn','tn'])
    close(r['accuracy'],(tp+tn)/114);close(r['macro_f1'],(2*tp/(2*tp+fp+fn)+2*tn/(2*tn+fp+fn))/2)
    close(r['sensitivity'],tp/(tp+fn));close(r['specificity'],tn/(tn+fp));close(r['balanced_accuracy'],(tp/(tp+fn)+tn/(tn+fp))/2)
    assert m['images']==114 and m['source_families']==19
    assert not m['test_threshold_sweep_performed'] and not m['original_Phase1_test_reopened']
    print('PUBLIC_SUMMARY_VERIFIED:',checks,'source/hash/arithmetic checks passed.')
    print('No private data read and no models loaded. Full private per-image audit is a separate recorded check.')


if __name__=='__main__':
    main()

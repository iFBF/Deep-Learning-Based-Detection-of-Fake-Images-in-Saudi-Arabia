import sys,tempfile,unittest
from pathlib import Path
import numpy as np,pandas as pd
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from phase1_corrections import *

class Corrections(unittest.TestCase):
    def test_identical_pixels_connect_different_caption_families(self):
        frame=pd.DataFrame({'coco_image_id':[1,2,2,3],'sha256':['same','same','next','next']})
        self.assertEqual(len(set(source_components_from_duplicates(frame).values())),1)
    def test_corrupt_and_duplicates_are_counted(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);Image.new('RGB',(8,8),'red').save(root/'a.png');Image.new('RGB',(8,8),'red').save(root/'b.png')
            (root/'bad.png').write_text('not an image')
            f=pd.DataFrame({'relative_path':['a.png','b.png','bad.png'],'label':[0,0,1]})
            clean,report=inspect_images(f,root,root/'out')
            self.assertEqual((report['unreadable_images'],report['duplicate_copies_removed'],len(clean)),(1,1,1))
    def test_publisher_real_and_fake_share_caption_parent(self):
        caps={'123':{'captions':[{'caption_id':9}]}}
        meta=[dict(index=0,image={'path':'123_9.jpg'},reference=False,model='GALIP',category='indoor'),
              dict(index=1,path='123_9.jpg',reference=True,model='COCO',category='indoor')]
        links=publisher_links(meta,caps,'revision')
        frame=pd.DataFrame({'source_model':['GALIP','COCO'],'source_index':[0,1],'label':[1,0]})
        self.assertEqual(attach_links(frame,links).group_id.nunique(),1)
        meta[0]['image']['path']='123_99.jpg'
        with self.assertRaises(ValueError):publisher_links(meta,caps,'revision')
    def test_missing_source_links_fail_closed(self):
        f=pd.DataFrame({'source_model':['GALIP'],'source_index':[999],'label':[1]})
        links=pd.DataFrame({'source_model':['COCO'],'source_index':[1],'coco_image_id':[123],'publisher_label':[0]})
        with self.assertRaises(ValueError):attach_links(f,links)
    def test_confidence_bound_needs_recorded_decisions_for_complete_sample(self):
        fixed=pd.DataFrame({'relative_path':[str(i) for i in range(100)],'sha256':['x']*100,'label':[0]*100,
                            'source_model':['COCO']*100,'source_index':range(100),'group_id':[str(i) for i in range(100)]})
        decisions=fixed.assign(reviewed_label=0,reviewer='source checker',decision_at_utc='2026-11-08',source_evidence='source record',decision_notes='content matched')
        result=audit_report(fixed,decisions);self.assertTrue(result['complete'])
        self.assertAlmostEqual(result['one_sided_95pct_upper_error_rate'],1-.05**.01,places=10)
        decisions.loc[0,'reviewed_label']=np.nan
        self.assertIsNone(audit_report(fixed,decisions)['one_sided_95pct_upper_error_rate'])
        decisions.loc[0,'relative_path']='changed'
        with self.assertRaises(ValueError):audit_report(fixed,decisions)
    def test_new_test_cannot_reuse_an_old_parent_or_pixel(self):
        old=pd.DataFrame({'group_id':['coco_1'],'sha256':['a'],'phash':['0000000000000000']})
        test=pd.DataFrame({'group_id':['coco_1','coco_2'],'sha256':['b','c'],'phash':['ffffffffffffffff']*2,'label':[0,1]})
        with self.assertRaises(ValueError):validate_new_holdout(old,test,old)
        test.group_id=['coco_3','coco_4'];self.assertTrue(validate_new_holdout(old,test,old))
    def test_reported_metrics_use_actual_predictions(self):
        f=pd.DataFrame({'label':[0]*82+[1]*82,'prediction':[0]*80+[1]*2+[0]*2+[1]*80,'group_id':range(164)})
        self.assertAlmostEqual(prediction_metrics(f)['accuracy'],160/164)
        lo,hi=grouped_accuracy_interval(f)
        self.assertLess(lo,160/164);self.assertGreater(hi,160/164)

if __name__=='__main__':unittest.main()

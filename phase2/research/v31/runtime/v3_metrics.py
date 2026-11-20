"""Classification metrics for a Boolean referral policy, without probability claims."""
import numpy as np


def flag_metrics(labels,prediction):
    y=np.asarray(labels,dtype=int);p=np.asarray(prediction,dtype=bool)
    tp=int(((y==1)&p).sum());tn=int(((y==0)&~p).sum());fp=int(((y==0)&p).sum());fn=int(((y==1)&~p).sum())
    safe=lambda a,b:a/b if b else 0.
    return {'n':len(y),'accuracy':float((p==y).mean()),'macro_f1':(safe(2*tp,2*tp+fp+fn)+safe(2*tn,2*tn+fp+fn))/2,
            'balanced_accuracy':(safe(tp,tp+fn)+safe(tn,tn+fp))/2,'fake_recall':safe(tp,tp+fn),'real_recall':safe(tn,tn+fp),
            'confusion_matrix':[[tn,fp],[fn,tp]]}

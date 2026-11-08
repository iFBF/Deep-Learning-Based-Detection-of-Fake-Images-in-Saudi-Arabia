"""Corrected training and single-final-evaluation workflow; run on Kaggle GPU."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, random, gc
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from phase1_corrections import dump, image_path, prediction_metrics, grouped_accuracy_interval

MODEL = 'umm-maybe/AI-image-detector'
REVISION = 'c7e223baf11bc40528af364ba7bdea030ef42f9e'
DEFAULTS = dict(seed=42, image_size=224, batch_size=8, accumulation_steps=1, cnn_epochs=25,
                transfer_epochs=15, cnn_lr=.01, head_lr=.01, backbone_lr=.001, momentum=.9,
                weight_decay=.0001, patience=5, min_delta=.0001, threshold=.5)

def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def seed_all(seed):
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

def build_model(mode):
    import torch
    from torch import nn
    if mode.startswith('scratch'):
        layers=[]; channels=3
        for width in (32,64,128,256):
            layers += [nn.Conv2d(channels,width,3,padding=1,bias=False),nn.BatchNorm2d(width),
                       nn.ReLU(),nn.MaxPool2d(2),nn.Dropout2d(.08)]
            channels=width
        return nn.Sequential(*layers,nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Dropout(.35),nn.Linear(256,2))
    from transformers import AutoModelForImageClassification
    model=AutoModelForImageClassification.from_pretrained(MODEL,revision=REVISION)
    model.classifier=nn.Linear(model.classifier.in_features,2)
    model.config.num_labels=2;model.config.id2label={0:'Real',1:'Fake'};model.config.label2id={'Real':0,'Fake':1}
    for parameter in model.parameters(): parameter.requires_grad=(mode=='full')
    for parameter in model.classifier.parameters(): parameter.requires_grad=True
    return model

def loader(frame, root, mode, config, training=False, corruption=None):
    import torch
    from torchvision import transforms
    from transformers import AutoImageProcessor
    processor=None if mode.startswith('scratch') else AutoImageProcessor.from_pretrained(config.get('processor_dir',MODEL),revision=None if config.get('processor_dir') else REVISION)
    augmentation=transforms.Compose([transforms.RandomHorizontalFlip(.5),transforms.ColorJitter(brightness=.1,contrast=.1)])
    cnn=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([.5]*3,[.5]*3)])
    class Images(torch.utils.data.Dataset):
        def __len__(self): return len(frame)
        def __getitem__(self,index):
            row=frame.iloc[index]
            with Image.open(image_path(root,row.relative_path)) as raw:
                image=ImageOps.exif_transpose(raw).convert('RGB')
                if training: image=augmentation(image)
                if corruption=='blur': image=image.filter(ImageFilter.GaussianBlur(1.5))
                if corruption=='darken': image=ImageEnhance.Brightness(image).enhance(.7)
                if corruption=='brighten': image=ImageEnhance.Brightness(image).enhance(1.3)
                if corruption=='synthetic_haze': image=Image.blend(image,Image.new('RGB',image.size,'white'),.2)
                x=cnn(image) if processor is None else processor(images=image,return_tensors='pt')['pixel_values'][0]
            return x.float(),int(row.label)
    return torch.utils.data.DataLoader(Images(),batch_size=config['batch_size'],shuffle=training,
        num_workers=0,generator=torch.Generator().manual_seed(config['seed']))

def logits(model,x,mode): return model(x) if mode.startswith('scratch') else model(pixel_values=x).logits

def probabilities(model, frame, root, mode, config, device, corruption=None):
    import torch
    scores=[];model.eval()
    with torch.inference_mode():
        for x,y in loader(frame,root,mode,config,corruption=corruption):
            scores.extend(torch.softmax(logits(model,x.to(device),mode),dim=-1)[:,1].cpu().tolist())
    return np.asarray(scores)

def validation_predictions(model,frame,root,mode,config,device):
    import torch
    model.eval();scores=[];loss_sum=0.;count=0
    with torch.inference_mode():
        for x,y in loader(frame,root,mode,config):
            y=y.to(device);output=logits(model,x.to(device),mode)
            loss_sum+=torch.nn.functional.cross_entropy(output,y,reduction='sum').item();count+=len(y)
            scores.extend(torch.softmax(output,dim=-1)[:,1].cpu().tolist())
    return np.asarray(scores),loss_sum/count

def load_checkpoint(mode,directory,device):
    import torch
    directory=Path(directory)
    if mode.startswith('scratch'):model=build_model(mode)
    else:
        from transformers import AutoConfig,AutoModelForImageClassification
        architecture=AutoConfig.from_pretrained(str(directory/'model_config.json'))
        model=AutoModelForImageClassification.from_config(architecture)
    model.load_state_dict(torch.load(directory/'best.pt',map_location='cpu',weights_only=True)['state_dict'])
    return model.to(device).eval()

def train(mode, frame, root, out, config):
    import torch
    from torch import nn
    seed_all(config['seed']); out=Path(out)/mode;out.mkdir(parents=True,exist_ok=True)
    training=frame.loc[frame.split.eq('train')].reset_index(drop=True)
    validation=frame.loc[frame.split.eq('val')].reset_index(drop=True)
    counts=training.label.value_counts().reindex([0,1],fill_value=0)
    if (counts==0).any(): raise ValueError('Training lacks a class')
    weights=torch.tensor(len(training)/(2*counts.to_numpy()),dtype=torch.float32)
    if mode=='scratch_unweighted': weights=torch.ones(2)
    dump(out/'training_class_weights.json', {'counts':counts.tolist(),'weights':weights.tolist(),
         'reduction':'fixed-denominator per-example weighted loss; weights do not cancel at physical batch size one'})
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');weights=weights.to(device)
    model=build_model(mode).to(device)
    if mode.startswith('scratch'): params=[{'params':model.parameters(),'lr':config['cnn_lr']}]
    else:
        params=[{'params':model.classifier.parameters(),'lr':config['head_lr']}]
        backbone=[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('classifier.')]
        if backbone: params.append({'params':backbone,'lr':config['backbone_lr']})
        model.config.to_json_file(str(out/'model_config.json'))
        from transformers import AutoImageProcessor
        AutoImageProcessor.from_pretrained(MODEL,revision=REVISION).save_pretrained(out/'processor')
    optimizer=torch.optim.SGD(params,momentum=config['momentum'],weight_decay=config['weight_decay'])
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=.5,patience=2)
    best=float('inf');bad=0;history=[];selected_row=None;criterion=nn.CrossEntropyLoss(reduction='none')
    try:
        import wandb
        run=wandb.init(project='phase1-real-fake',name='corrected_'+mode,mode='offline',dir=str(out.resolve()),config=config,reinit='finish_previous')
    except ImportError:run=None
    maximum=config['cnn_epochs'] if mode.startswith('scratch') else config['transfer_epochs']
    for epoch in range(1,maximum+1):
        model.train()
        if mode=='frozen': model.eval();model.classifier.train()
        batches=loader(training,root,mode,config,training=True);optimizer.zero_grad(set_to_none=True)
        total=0.; n=0
        for step,(x,y) in enumerate(batches):
            x,y=x.to(device),y.to(device); losses=criterion(logits(model,x,mode),y)*weights[y]
            window_start=(step//config['accumulation_steps'])*config['accumulation_steps']*config['batch_size']
            window_n=min(config['accumulation_steps']*config['batch_size'],len(training)-window_start)
            (losses.sum()/window_n).backward();total+=losses.sum().item();n+=len(y)
            if (step+1)%config['accumulation_steps']==0 or step+1==len(batches):
                nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step();optimizer.zero_grad(set_to_none=True)
        p,loss=validation_predictions(model,validation,root,mode,config,device)
        if not np.isfinite(loss) or not np.isfinite(p).all():raise FloatingPointError('Non-finite validation output')
        metrics=prediction_metrics(validation.assign(prediction=(p>=.5).astype(int)))
        row=dict(epoch=epoch,train_weighted_loss=total/n,val_loss=loss,
                 val_accuracy=metrics['accuracy'],val_macro_f1=metrics['macro_f1'],
                 val_real_recall=metrics['real_recall'],val_fake_recall=metrics['fake_recall'])
        history.append(row);pd.DataFrame(history).to_csv(out/'history.csv',index=False);print(mode,row,flush=True)
        if run:run.log(row,step=epoch)
        scheduler.step(loss)
        improvement=best-loss
        if loss < best:
            best=loss;selected_row=row.copy()
            torch.save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'epoch':epoch,
                        'mode':mode,'config':config,'val_metrics':row},out/'best.pt')
            validation.assign(prob_fake=p,prediction=(p>=.5).astype(int)).to_csv(out/'selected_validation_predictions.csv',index=False)
        bad=0 if improvement>=config['min_delta'] else bad+1
        if bad>=config['patience']: break
    del model,optimizer;gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    if run:
        run.summary.update({'selected_epoch':selected_row['epoch'],'best_val_loss':selected_row['val_loss'],'selected_val_macro_f1':selected_row['val_macro_f1']})
        run.finish()
    return {'mode':mode,'checkpoint':str(out/'best.pt'),'checkpoint_sha256':digest(out/'best.pt'),**selected_row}

def train_and_lock(protocol, image_root, output, overrides=None):
    import platform, importlib.metadata
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if (output/'selection_locked.json').exists(): raise ValueError('Run already locked; use a new directory for a new validation experiment')
    config={**DEFAULTS,**(overrides or {})}
    if config['image_size']!=224: raise ValueError('This protocol fixes input resolution at 224')
    frame=pd.read_csv(protocol,dtype={'phash':str})
    if frame.groupby('group_id').split.nunique().max()>1: raise ValueError('Group leakage')
    dump(output/'config.json',config)
    import shutil,uuid
    shutil.copyfile(protocol,output/'experiment_manifest.csv')
    import re
    names={re.sub(r'[-_.]+','-',d.metadata['Name']).lower() for d in importlib.metadata.distributions() if d.metadata.get('Name')}
    (output/'requirements-lock-executed.txt').write_text('\n'.join(f'{p}=={importlib.metadata.version(p)}' for p in sorted(names))+'\n')
    dump(output/'environment.json',{'python':platform.python_version(),'platform':platform.platform(),'run_id':uuid.uuid4().hex,
             'started_at_utc':datetime.now(timezone.utc).isoformat(),
             'code_hashes':{name:digest(Path(__file__).parent/name) for name in ['phase1_training.py','phase1_corrections.py','dani_provenance.py']}})
    results=[train(m,frame,image_root,output,config) for m in ['scratch_unweighted','scratch','frozen','full']]
    pd.DataFrame(results).to_csv(output/'validation_comparison.csv',index=False)
    selected=min(results,key=lambda row:(row['val_loss'],row['mode']))
    dump(output/'selection_locked.json',{'selected_mode':selected['mode'],'criterion':'minimum unweighted validation cross-entropy',
        'protocol_sha256':digest(protocol),'selected_checkpoint_sha256':selected['checkpoint_sha256'],
        'all_checkpoint_hashes':{r['mode']:r['checkpoint_sha256'] for r in results},
        'locked_at_utc':datetime.now(timezone.utc).isoformat(),'config_sha256':digest(output/'config.json')})
    return results

def final_test(protocol, image_root, output):
    """One frozen evaluation; full and balanced reports derive from the SAME saved predictions."""
    import torch
    output=Path(output);lock=json.loads((output/'selection_locked.json').read_text())
    if digest(protocol)!=lock['protocol_sha256'] or digest(output/'config.json')!=lock['config_sha256']:
        raise ValueError('Protocol/config changed after model lock')
    for mode,expected_hash in lock['all_checkpoint_hashes'].items():
        if digest(output/mode/'best.pt')!=expected_hash:raise ValueError('Locked checkpoint changed')
    if (output/'final_test_metrics.csv').exists():
        if digest(output/'final_test_metrics.csv')!=(output/'final_test_metrics.sha256').read_text().strip():
            raise ValueError('Final metrics changed')
        return pd.read_csv(output/'final_test_metrics.csv')
    frame=pd.read_csv(protocol,dtype={'phash':str});test=frame.loc[frame.split.eq('test')].reset_index(drop=True)
    if test.empty: raise ValueError('A genuinely new, source-disjoint test must be added before final evaluation')
    config=json.loads((output/'config.json').read_text());device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    marker=output/'test_evaluation_started.json'
    if marker.exists() and json.loads(marker.read_text())['selection_sha256']!=digest(output/'selection_locked.json'):
        raise ValueError('Test already opened under a different model selection')
    dump(marker,{'selection_sha256':digest(output/'selection_locked.json'),'started_at_utc':datetime.now(timezone.utc).isoformat()})
    n=int(test.label.value_counts().min());balanced=set()
    for _,part in test.groupby('label'): balanced.update(part.sample(n=n,random_state=config['seed']).relative_path)
    rows=[]
    for mode,expected_hash in lock['all_checkpoint_hashes'].items():
        checkpoint=output/mode/'best.pt'
        if digest(checkpoint)!=expected_hash: raise ValueError('Checkpoint changed after selection lock')
        path=output/mode/'final_test_predictions.csv'
        if path.exists():
            if digest(path)!=(path.with_suffix('.sha256')).read_text().strip():raise ValueError('Cached test predictions changed')
            predictions=pd.read_csv(path)
        else:
            model=load_checkpoint(mode,output/mode,device)
            runtime_config={**config,**({} if mode.startswith('scratch') else {'processor_dir':str(output/mode/'processor')})}
            p=probabilities(model,test,image_root,mode,runtime_config,device)
            predictions=test.assign(prob_fake=p,prediction=(p>=.5).astype(int));predictions.to_csv(path,index=False)
            path.with_suffix('.sha256').write_text(digest(path)+'\n')
            del model;gc.collect()
            if torch.cuda.is_available():torch.cuda.empty_cache()
        if predictions.relative_path.tolist()!=test.relative_path.tolist():raise ValueError('Cached test identities differ')
        from sklearn.metrics import precision_recall_fscore_support
        precision,recall,f1,support=precision_recall_fscore_support(predictions.label,predictions.prediction,labels=[0,1],zero_division=0)
        pd.DataFrame({'class':['Real','Fake'],'precision':precision,'recall':recall,'f1':f1,'support':support}).to_csv(output/mode/'per_class_test_metrics.csv',index=False)
        for population,part in [('full_unbalanced',predictions),('balanced_secondary',predictions.loc[predictions.relative_path.isin(balanced)])]:
            metrics=prediction_metrics(part);ci=grouped_accuracy_interval(part,seed=config['seed'])
            confidence=np.maximum(part.prob_fake,1-part.prob_fake);correct=part.label.eq(part.prediction).to_numpy()
            indices=np.minimum((confidence.to_numpy()*10).astype(int),9);ece=0.
            for bin_id in np.unique(indices):
                mask=indices==bin_id;ece+=mask.mean()*abs(correct[mask].mean()-confidence.to_numpy()[mask].mean())
            rows.append({'mode':mode,'population':population,**{k:v for k,v in metrics.items() if k!='confusion_matrix'},
                         'accuracy_ci95_low':ci[0],'accuracy_ci95_high':ci[1],'ece':float(ece),
                         'selected_on_validation':mode==lock['selected_mode']})
    result=pd.DataFrame(rows);result.to_csv(output/'final_test_metrics.csv',index=False)
    (output/'final_test_metrics.sha256').write_text(digest(output/'final_test_metrics.csv')+'\n');return result

def validation_robustness(protocol,image_root,output):
    """Defined interventions test sensitivity; they do not retrospectively prove a cause of old errors."""
    import torch
    output=Path(output);config=json.loads((output/'config.json').read_text())
    validation=pd.read_csv(protocol);validation=validation.loc[validation.split.eq('val')].reset_index(drop=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');rows=[]
    for mode in ['scratch','frozen','full']:
        model=load_checkpoint(mode,output/mode,device)
        runtime_config={**config,**({} if mode.startswith('scratch') else {'processor_dir':str(output/mode/'processor')})}
        base=None
        for condition in [None,'blur','darken','brighten','synthetic_haze']:
            p=probabilities(model,validation,image_root,mode,runtime_config,device,corruption=condition)
            metrics=prediction_metrics(validation.assign(prediction=(p>=.5).astype(int)))
            if condition is None:base=metrics['accuracy']
            rows.append({'mode':mode,'condition':condition or 'original','accuracy':metrics['accuracy'],
                         'accuracy_change':metrics['accuracy']-base,'macro_f1':metrics['macro_f1'],'split':'validation'})
        del model
    pd.DataFrame(rows).to_csv(output/'validation_condition_interventions.csv',index=False)
    return pd.DataFrame(rows)

"""Read original DANI records, not lossy viewer thumbnails, at a fixed revision."""
from pathlib import Path
import io, hashlib, json, urllib.request
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from phase1_corrections import fingerprint, image_path, dump, validate_new_holdout,source_components_from_duplicates

REVISION='870e29fcdc13c405fae35442899e9ba1da11691d'
SHARDS={0:(0,45022),9:(405195,45021)}
SHARD0_SHA='4eec87b76e59fbedf673ec2901a7e15c2244343ddc226acaefd3e095adb75539'

def original_records(indices,cache):
    import pyarrow.parquet as pq
    import fsspec
    cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
    wanted=set(map(int,indices));result={}
    for shard,(start,count) in SHARDS.items():
        selected=sorted(i for i in wanted if start<=i<start+count)
        if not selected:continue
        filename=f'train-{shard:05d}-of-00012.parquet'
        url=f'https://huggingface.co/datasets/Renyang/DANI/resolve/{REVISION}/data/{filename}'
        local=cache/filename
        if shard==0 and not local.exists():
            temporary=local.with_suffix('.partial')
            with urllib.request.urlopen(url+'?download=true',timeout=60) as response,temporary.open('wb') as out:
                while chunk:=response.read(4*1024*1024):out.write(chunk)
            if hashlib.sha256(temporary.read_bytes()).hexdigest()!=SHARD0_SHA:
                temporary.unlink();raise ValueError('Publisher shard hash mismatch')
            temporary.replace(local)
        if local.exists():
            if shard==0:
                with local.open('rb') as check:
                    if hashlib.file_digest(check,'sha256').hexdigest()!=SHARD0_SHA:raise ValueError('Cached publisher shard was changed')
            stream=local.open('rb')
        else:stream=fsspec.open(url,'rb',block_size=65536,cache_type='none',client_kwargs={'trust_env':True}).open()
        try:
            parquet=pq.ParquetFile(stream)
            if parquet.metadata.num_rows!=count:raise ValueError('Pinned shard row count changed')
            boundaries=np.cumsum([0]+[parquet.metadata.row_group(i).num_rows for i in range(parquet.num_row_groups)])
            groups={int(np.searchsorted(boundaries,i-start,side='right')-1) for i in selected}
            for position,group in enumerate(sorted(groups)):
                for row in parquet.read_row_group(group).to_pylist():
                    index=int(row['index'])
                    if index in wanted:result[index]=row
                if position%10==0 or position+1==len(groups):
                    print('Publisher shard',shard,':',position+1,'/',len(groups),'requested row groups',flush=True)
        finally:stream.close()
    if set(result)!=wanted:raise ValueError('Some source indices are outside the verified publisher shards')
    return result

def source_image(record):
    return ImageOps.exif_transpose(Image.open(io.BytesIO(record['image']['bytes']))).convert('RGB')

def source_audit(fixed,root,cache,output):
    """Independent reference-field decisions after content matching; never copy the local label."""
    from datetime import datetime,timezone
    from phase1_corrections import audit_report
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    fixed=pd.read_csv(fixed);records=original_records(fixed.source_index,cache);decisions=fixed.copy();evidence=[]
    for field in ['reviewed_label','reviewer','decision_at_utc','source_evidence','decision_notes']:decisions[field]=''
    for i,row in fixed.iterrows():
        record=records[int(row.source_index)]
        item={'relative_path':row.relative_path,'source_index':int(row.source_index),'status':'pending'}
        if record['model']!=row.source_model:raise ValueError('Publisher model mismatch')
        if fingerprint(image_path(root,row.relative_path))[0]!=row.sha256:
            raise ValueError('Audit image changed after the sample was frozen')
        with Image.open(image_path(root,row.relative_path)) as image:
            local=ImageOps.exif_transpose(image).convert('RGB');a=np.asarray(local,dtype=float)
        publisher=source_image(record)
        # A single declared conversion, not a per-image search for the best fit.
        publisher=publisher.resize(local.size,Image.Resampling.LANCZOS) if publisher.size!=local.size else publisher
        b=np.asarray(publisher,dtype=float);mse=float(((a-b)**2).mean());difference=float(abs(a-b).mean())
        psnr=999. if mse==0 else float(10*np.log10(255**2/mse))
        item.update(psnr_db=psnr,mean_rgb_difference=difference,publisher_filename=record['image']['path'])
        if psnr>=50 and difference<=.5:
            if not isinstance(record['reference'],bool):raise ValueError('Non-boolean source reference')
            decision=0 if record['reference'] else 1
            decisions.at[i,'reviewed_label']=decision
            decisions.at[i,'reviewer']='Automated independent DANI publisher-record check'
            decisions.at[i,'decision_at_utc']=datetime.now(timezone.utc).isoformat()
            decisions.at[i,'source_evidence']=f'https://huggingface.co/datasets/Renyang/DANI/tree/{REVISION}/data'
            decisions.at[i,'decision_notes']=f"Reference={record['reference']}; original parquet {record['image']['path']}; PSNR={psnr:.5f}; RGB difference={difference:.5f}. No human review claimed."
            item.update(status='matched',publisher_label=decision)
        evidence.append(item)
    decisions.to_csv(output/'label_audit_source_decisions.csv',index=False)
    dump(output/'source_audit_evidence.json',{'revision':REVISION,'method':'Original publisher bytes, declared Lanczos resize if needed, PSNR>=50 dB and mean RGB difference<=0.5','rows':evidence})
    report=audit_report(fixed,decisions);dump(output/'label_audit_report.json',report);return report

def acquire_holdout(reserved,development,old_population,root,cache,output):
    reserved=pd.read_csv(reserved);records=original_records(reserved.source_index,cache)
    rows=[]
    for row in reserved.to_dict('records'):
        record=records[int(row['source_index'])]
        if record['image']['path']!=row['publisher_filename'] or record['model']!=row['source_model']:
            raise ValueError('Reserved source identity changed')
        if (0 if record['reference'] else 1)!=row['label']:
            raise ValueError('Reserved label differs from publisher reference')
        image=source_image(record).resize((224,224),Image.Resampling.LANCZOS)
        path=image_path(root,row['relative_path']);path.parent.mkdir(parents=True,exist_ok=True);image.save(path)
        sha,phash,width,height=fingerprint(path)
        rows.append({**row,'sha256':sha,'phash':phash,'width':width,'height':height})
    holdout=pd.DataFrame(rows)
    development=pd.read_csv(development,dtype={'phash':str});old_population=pd.read_csv(old_population,dtype={'phash':str})
    return finalize_holdout(holdout,development,old_population,output)

def finalize_holdout(holdout,development,old_population,output):
    """Predeclared data-quality rules run before any model predictions are available."""
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    holdout.to_csv(output/'acquired_holdout_manifest.csv',index=False)
    old_pixels=set(old_population.sha256);old_phashes=[int(str(x),16) for x in old_population.phash]
    exclude=set();reasons=[]
    for row in holdout.itertuples():
        reason=None
        if row.sha256 in old_pixels:reason='identical historical pixels'
        elif any((int(str(row.phash),16)^old).bit_count()<=4 for old in old_phashes):reason='historical near-duplicate candidate (pHash <=4)'
        if reason:
            exclude.add(int(row.coco_image_id));reasons.append({'coco_image_id':int(row.coco_image_id),'relative_path':row.relative_path,'reason':reason})
    reserved_count=len(holdout);reserved_parents=holdout.coco_image_id.nunique()
    if set(holdout.loc[holdout.label.eq(1),'source_model'])!={'DFGAN'}:raise ValueError('The preregistered held-out generator must be DFGAN')
    if set(development.loc[development.label.eq(1),'source_model'])!={'GALIP'}:raise ValueError('Development generator changed from the declared protocol')
    holdout=holdout.loc[~holdout.coco_image_id.isin(exclude)].copy()
    if holdout.empty:raise ValueError('No source-disjoint holdout survives data-quality screening')
    components=source_components_from_duplicates(holdout)
    holdout['group_id']=holdout.coco_image_id.map(components)
    duplicate_count=int(holdout.sha256.duplicated().sum())
    holdout=holdout.drop_duplicates('sha256').reset_index(drop=True)
    pd.DataFrame(reasons,columns=['coco_image_id','relative_path','reason']).to_csv(output/'holdout_exclusions.csv',index=False)
    validate_new_holdout(development,holdout,old_population)
    protocol=pd.concat([development,holdout],ignore_index=True);protocol.to_csv(output/'protocol_manifest.csv',index=False)
    if protocol.groupby('group_id').split.nunique().max()>1:raise ValueError('Source component crosses protocol splits')
    dump(output/'holdout_verification.json',{'reserved_parent_groups':int(reserved_parents),'reserved_images':reserved_count,
        'excluded_historical_overlap_parent_groups':len(exclude),'within_holdout_duplicate_copies_removed':duplicate_count,
        'groups':holdout.group_id.nunique(),'retained_parent_ids':holdout.coco_image_id.nunique(),'images':len(holdout),
        'class_counts':{str(k):int(v) for k,v in holdout.label.value_counts().items()},'historical_group_overlap':0,
        'historical_pixel_overlap':0,'historical_near_duplicate_candidates':0,'new_generator':'DFGAN',
        'protocol_sha256':hashlib.sha256((output/'protocol_manifest.csv').read_bytes()).hexdigest(),
        'status':'Acquired, overlap checks passed; not evaluated by a model'})
    return protocol

"""Read-only input recovery and explicit development-only image manifests."""
from pathlib import Path
from io import BytesIO
from contextlib import contextmanager
import hashlib
import json
import shutil
import zipfile

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy.fft import dctn

KIT = Path(__file__).resolve().parent
PREFIX = 'CV-PROJECT/SELF_COLLECTED_DATA/Real_Fake_Images_Dataset/'
SELF_LABELS_SHA256 = '28f680582792088a1b3154a737b088afb34ce09cdf3695773e0501a13b05e49e'


class DirectoryArchive:
    """Read the same logical members from a fully extracted Kaggle input."""
    def __init__(self, root, strip_prefix=''):
        self.root=Path(root).resolve();self.strip_prefix=strip_prefix
    def path(self, name):
        if self.strip_prefix:
            if not name.startswith(self.strip_prefix):raise ValueError('Unexpected project member prefix.')
            name=name[len(self.strip_prefix):]
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe extracted member path.')
        target=(self.root/relative).resolve()
        if not target.is_relative_to(self.root):raise ValueError('Extracted member escapes its input root.')
        if not target.is_file():raise FileNotFoundError('Missing extracted dataset member: '+str(target))
        return target
    def open(self, name):return self.path(name).open('rb')
    def read(self, name):return self.path(name).read_bytes()
    def __enter__(self):return self
    def __exit__(self, *args):return False


def input_reader(source, project=False):
    return DirectoryArchive(source,'CV-PROJECT/' if project else '') if Path(source).is_dir() else zipfile.ZipFile(source)


def extracted_image_root(project, development):
    data_root=(Path(project)/'DATA').resolve()
    if not data_root.is_dir():raise FileNotFoundError('Missing extracted DATA folder: '+str(data_root))
    relative=[Path(name) for name in development.relative_path]
    if any(p.is_absolute() or '..' in p.parts for p in relative):raise ValueError('Unsafe recorded image path.')
    # Kaggle can produce DATA/data/... or DATA/data/data/... when unpacking data.zip.
    candidates=[data_root,data_root/'data']
    first=relative[0]
    for match in sorted(data_root.rglob(first.name)):
        if match.is_file() and tuple(match.parts[-len(first.parts):])==first.parts:
            candidates.append(match.parents[len(first.parts)-1])
    for root in dict.fromkeys(candidates):
        root=root.resolve()
        if root.is_relative_to(data_root) and all((root/name).is_file() for name in relative):return root
    raise FileNotFoundError('The extracted DATA folder does not contain every fixed training/validation image. No image hashes or split requirements were relaxed. Root: '+str(data_root))


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def verify_kit():
    manifest=KIT/'PAYLOAD_SHA256.json'
    if manifest.is_file():
        for name,digest in json.loads(manifest.read_text())['files'].items():
            if sha(KIT/name)!=digest:raise ValueError('Enhancement payload changed: '+name)


def dump(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.writing')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)


def fingerprint(path):
    with Image.open(path) as raw:
        image=ImageOps.exif_transpose(raw).convert('RGB');image.load()
        rgb=hashlib.sha256(str(image.size).encode()+image.tobytes()).hexdigest()
        block=dctn(np.asarray(image.convert('L').resize((32,32),Image.Resampling.LANCZOS),dtype=float),type=2,norm='ortho')[:8,:8].flatten()
        phash=sum(int(x)<<i for i,x in enumerate(block>np.median(block[1:])))
        return rgb,f'{phash:016x}',image.width,image.height


def safe_copy_member(archive, name, destination):
    if isinstance(archive,DirectoryArchive):member=name
    else:
        matches=[x for x in archive.infolist() if x.filename==name]
        if len(matches)!=1:raise ValueError('Expected one archive member: '+name)
        member=matches[0]
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_name(destination.name+'.writing')
    with archive.open(member) as source,temporary.open('wb') as target:shutil.copyfileobj(source,target,8*1024*1024)
    temporary.replace(destination)
    return destination


def project_archive(input_root, cache):
    identity=json.loads((KIT/'evidence/input_identity.json').read_text())
    cached=Path(cache)/'CV-PROJECT (1).zip'
    if cached.is_file() and cached.stat().st_size==identity['cv_project_archive_bytes'] and sha(cached)==identity['cv_project_archive_sha256']:return cached
    candidates=sorted(Path(input_root).rglob('CV-PROJECT*.zip'))
    for path in candidates:
        if path.stat().st_size==identity['cv_project_archive_bytes'] and sha(path)==identity['cv_project_archive_sha256']:return path
    for path in sorted(Path(input_root).rglob('Real-Abshar_Project_Archive*.zip')):
        with zipfile.ZipFile(path) as archive:
            names=[n for n in archive.namelist() if n.endswith('CV-PROJECT (1).zip')]
            if len(names)==1:
                target=safe_copy_member(archive,names[0],cache/'CV-PROJECT (1).zip')
                if sha(target)==identity['cv_project_archive_sha256']:return target
    # A fully extracted upload has no ZIP files left. Authenticate its labels now;
    # every required original RGB hash and every self-image byte hash follow in prepare.
    suffix=tuple(Path('SELF_COLLECTED_DATA/Real_Fake_Images_Dataset/labels.csv').parts)
    invalid=[]
    for labels in sorted(Path(input_root).rglob('labels.csv')):
        if tuple(labels.parts[-len(suffix):])!=suffix:continue
        root=labels.parents[2]
        if not (root/'DATA').is_dir():continue
        if sha(labels)!=SELF_LABELS_SHA256:
            invalid.append(str(labels));continue
        print('Found fully extracted project:',root,flush=True)
        return root
    if invalid:raise ValueError('Extracted self-collected labels differ from the pinned original: '+', '.join(invalid))
    raise FileNotFoundError('Attach the private project archive or its fully extracted CV-PROJECT folder containing DATA and SELF_COLLECTED_DATA. No matching ZIP or authenticated extracted labels were found under '+str(input_root))


def baseline_model(input_root, cache, spec):
    cached=Path(cache)/'original_classifier.onnx'
    if cached.is_file() and cached.stat().st_size==spec['bytes'] and sha(cached)==spec['sha256']:return cached
    for path in sorted(Path(input_root).rglob('classifier.onnx')):
        if path.stat().st_size==spec['bytes'] and sha(path)==spec['sha256']:return path
    for path in sorted(Path(input_root).rglob('*Service_Release.zip')):
        with zipfile.ZipFile(path) as archive:
            names=[n for n in archive.namelist() if n==spec['file'] or n.endswith('/'+spec['file'])]
            if len(names)==1:
                target=safe_copy_member(archive,names[0],cache/'original_classifier.onnx')
                if target.stat().st_size==spec['bytes'] and sha(target)==spec['sha256']:return target
    # The optional locally completed master archive can contain the private ZIP.
    for path in sorted(Path(input_root).rglob('Real-Abshar_Project_Archive*.zip')):
        with zipfile.ZipFile(path) as archive:
            names=[n for n in archive.namelist() if n.endswith('Real-Abshar_Phase2_M8_Service_Release.zip')]
            if len(names)==1:
                nested=safe_copy_member(archive,names[0],cache/'Real-Abshar_Phase2_M8_Service_Release.zip')
                with zipfile.ZipFile(nested) as release:
                    matches=[n for n in release.namelist() if n==spec['file'] or n.endswith('/'+spec['file'])]
                    if len(matches)==1:
                        target=safe_copy_member(release,matches[0],cache/'original_classifier.onnx')
                        if target.stat().st_size==spec['bytes'] and sha(target)==spec['sha256']:return target
    raise FileNotFoundError('Attach the private Real-Abshar_Phase2_M8_Service_Release.zip, or its extracted models/classifier.onnx. The baseline graph must match the frozen SHA-256.')


def class_group_weights(frame, views=1):
    counts=frame.groupby(['label','group_id']).size()
    groups=frame.groupby('label').group_id.nunique()
    if set(groups.index)!={0,1}:raise ValueError('Training requires both declared classes.')
    return np.asarray([len(frame)/(2*groups.loc[r.label]*counts.loc[(r.label,r.group_id)]*views) for r in frame.itertuples()],dtype=np.float64)


def prepare(input_root, output, require_baseline=True):
    verify_kit()
    output=Path(output);cache=output/'cache';cache.mkdir(parents=True,exist_ok=True)
    development=pd.read_csv(KIT/'evidence/development_manifest.csv',dtype={'phash':str})
    if not set(development.split)<={'train','val'} or development.groupby('group_id').split.nunique().max()!=1:
        raise ValueError('The development input contains a closed-test row or a group crossing splits.')
    if development.split.value_counts().to_dict()!={'train':1165,'val':233}:raise ValueError('Unexpected fixed development cohort.')
    legacy=pd.read_csv(KIT/'evidence/legacy_exclusion_ledger.csv',dtype={'phash':str})
    legacy_rgb=set(legacy.sha256);legacy_ph=[int(x,16) for x in legacy.phash.dropna()]
    source=project_archive(input_root,cache)
    image_root=cache/'images';rows=[]
    with input_reader(source,project=True) as archive:
        if source.is_dir():
            data_zip=source/'DATA/data.zip'
            if not data_zip.is_file():data_zip=extracted_image_root(source,development)
        else:
            data_zip=cache/'original_data.zip'
            if not data_zip.is_file():safe_copy_member(archive,'CV-PROJECT/DATA/data.zip',data_zip)
        with input_reader(data_zip) as images:
            for record in development.to_dict('records'):
                relative=Path(record['relative_path'])
                if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe recorded image path.')
                target=image_root/relative
                if not target.is_file():safe_copy_member(images,relative.as_posix(),target)
                rgb,ph,w,h=fingerprint(target)
                if rgb!=record['sha256']:raise ValueError('Development image RGB hash differs: '+str(relative))
                rows.append({'id':relative.as_posix(),'path':str(target.resolve()),'label':int(record['label']),
                             'group_id':record['group_id'],'role':'train' if record['split']=='train' else 'validation_original',
                             'source':record['source_model'],'rgb_sha256':rgb,'phash':ph,'width':w,'height':h})
        self_labels=archive.read(PREFIX+'labels.csv')
        if hashlib.sha256(self_labels).hexdigest()!=SELF_LABELS_SHA256:raise ValueError('Self-collected labels differ from the pinned original.')
        self_frame=pd.read_csv(BytesIO(self_labels))
        if len(self_frame)!=72 or self_frame.label.value_counts().to_dict()!={0:36,1:36}:raise ValueError('Unexpected self-collected cohort.')
        overlap=[];self_rows=[]
        for record in self_frame.to_dict('records'):
            relative=Path(record['relative_path'])
            if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe self-collected image path.')
            target=image_root/'self72'/relative
            if not target.is_file():safe_copy_member(archive,PREFIX+relative.as_posix(),target)
            if sha(target)!=record['sha256']:raise ValueError('Self-collected file hash differs: '+str(relative))
            rgb,ph,w,h=fingerprint(target)
            distance=min((int(ph,16)^old).bit_count() for old in legacy_ph)
            overlap.append({'id':relative.as_posix(),'group_id':record['group_id'],'legacy_exact_match':rgb in legacy_rgb,'nearest_legacy_phash_distance':distance})
            self_rows.append({'id':'self72/'+relative.as_posix(),'path':str(target.resolve()),'label':int(record['label']),
                              'group_id':'self72_'+str(record['group_id']),'role':'validation_self72','source':'user_real' if record['label']==0 else 'ChatGPT_user_dataset_card',
                              'rgb_sha256':rgb,'phash':ph,'width':w,'height':h})
        pd.DataFrame(overlap).to_csv(output/'self72_overlap_audit.csv',index=False)
        if any(r['legacy_exact_match'] or r['nearest_legacy_phash_distance']<=4 for r in overlap):
            raise ValueError('Self-collected overlap candidates require review before using this validation cohort.')
        if len({r['rgb_sha256'] for r in self_rows})!=len(self_rows):raise ValueError('Duplicate decoded self-collected images require review.')
        # Link near-duplicate scene groups conservatively.
        parent={r['group_id']:r['group_id'] for r in self_rows}
        def find(g):
            while parent[g]!=g:g=parent[g]
            return g
        for i,a in enumerate(self_rows):
            for b in self_rows[i+1:]:
                if (int(a['phash'],16)^int(b['phash'],16)).bit_count()<=4:
                    parent[find(b['group_id'])]=find(a['group_id'])
        for row in self_rows:row['group_id']=find(row['group_id'])
        rows.extend(self_rows)
    frame=pd.DataFrame(rows);frame.to_csv(output/'development_images.csv',index=False)
    manifest=json.loads((KIT/'evidence/frozen_service_manifest.json').read_text())
    model=baseline_model(input_root,cache,manifest['models']['classifier']) if require_baseline else None
    summary={'status':'passed','roles':frame.role.value_counts().to_dict(),'labels':{role:part.label.value_counts().to_dict() for role,part in frame.groupby('role')},
             'groups':frame.groupby('role').group_id.nunique().to_dict(),'source_counts':frame.groupby(['role','source']).size().rename('images').reset_index().to_dict('records'),
             'self72_prior_use':'Unconfirmed; entire cohort is development validation, never a fresh final test.',
             'self72_provenance':'User dataset card: real uploads and ChatGPT-generated counterparts. Labels/source are provided provenance, not independent authenticity verification.',
             'closed_test_images_loaded':0,'legacy_overlap_candidates':0,'baseline_graph_sha256':sha(model) if model else None,
             'input_layout':'fully_extracted_project' if source.is_dir() else 'original_zip',
             'input_source':str(source.resolve()),
             'original_archive_sha256':sha(source) if source.is_file() else None,
             'archive_hash_verified':source.is_file(),
             'content_checks':'Every selected original decoded-RGB hash, pinned self-labels SHA-256, and all self-image byte hashes are verified in either layout.',
             'development_manifest_sha256':sha(KIT/'evidence/development_manifest.csv'),
             'self_labels_file_sha256':hashlib.sha256(self_labels).hexdigest(),
             'limits':'Exact RGB and pHash checks cannot establish the absence of every hidden source relation. Scene groups are conservative subject/series groups, not verified event IDs.'}
    dump(output/'preflight.json',summary)
    print(json.dumps(summary,indent=2),flush=True)
    return frame,model

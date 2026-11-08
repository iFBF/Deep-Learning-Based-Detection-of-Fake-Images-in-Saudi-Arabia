"""Evidence-based Phase 1 repair utilities. No predicted labels are used as audit truth."""
from pathlib import Path
import hashlib
import json
import re
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy.fft import dctn
from scipy.ndimage import laplace
from scipy.stats import beta, fisher_exact
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold

IMAGE_SIZE = 224

def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False))

def image_path(root, relative):
    root = Path(root).resolve(); path = (root / str(relative)).resolve()
    if not path.is_relative_to(root): raise ValueError('Image path escapes the dataset root')
    return path

def fingerprint(path):
    with Image.open(path) as raw:
        image = ImageOps.exif_transpose(raw).convert('RGB')
        image.load()
        digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
        gray = np.asarray(image.convert('L').resize((32, 32), Image.Resampling.LANCZOS), dtype=float)
        block = dctn(gray, type=2, norm='ortho')[:8, :8].flatten()
        phash = sum(int(x) << i for i, x in enumerate(block > np.median(block[1:])))
        return digest, f'{phash:016x}', image.width, image.height

def inspect_images(manifest, root, output):
    """Count unreadable files explicitly and keep a reproducible exclusion record."""
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(manifest) if not isinstance(manifest, pd.DataFrame) else manifest.copy()
    if frame.relative_path.duplicated().any(): raise ValueError('Duplicate manifest paths')
    if not frame.label.isin([0, 1]).all(): raise ValueError('Labels must be binary')
    good, bad = [], []
    for row in frame.to_dict('records'):
        try:
            sha, phash, width, height = fingerprint(image_path(root, row['relative_path']))
            good.append({**row, 'sha256': sha, 'phash': phash, 'width': width, 'height': height})
        except (OSError, ValueError) as error:
            bad.append({'relative_path': row['relative_path'], 'reason': str(error)})
    readable = pd.DataFrame(good)
    conflicts = readable.groupby('sha256').label.nunique()
    if (conflicts > 1).any(): raise ValueError('Conflicting labels for identical decoded images')
    duplicates = readable.loc[readable.sha256.duplicated(keep='first')].copy()
    clean = readable.drop_duplicates('sha256').reset_index(drop=True)
    pd.DataFrame(bad, columns=['relative_path', 'reason']).to_csv(output/'invalid_images.csv', index=False)
    duplicates.to_csv(output/'exact_duplicates_removed.csv', index=False)
    readable.to_csv(output/'all_readable_manifest.csv', index=False)
    clean.to_csv(output/'clean_manifest.csv', index=False)
    counts = {'input_images': len(frame), 'unreadable_images': len(bad), 'readable_images': len(readable),
              'duplicate_copies_removed': len(duplicates), 'retained_images': len(clean),
              'conflicting_exact_labels': 0, 'input_resolution': [IMAGE_SIZE, IMAGE_SIZE]}
    dump(output/'cleaning_report.json', counts)
    return clean, counts

def publisher_links(metadata, captions, revision):
    """Decode publisher filenames, validating every caption against its COCO parent."""
    links = []
    for row in metadata:
        filename = str(row.get('image', {}).get('path', row.get('path', '')))
        name = Path(filename).name
        real = row['reference']
        if not isinstance(real, bool): raise ValueError('Publisher reference is not boolean')
        pair = re.fullmatch(r'(\d+)_(\d+)\.(?:jpg|jpeg|png)', name, re.I)
        if pair:
            image_id, caption_id = map(int, pair.groups())
        elif real:
            match = re.fullmatch(r'(?:COCO_(?:val|train)201[47]_)?(\d+)\.(?:jpg|jpeg|png)', name, re.I)
            if not match: continue
            image_id, caption_id = int(match[1]), None
        else:
            continue
        reference = captions.get(str(image_id))
        if reference is None: continue
        if caption_id is not None and caption_id not in {int(c['caption_id']) for c in reference['captions']}:
            raise ValueError(f'Caption {caption_id} does not belong to COCO {image_id}')
        links.append({'source_model': row['model'], 'source_index': int(row['index']),
                      'coco_image_id': image_id, 'caption_id': caption_id,
                      'publisher_label': 0 if real else 1, 'publisher_scene': row['category'],
                      'publisher_filename': filename, 'publisher_revision': revision,
                      'mapping_evidence': 'DANI parquet image.path + publisher COCO-caption dictionary'})
    return pd.DataFrame(links)

def attach_links(frame, links):
    if links.duplicated(['source_model', 'source_index']).any(): raise ValueError('Ambiguous source map')
    joined = frame.drop(columns=['group_id', 'coco_image_id', 'caption_id'], errors='ignore').merge(
        links, on=['source_model', 'source_index'], how='left', validate='many_to_one')
    if joined.coco_image_id.isna().any():
        missing = joined.loc[joined.coco_image_id.isna(), ['source_model', 'source_index']]
        raise ValueError(f'{len(missing)} images lack verified COCO-caption links; do not split by image index')
    if not joined.label.eq(joined.publisher_label).all(): raise ValueError('Publisher-label disagreement')
    joined.coco_image_id = joined.coco_image_id.astype(int)
    joined['group_id'] = 'coco_' + joined.coco_image_id.astype(str)
    if 'canonical_group_id' in joined:
        joined['group_id']=joined.canonical_group_id.fillna(joined.group_id)
    return joined

def source_components_from_duplicates(frame):
    """Connect COCO families when identical decoded pixels link different parents."""
    parent={int(x):int(x) for x in frame.coco_image_id}
    def find(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    for _,part in frame.groupby('sha256'):
        ids=list(map(int,part.coco_image_id.unique()))
        for x in ids[1:]:
            a,b=find(ids[0]),find(x);parent[max(a,b)]=min(a,b)
    return {x:'coco_'+str(find(x)) for x in parent}

def development_split(frame, seed=42):
    """Historical images may be train/validation, never relabelled as an unseen test."""
    frame = frame.copy().reset_index(drop=True)
    strata = frame.scene.astype(str) + '_' + frame.label.astype(str)
    train, val = next(StratifiedGroupKFold(6, shuffle=True, random_state=seed).split(frame, strata, frame.group_id))
    frame['split'] = 'train'; frame.loc[val, 'split'] = 'val'
    if frame.groupby('group_id').split.nunique().max() != 1: raise ValueError('Group leakage')
    for part in ['train', 'val']:
        if set(frame.loc[frame.split.eq(part), 'label']) != {0, 1}: raise ValueError('Split lacks a class')
    return frame

def validate_new_holdout(development, holdout, old_population, near_distance=4):
    """Reject both old pixels and caption-linked relatives of ANY historical image."""
    if holdout.empty or set(holdout.label) != {0, 1}: raise ValueError('Holdout needs both classes')
    if set(holdout.group_id) & set(old_population.group_id): raise ValueError('Historical COCO-group overlap')
    if set(holdout.sha256) & set(old_population.sha256): raise ValueError('Historical pixel overlap')
    if set(holdout.group_id) & set(development.group_id): raise ValueError('Development group overlap')
    old_hashes = [int(str(x), 16) for x in old_population.phash]
    for phash in holdout.phash:
        if any((int(str(phash), 16) ^ old).bit_count() <= near_distance for old in old_hashes):
            raise ValueError('Historical near-duplicate candidate; review before finalizing holdout')
    return True

def make_audit_sample(frame, path, n=100, seed=42):
    expected = frame.sample(n=min(n, len(frame)), random_state=seed).copy()
    fields = ['relative_path', 'sha256', 'label', 'source_model', 'source_index', 'group_id']
    expected = expected[fields].reset_index(drop=True)
    path = Path(path)
    if path.exists():
        existing = pd.read_csv(path)
        if existing[fields].astype(str).to_dict('records') != expected.astype(str).to_dict('records'):
            raise ValueError('Audit sample is frozen; changed sample requires a new named audit')
        return existing
    for field in ['reviewed_label', 'reviewer', 'decision_at_utc', 'source_evidence', 'decision_notes']:
        expected[field] = ''
    path.parent.mkdir(parents=True, exist_ok=True); expected.to_csv(path, index=False)
    return expected

def audit_report(fixed, decisions):
    keys = ['relative_path', 'sha256', 'label', 'source_model', 'source_index', 'group_id']
    if fixed[keys].astype(str).to_dict('records') != decisions[keys].astype(str).to_dict('records'):
        raise ValueError('Audit sample or labels changed')
    reviewed = pd.to_numeric(decisions.reviewed_label, errors='coerce')
    valid = reviewed.isin([0, 1])
    for field in ['reviewer', 'decision_at_utc', 'source_evidence', 'decision_notes']:
        valid &= decisions[field].fillna('').astype(str).str.strip().ne('')
    n = int(valid.sum()); errors = int((reviewed[valid] != decisions.loc[valid, 'label']).sum())
    complete = bool(len(fixed) >= 100 and valid.all())
    upper = float(beta.ppf(.95, errors+1, n-errors)) if complete and errors < n else (1.0 if complete else None)
    return {'sample_size': len(fixed), 'decisions_recorded': n, 'errors': errors, 'complete': complete,
            'one_sided_95pct_upper_error_rate': upper,
            'method': 'Clopper-Pearson; completed fixed random sample; source-label correspondence, not real-world authenticity'}

def measured_conditions(path):
    with Image.open(path) as raw:
        gray = np.asarray(ImageOps.exif_transpose(raw).convert('L').resize((224, 224)), dtype=float)/255.
    return {'brightness': float(gray.mean()), 'contrast': float(gray.std()),
            'sharpness': float(laplace(gray, mode='reflect').var())}

def grouped_accuracy_interval(frame, seed=42, iterations=2000):
    agg = frame.assign(correct=frame.label.eq(frame.prediction)).groupby('group_id').correct.agg(['sum','count'])
    if len(agg) < 2: raise ValueError('At least two groups required for interval')
    rng = np.random.default_rng(seed); a = agg['sum'].to_numpy(); b = agg['count'].to_numpy()
    values = []
    for _ in range(iterations):
        indices = rng.integers(0, len(a), len(a)); values.append(a[indices].sum()/b[indices].sum())
    return [float(x) for x in np.quantile(values, [.025,.975])]

def prediction_metrics(frame):
    precision, recall, f1, support = precision_recall_fscore_support(frame.label, frame.prediction, labels=[0,1], zero_division=0)
    return {'n': len(frame), 'accuracy': float(accuracy_score(frame.label, frame.prediction)),
            'macro_f1': float(f1.mean()), 'real_recall': float(recall[0]), 'fake_recall': float(recall[1]),
            'predicted_fake_fraction': float(frame.prediction.eq(1).mean()),
            'confusion_matrix': confusion_matrix(frame.label, frame.prediction, labels=[0,1]).tolist()}

def analyse_historical(analysis_root, image_root, output):
    analysis_root, output = Path(analysis_root), Path(output); output.mkdir(parents=True, exist_ok=True)
    experiment = pd.read_csv(analysis_root/'experiment_manifest.csv', dtype={'phash': str})
    features = pd.DataFrame([{'relative_path': p, **measured_conditions(image_path(image_root,p))} for p in experiment.relative_path])
    training = experiment.loc[experiment.split.eq('train')].merge(features, on='relative_path')
    limits = {'dark': float(training.brightness.quantile(.1)), 'bright': float(training.brightness.quantile(.9)),
              'low_contrast': float(training.contrast.quantile(.1)), 'low_sharpness': float(training.sharpness.quantile(.1))}
    features['dark'] = features.brightness.lt(limits['dark'])
    features['bright'] = features.brightness.gt(limits['bright'])
    features['low_contrast'] = features.contrast.lt(limits['low_contrast'])
    features['low_sharpness'] = features.sharpness.lt(limits['low_sharpness'])
    dump(output/'condition_thresholds.json', {'fit_split':'historical training only', 'resolution':[224,224], 'thresholds':limits,
         'interpretation':'Image statistics are proxies; low contrast is not a confirmed haze annotation.'})
    slices, tests, summaries = [], [], {}
    for mode in ['scratch','frozen','full']:
        predictions_path=analysis_root/mode/'final_test_predictions.csv'
        if not predictions_path.exists():predictions_path=analysis_root/mode/'test_predictions.csv'
        predictions = pd.read_csv(predictions_path).merge(features, on='relative_path', validate='one_to_one')
        predictions['error'] = predictions.label.ne(predictions.prediction)
        summaries[mode] = prediction_metrics(predictions)
        predictions.to_csv(output/(mode+'_predictions_with_conditions.csv'), index=False)
        errors = predictions.loc[predictions.error].copy()
        errors['haze_visual_review'] = 'unreviewed'
        errors['cause_status'] = 'descriptive association only; controlled intervention pending checkpoint rerun'
        errors.to_csv(output/(mode+'_error_review.csv'), index=False)
        for dimension in ['scene','source_model','label','dark','bright','low_contrast','low_sharpness']:
            for value, part in predictions.groupby(dimension):
                slices.append({'mode':mode,'dimension':dimension,'value':str(value),'n':len(part),'errors':int(part.error.sum()),
                               'error_rate':float(part.error.mean()),'accuracy':float((~part.error).mean())})
        for condition in ['dark','bright','low_contrast','low_sharpness']:
            for label in ['all',0,1]:
                subset = predictions if label == 'all' else predictions.loc[predictions.label.eq(label)]
                yes, no = subset.loc[subset[condition]], subset.loc[~subset[condition]]
                table = [[int(yes.error.sum()), int((~yes.error).sum())], [int(no.error.sum()),int((~no.error).sum())]]
                odds, p = fisher_exact(table)
                tests.append({'mode':mode,'condition':condition,'label':str(label),'condition_n':len(yes),
                              'condition_errors':int(yes.error.sum()),'other_n':len(no),'other_errors':int(no.error.sum()),
                              'odds_ratio': float(odds) if np.isfinite(odds) else None,'p_two_sided':float(p)})
    # Holm adjustment across the whole predeclared collection, including class-conditioned checks.
    tests = pd.DataFrame(tests); order = np.argsort(tests.p_two_sided.to_numpy())
    adjusted = np.maximum.accumulate(np.minimum(1, tests.p_two_sided.to_numpy()[order]*(len(tests)-np.arange(len(tests)))))
    tests['p_holm'] = 1.; tests.loc[tests.index[order], 'p_holm'] = adjusted
    tests.to_csv(output/'condition_association_tests.csv', index=False)
    slices = pd.DataFrame(slices); slices.to_csv(output/'error_slices.csv', index=False)
    fake = summaries['scratch']['predicted_fake_fraction']
    train_counts = training.label.value_counts().sort_index()
    summaries['scratch_diagnosis'] = {'train_real':int(train_counts[0]),'train_fake':int(train_counts[1]),
        'training_fake_fraction':float(train_counts[1]/train_counts.sum()), 'predicted_fake_fraction':fake,
        'always_fake_actual_test_accuracy':float(predictions.label.eq(1).mean()),
        'always_fake_actual_test_macro_f1':prediction_metrics(predictions.assign(prediction=1))['macro_f1'],
        'always_fake_balanced_test_accuracy':.5,'always_fake_balanced_test_macro_f1':1/3,
        'interpretation':'Observed class bias and weak validation fit. Imbalance is plausible, not proven as the sole cause. Compare unweighted/weighted scratch with identical groups and initialization; select on validation.'}
    dump(output/'historical_analysis.json', summaries)
    return summaries, slices, tests

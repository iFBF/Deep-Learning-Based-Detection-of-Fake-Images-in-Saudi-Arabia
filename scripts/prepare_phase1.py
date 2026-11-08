"""Portable data/audit/holdout preparation. No account login is required."""
import argparse,sys
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from phase1_corrections import inspect_images,attach_links,development_split,make_audit_sample
from dani_provenance import source_audit,acquire_holdout

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image-root',required=True,help='Parent of data/manifest.csv, writable to allow a fresh_holdout/ directory')
    p.add_argument('--output',default='outputs/phase1_prepared')
    p.add_argument('--cache',default='data/cache')
    p.add_argument('--acquire-holdout',action='store_true')
    p.add_argument('--verify-audit',action='store_true')
    a=p.parse_args();root=Path(a.image_root).resolve();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
    evidence=Path(__file__).resolve().parents[1]/'docs/phase1_corrections/evidence'
    clean,counts=inspect_images(root/'data/manifest.csv',root,out/'cleaning')
    links=pd.read_csv(evidence/'dani_source_links.csv')
    original=attach_links(pd.read_csv(out/'cleaning/all_readable_manifest.csv',dtype={'phash':str}),links)
    original.to_csv(out/'all_historical_linked_manifest.csv',index=False)
    import hashlib

    # Load the fixed split shipped with the corrections dataset.
    split_file = evidence / "corrected_development_manifest.csv"
    assert hashlib.sha256(split_file.read_bytes()).hexdigest() == "15f17c6e5d7f6c5df0ce4ac0a6f09023b797e6710e3dc64870bfb3ecbf0ad138", "Fixed split file changed"
    frozen = pd.read_csv(split_file, dtype={"sha256": str, "phash": str})

    # Verify the fixed records against the images checked in this run.
    assert frozen.relative_path.is_unique and frozen.sha256.is_unique
    assert set(frozen.sha256) == set(clean.sha256), "Retained image population changed"
    observed = original.set_index("relative_path").reindex(frozen.relative_path)
    for field in ["sha256", "label", "group_id"]:
        assert observed[field].tolist() == frozen[field].tolist(), f"Image identity mismatch: {field}"

    # Save this exact split for training and holdout preparation.
    development = frozen.copy()
    assert set(development.split) == {"train", "val"}
    assert development.groupby("group_id").split.nunique().max() == 1
    assert development.group_id.nunique() == 498
    development.to_csv(out / "development_manifest.csv", index=False)
    make_audit_sample(attach_links(clean,links),out/'label_audit_fixed.csv')
    print('Cleaning:',counts)
    print('Caption groups:',development.groupby('split').group_id.nunique().to_dict())
    if a.verify_audit:print('Source audit:',source_audit(out/'label_audit_fixed.csv',root,a.cache,out/'audit'))
    if a.acquire_holdout:
        protocol=acquire_holdout(evidence/'reserved_holdout_sources.csv',out/'development_manifest.csv',out/'all_historical_linked_manifest.csv',root,a.cache,out)
        print('Protocol populations:',protocol.groupby(['split','label']).size().to_dict())

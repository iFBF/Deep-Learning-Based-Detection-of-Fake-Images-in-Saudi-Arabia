"""Restore only the three hash-checked ONNX graphs from the private frozen release ZIP."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--archive',required=True)
    p.add_argument('--release',default=str(Path(__file__).resolve().parent/'service'))
    args=p.parse_args()
    release=Path(args.release).resolve()
    manifest=json.loads((release/'manifest.json').read_text())
    lock=json.loads((release/'release_lock.json').read_text())
    if sha(release/'manifest.json')!=lock['manifest_sha256']:
        raise SystemExit('Frozen manifest hash mismatch.')
    for relative,digest in lock['source_and_config_sha256'].items():
        target=(release/relative).resolve()
        if not target.is_relative_to(release) or sha(target)!=digest:
            raise SystemExit('Frozen source/configuration mismatch: '+relative)
    with zipfile.ZipFile(Path(args.archive)) as z:
        for name,spec in manifest['models'].items():
            target=release/spec['file']
            if target.is_file():
                if sha(target)!=spec['sha256'] or target.stat().st_size!=spec['bytes']:
                    raise SystemExit('Existing model differs; preserve it before restoring: '+name)
                print('Already verified:',name)
                continue
            matches=[n for n in z.namelist() if n==spec['file'] or n.endswith('/'+spec['file'])]
            if len(matches)!=1:
                raise SystemExit('Expected exactly one private release model entry: '+name)
            if z.getinfo(matches[0]).file_size!=spec['bytes']:
                raise SystemExit('Private model size mismatch: '+name)
            target.parent.mkdir(parents=True,exist_ok=True)
            temporary=target.with_suffix('.onnx.restoring')
            try:
                with z.open(matches[0]) as source, temporary.open('wb') as output:
                    shutil.copyfileobj(source,output)
                if sha(temporary)!=spec['sha256']:
                    raise SystemExit('Private model hash mismatch: '+name)
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
            print('Restored and verified:',name)
    print('The three frozen graphs are ready. Start service/serve.py using RUN_LOCAL.md.')


if __name__=='__main__':
    main()

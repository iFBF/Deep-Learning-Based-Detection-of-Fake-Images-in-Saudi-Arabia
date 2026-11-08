"""Export real trained weights and all reproduction evidence, excluding image data."""
import argparse,sys
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from phase1_training import digest
import json

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();root=Path(a.run).resolve();output=Path(a.output).resolve()
    if output.is_relative_to(root):raise ValueError('Put the ZIP outside the run directory')
    lock=json.loads((root/'selection_locked.json').read_text())
    for mode,sha in lock['all_checkpoint_hashes'].items():
        if digest(root/mode/'best.pt')!=sha:raise ValueError('Checkpoint identity mismatch')
    required=['config.json','experiment_manifest.csv','environment.json','final_test_metrics.csv','requirements-lock-executed.txt']
    for name in required:
        if not (root/name).is_file():raise ValueError('Missing '+name)
    with ZipFile(output,'w',ZIP_DEFLATED) as z:
        for path in sorted(root.rglob('*')):
            if path.is_file() and not path.name.endswith(('.png','.jpg','.jpeg')):z.write(path,path.relative_to(root).as_posix())
    print('Actual trained-model bundle:',output,output.stat().st_size,'bytes')

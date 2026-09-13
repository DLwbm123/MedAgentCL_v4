import argparse
from pathlib import Path
import traceback
from .gate import run
from med_prism.transport.diagnostics import write_json


def main():
    p=argparse.ArgumentParser(description='Boundary-only MS-CRC finite-forward gate; no training')
    p.add_argument('--state',required=True);p.add_argument('--data-root',required=True)
    p.add_argument('--output-root',required=True);p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    if Path(args.output_root).exists():p.error('Fresh output directory required')
    try:run(args)
    except Exception as exc:
        root=Path(args.output_root)
        if root.exists():write_json(root/'failure.json',{'status':'FAIL','error':str(exc),'traceback':traceback.format_exc()})
        raise


if __name__=='__main__':main()

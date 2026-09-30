"""Recompute following from a saved run without rerunning the detector."""
import argparse
import json
from pathlib import Path
from src.shells.following import receiver_following


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): parser.error('Choose a new output file')
    report=json.loads((args.run/'result.json').read_text())
    config=json.loads((args.run/'config.json').read_text())
    frames=[json.loads(line) for line in (args.run/'tracks.jsonl').read_text().splitlines()]
    fps=report['snap_frame']/report['snap_time_seconds']
    result=receiver_following(frames,report['snap_frame'],fps,config['geometry_at_pre'],
                              report['corner_motion']['selected_corner_ids'])
    args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()

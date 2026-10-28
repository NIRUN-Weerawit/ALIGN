"""Print and save a quick summary of existing evaluation results; no GPU needed."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from eval.evaluation_summary import write_evaluation_summary,render_evaluation_summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result',type=Path,help='Evaluation result.json')
    parser.add_argument('--compare',type=Path,help='Optional previous result.json; different episode sets are identified')
    parser.add_argument('--output-dir',type=Path,help='Default: beside the evaluation result')
    parser.add_argument('--full',action='store_true',help='Also print per-task results and individual handover mistakes')
    args=parser.parse_args()
    try:
        summary,paths=write_evaluation_summary(args.result,args.output_dir,args.compare)
    except (OSError,ValueError,KeyError,TypeError) as error:
        parser.error(str(error))
    print(render_evaluation_summary(summary,detailed=args.full))
    print('Saved: '+', '.join(str(path) for path in paths))


if __name__=='__main__':main()

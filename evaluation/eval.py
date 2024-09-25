import argparse
import json
import os
from evaluators.evaluator_LR import Evaluator_LR
from evaluators.evaluator_RR import Evaluator_RR
from evaluators.evaluator_VM import Evaluator_VM
from evaluators.evaluator_DC import Evaluator_DC
from evaluators.evaluator_NU import Evaluator_NU
from evaluators.evaluator_base import Evaluator

from utils.nav_graph_loader import load_nav_graphs
import pprint
pp = pprint.PrettyPrinter(indent=4)

parser = argparse.ArgumentParser()
parser.add_argument('--annotation_root', type=str, required=True,
                        help='meta info')
parser.add_argument('--submission_root', type=str, required=True,
                        help='submission file')
parser.add_argument('--out_root', type=str,default='outs',required=False,
                        help='output path')
parser.add_argument('--scans_dir', type=str, required=False,
                        help='directory to scans of matterport3d')
args = parser.parse_args()

scans = os.listdir(args.scans_dir)
graphs, paths, distances = load_nav_graphs(scans)

def main():
    evaluators = []
    evaluators.append(Evaluator(graphs, paths, distances, annt_file=os.path.join(args.annotation_root, 'R2R_val_unseen.json'), submission_file=os.path.join(args.submission_root, 'submit_val_unseen.json')))
    evaluators.append(Evaluator_LR(graphs, paths, distances, annt_file=os.path.join(args.annotation_root, 'R2R_LR.json'), submission_file=os.path.join(args.submission_root, 'submit_LR.json')))
    evaluators.append(Evaluator_RR(graphs, paths, distances, annt_file=os.path.join(args.annotation_root, 'R2R_RR.json'), submission_file=os.path.join(args.submission_root, 'submit_RR.json')))
    evaluators.append(Evaluator_VM(graphs, paths, distances, annt_file=os.path.join(args.annotation_root, 'R2R_VM.json'), submission_file=os.path.join(args.submission_root, 'submit_VM.json')))
    evaluators.append(Evaluator_DC(graphs, paths, distances, annt_file=os.path.join(args.annotation_root, 'R2R_DC.json'), submission_file=os.path.join(args.submission_root, 'submit_DC.json')))
    evaluators.append(Evaluator_NU(graphs, paths, distances, annt_file=os.path.join(args.annotation_root, 'R2R_NU.json'), submission_file=os.path.join(args.submission_root, 'submit_NU.json')))
    
    # Display results
    results = {}
    for e in evaluators:
        res = e.eval()
        print(f'\033[1;32m [INFO]\033[0m {e.id}, {len(e.preds)} data')
        pp.pprint(res)
        results[e.id] = res
        print()

    os.makedirs(args.out_root, exist_ok=True)
    json.dump(results, open(os.path.join(args.out_root, "results.json"), 'w'), indent=4)

if __name__ == '__main__':
    main()
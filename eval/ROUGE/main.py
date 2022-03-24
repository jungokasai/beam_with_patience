import argparse, json
from rouge_score import rouge_scorer

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument('--src', type=str, metavar='N',
                    help='source file')
parser.add_argument('--hyp', type=str, metavar='N',
                    help='hypothesis file')
parser.add_argument('--refs', type=str, metavar='N',
                    help='reference file')
parser.add_argument('--outfile', type=str, metavar='N',
                    help='output file')
args = parser.parse_args()

def read_jsonl(infile, extract_key=None):
    f = open(infile, 'r')
    if extract_key is None:
        out = [json.loads(line.strip()) for line in f]
    else:
        out = [json.loads(line.strip())[extract_key] for line in f]
    f.close()
    return out

def read_txt(infile, extract_key=None):
    f = open(infile, 'r')
    out = [line.strip() for line in f]
    f.close()
    return out

def score(src, hyp, refs, outfile):
    src = read_txt(src)
    hyp = read_txt(hyp)
    refs = read_jsonl(refs, 'refs')
    scores = []
    #rouge_types = ['rougeL']
    #rouge_types = ['rouge2']
    rouge_types = ['rouge3']
    scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=False)
    for hyp_s, refs_s in zip(hyp, refs):
        scores_s = []
        scores_s = [scorer.score(ref_s, hyp_s)[rouge_types[0]].fmeasure for ref_s in refs_s]
        scores.append(max(scores_s))
    with open(outfile, 'wt') as fout:
        for score in scores:
            fout.write(str(score))
            fout.write('\n')


if __name__ == '__main__':
    score(args.src, args.hyp, args.refs, args.outfile)

import argparse, os
from fairseq.models.bart import BARTModel



def translate(model_dir,
              in_file,
              out_file,
              batch_size,
              model_name,
              num_shards,
              shard_id,
              lenpen,
              beam,
              patience_factor,
              ):
                                                
    model = BARTModel.from_pretrained('/home/acd13578qu/data/xsum/bart.large.xsum/', checkpoint_file='model.pt', data_name_or_path='/home/acd13578qu/data/xsum/bart.large.xsum/')
    XSUM_KWARGS = dict(beam=beam, lenpen=lenpen, max_len_b=60, min_len=10, no_repeat_ngram_size=3, patience_factor=patience_factor)
    start_id, end_id = get_line_ids(in_file, num_shards, shard_id)
    print(start_id, end_id)
    src_sents = []
    model.cuda()
    with open(in_file) as fin:
        for i, line in enumerate(fin):
            if start_id <= i < end_id:
                line = line.strip()
                src_sents.append(line)
    nb_sents = len(src_sents)
    nb_batches = (nb_sents+batch_size-1)//batch_size
    outputs = []
    for i in range(nb_batches):
        print('Batch ID: {}/{}'.format(i, nb_batches))
        output = model.sample(src_sents[i*batch_size:(i+1)*batch_size], **XSUM_KWARGS)
        outputs.extend(output)
    with open(out_file, 'wt') as fout:
        for output in outputs:
            fout.write(output)
            fout.write('\n')

def get_line_ids(in_file, num_shards, shard_id):
    nb_lines = sum(1 for i in open(in_file, 'rb'))
    shard_size = nb_lines//num_shards
    remainder = nb_lines - shard_size*num_shards
    start_id = shard_size*shard_id + min([shard_id, remainder])
    end_id = shard_size*(shard_id+1) + min([shard_id+1, remainder])
    return start_id, end_id

if __name__ == '__main__':
    parser = argparse.ArgumentParser(allow_abbrev=False)

    parser.add_argument('--batch-size', default=20, type=int, metavar='N',
                        help='batch size')
    parser.add_argument('--model-dir', type=str, metavar='N', 
                        default='/home/acd13578qu/data/xsum/bart.large.xsum/', help='model directory')
    parser.add_argument('--model-name', type=str, metavar='N', 
                        default='model.pt', help='model name')
    parser.add_argument('--in-file', type=str, metavar='N',
                        default='summ/xsum/src/xsum_src.txt', help='source file')
    parser.add_argument('--out-file', type=str, metavar='N',
                        default='cnndm_bart.txt', help='target output file')
    parser.add_argument('--num-shards', default=1, type=int, metavar='N',
                        help='number of shards')
    parser.add_argument('--shard-id', default=0, type=int, metavar='N',
                        help='shard id')
    parser.add_argument('--lenpen', default=1.0, type=float, metavar='N',
                        help='length penalty')
    parser.add_argument('--beam', default=6, type=int, metavar='N',
                        help='beam size')
    #CNN_KWARGS = dict(beam=beam, lenpen=lenpen, max_len_b=140, min_len=55, no_repeat_ngram_size=3, patience_factor=patience_factor)
    parser.add_argument('--patience-factor', default=1.0, type=float, metavar='N',
                        help='patience factor')
    args = parser.parse_args()
    translate(args.model_dir,
              args.in_file,
              args.out_file,
              args.batch_size,
              args.model_name,
              args.num_shards,
              args.shard_id,
              args.lenpen,
              args.beam,
              args.patience_factor,
              )

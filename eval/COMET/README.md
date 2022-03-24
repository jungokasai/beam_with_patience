```
python main.py --src  ../../bidimensional-leaderboards/tasks/wmt2020-zh-en/source/wmt2020-zh-en_src.jsonl --hyp ../../bidimensional-leaderboards/tasks/wmt2020-zh-en/generators/wmt2020-zh-en_DeepMind.381.jsonl --refs ../../bidimensional-leaderboards/tasks/wmt2020-zh-en/references/wmt2020-zh-en_refs-AB.jsonl  --outfile scores.txt
docker build -t wmt2020-zh-en_wmt20-comet-da .
docker image save wmt2020-zh-en_wmt20-comet-qe-da | gzip > wmt2020-zh-en_wmt20-comet-da.tar.gz
```

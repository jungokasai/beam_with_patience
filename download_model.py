   
import subprocess

def download_model(url, model_name):
    command = 'gdown {}'.format(url)
    subprocess.run(command, shell=True)
    command = 'tar xvzf {}'.format(model_name)
    subprocess.run(command, shell=True)

def run_model(model_dir, in_file, out_file, bpe=False):
    command = 'python translate.py --in-file {} --out-file {} --model-dir {}'.format(in_file, out_file, model_dir)
    if bpe:
        command += ' --moses'
    else:   
        command += ' --spiece'
    subprocess.run(command, shell=True)

if __name__ == '__main__':
    ja_en_url = 'https://drive.google.com/uc?id=1M8zKXtcrzMWNIytFREgd6l9lSOfgmMbQ' 
    ja_en_name = 'trans_jaen_base.tar.gz'
    download_model(ja_en_url, ja_en_name)
    ja_en_model_dir = 'trans_jaen_base' 
    in_file = 'examples/example_ja.txt'
    out_file = 'examples/example_ja-en.en.txt'
    run_model(ja_en_model_dir, in_file, out_file, bpe=False)

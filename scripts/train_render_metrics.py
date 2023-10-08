import os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--ip', type=str, default="127.0.0.1")
parser.add_argument("--source_path", "-s", required=True, type=str)
parser.add_argument("--output_path", default="./eval")
parser.add_argument("--data_device", default="cuda")
parser.add_argument('--device', type=str, default="cuda:0")
args, _ = parser.parse_known_args()

os.system("python train.py --data_device "+ args.data_device +" -s " + args.source_path + " -m " + args.output_path + " --ip " + args.ip + " --device " + args.device)
 

os.system("python render.py -m " + args.output_path + " --device " + args.device)
 
os.system("python metrics_on_train.py -m " + args.output_path + " --device " + args.device)
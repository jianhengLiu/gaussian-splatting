import os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--ip', type=str, default="127.0.0.1")
parser.add_argument("--source_path", "-s", required=True, type=str)
parser.add_argument("--model_paths", "-m",required=True, type=str)
parser.add_argument("--data_device", default="cuda")
parser.add_argument('--device', type=str, default="cuda:0")
parser.add_argument('--densify_from_iter', type=int, default=500)
parser.add_argument('--iterations', type=int, default=30000)
parser.add_argument('--opacity_reset_interval', type=int, default=3_000)



args, _ = parser.parse_known_args()

exit_code = os.system("python train.py --data_device "+ args.data_device +" -s " + args.source_path + " -m " + args.model_paths + " --ip " + args.ip + " --device " + args.device + " --densify_from_iter " + str(args.densify_from_iter) + " --iterations " + str(args.iterations) + " --opacity_reset_interval " + str(args.opacity_reset_interval) + " --eval")
 
if exit_code != 0:
    exit(exit_code)

exit_code = os.system("python render.py -m " + args.model_paths + " --device " + args.device)
 
if exit_code != 0:
    exit(exit_code)
    
# os.system("python metrics_on_train.py -m " + args.model_paths + " --device " + args.device)
exit_code = os.system("python metrics.py -m " + args.model_paths + " --device " + args.device)
if exit_code != 0:
    exit(exit_code)
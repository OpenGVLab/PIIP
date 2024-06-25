import torch
import argparse

parser = argparse.ArgumentParser(description='Hyperparams')
parser.add_argument('filename', nargs='?', type=str, default=None)

args = parser.parse_args()

model = torch.load(args.filename, map_location=torch.device('cpu'))

state_dict = model['module']
# new_state_dict = {}
# for k, v in state_dict.items():
#     if "decode_head" in k:
#         new_state_dict[k] = v
# print(new_state_dict.keys())
# new_dict = {'module': new_state_dict}
print(state_dict.keys())


new_dict = {'module': state_dict}
torch.save(new_dict, args.filename.replace(".pt", "_release.pt"))
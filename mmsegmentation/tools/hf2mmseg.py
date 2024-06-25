import os

import torch
import argparse

parser = argparse.ArgumentParser(description='Hyperparams')
parser.add_argument('input', nargs='?', type=str, default=None)
parser.add_argument('output', nargs='?', type=str, default=None)

args = parser.parse_args()
basename = os.path.basename(args.input)

model = torch.load(args.input, map_location=torch.device('cpu'))
state_dict = model['module']
new_state_dict = {}
for k, v in state_dict.items():
    if "vision_model" in k:
        k = k.replace('embeddings.class_embedding', "cls_token")
        k = k.replace('embeddings.position_embedding', "pos_embed", )
        k = k.replace('embeddings.patch_embedding.weight', "patch_embed.proj.weight")
        k = k.replace('embeddings.patch_embedding.bias', "patch_embed.proj.bias")
        k = k.replace('ls1', "ls1.gamma")
        k = k.replace('ls2', "ls2.gamma")
        k = k.replace('encoder.layers.', "blocks.")
        k = k.replace('vision_model.', "")
        new_state_dict[k] = v
print(new_state_dict.keys())
new_dict = {'module': new_state_dict}

out_path = os.path.join(args.output, basename)
torch.save(new_dict, out_path.replace(".pt", ".pth"))
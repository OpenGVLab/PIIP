import torch
import open_clip
# from modeling_clip import OpenCLIPVisionTextDualEncoderModel
from transformers import AutoModel

pixel_values = torch.randn(1, 3, 320, 320)

openclip_model, _, _ = open_clip.create_model_and_transforms('convnext_base_w_320', pretrained='laion_aesthetic_s13b_b82k_augreg')

model = AutoModel.from_pretrained("/mnt/petrelfs/liqingyun/yx/llava_piip/pretrained/convnext_base_w_320.laion_aesthetic_s13b_b82k_augreg_dual")

v1 = model.get_image_features(pixel_values)
v2 = openclip_model.encode_image(pixel_values)
import ipdb; ipdb.set_trace()
print(torch.allclose(v1, v2, atol=1e-4))
# False
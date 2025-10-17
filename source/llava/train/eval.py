
import os
from tqdm import tqdm

import time

import torch
# import random
import numpy as np

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image

import json

from safetensors.torch import save_file, load_file  

from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path



import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image



def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set() 
    # multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler', 's_update', 's_projector', 'gate_proj', 'up_proj', 'down_proj',  'post_attention_layernorm', 'o_proj', 'input_layernorm', ]
    # multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler', 'layers.0.', 'layers.1.', 'layers.2.', 'layers.3.', 'layers.4.', 'layers.5.', 'layers.6.', 'layers.7.', 'layers.8.', 'layers.9.', 'layers.10.', 'layers.11.', 'layers.12.', 'layers.13.', 'layers.14.', 'layers.15.', 'layers.16.', 'layers.17.']
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    preserve_keywords = ['q_proj', 'v_proj']
    for name, module in model.named_modules():
        # print(name)
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if not any(key in name for key in preserve_keywords):
            continue

        if isinstance(module, cls):
            lora_module_names.add(name)
            # names = name.split('.')
            # lora_module_names.add(names[0] if len(names) == 1 else names[-1])
        
    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def find_sample(result, img):
    for item in result:
        if img == item['img']:
            return True
    return False


def visualize_feature_map(feature_map, num_patches=32, patch_size=8, save_path=None):
    fig, axs = plt.subplots(1, num_patches, figsize=(20, 20))
    for i, ax in enumerate(axs.flat):
        if i < feature_map.shape[1]:
            # 将每个 patch 的激活值调整为图像的 patch 大小
            patch = feature_map[0, i].reshape(patch_size, patch_size).cpu().numpy()
            ax.imshow(patch, cmap='viridis')
        ax.axis("off")
    # 保存图片
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)  # 关闭图像以释放内存


def eval(model, tokenizer, image_processor, rank_id, image_root, data_dir, output_dir):

    data_json = data_dir + "test%d.json" % rank_id
    output_file = output_dir + "ans%d.json" % rank_id

    device = "cuda"
    conv_mode = 'vicuna_v1'
    temperature = 0
    top_p = None
    num_beams = 1
    max_new_tokens = 1024

    bf16 = False
    fp16 = True
    computype = torch.float16
    
    model.eval()



    with open(data_json, 'r') as f:
        data = json.load(f)

    # if os.path.exists(output_file):
    #     with open(output_file, 'r') as f:
    #         results = json.load(f)
    # else:
    results = []

    cnt = 0
    for line in tqdm(data):

        if "image" in line:
            image_file = line["image"]
        else:
            image_file = -1

        # if find_sample(results, image_file):
        #     continue

        questions = line["conversations"]

        output_conv = {
            "img": image_file,
            "qa": [],
        }
        for qid in range(0, len(questions), 2):
            assert questions[qid]["form"] == "question"
            assert questions[qid+1]["form"] == "answer"

            question = questions[qid]['value']
            answer = questions[qid+1]['value']

            if "<image>\n" in question:
                question = question.split('<image>\n')[-1]

            cur_prompt = question
            if model.config.mm_use_im_start_end:
                cur_prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + cur_prompt
            else:
                cur_prompt = DEFAULT_IMAGE_TOKEN + '\n' + cur_prompt

            conv = conv_templates[conv_mode].copy()
            conv.append_message(conv.roles[0], cur_prompt)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            # print(prompt)

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)

            if image_file == -1:
                image_tensor = torch.zeros(3, 336, 336)
                img_size = (336,336)
            else:
                image = Image.open(os.path.join(image_root, image_file)).convert('RGB')
                image_tensor = process_images([image], image_processor, model.config)[0]
                img_size = image.size
                

  
            with torch.inference_mode():
                output_ids = model.generate(
                    inputs=input_ids.cuda(),
                    images=image_tensor.unsqueeze(0).cuda().to(computype),
                    image_sizes=[img_size],
                    num_beams=num_beams,
                    max_new_tokens=max_new_tokens,
                    # no_repeat_ngram_size=3,
                    # repetition_penalty=1.2,
                    use_cache=True)
                

                
            outputs = tokenizer.decode(output_ids[0]).strip().split('<s>')[1].split('</s>')[0]

            output_conv["qa"].append({
                "question": question,
                "answer": answer,
                "prediction": outputs,
            })
        
      
        # quit()
        
        results.append(output_conv)


        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)



import os
import gc
import ast
import json
import bisect
import random
import argparse
import tempfile
from typing import Tuple, List

import cv2
import torch
import numpy as np
from PIL import Image
from torchvision.ops import box_convert

# Grounding DINO (install from https://github.com/IDEA-Research/GroundingDINO)
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# Segment Anything 2 (install from https://github.com/facebookresearch/sam2)
from sam2.build_sam import build_sam2_video_predictor

# diffusers
from diffusers import StableDiffusion3InpaintPipeline as SD3Pipeline
from transformers import AutoImageProcessor, AutoModel

from read_h5 import ReadH5Files
import config
def pytorch_gc():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()

class DinoExtractor():
    def __init__(self, ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(config.DINOV2_PATH)
        self.dino_model = AutoModel.from_pretrained(config.DINOV2_PATH).to(self.device)

    def set_encoder_images(self, imgs_array):
        image_features = []
        for i in range(len(imgs_array)):
            inputs_feature = self.processor(images=Image.fromarray(imgs_array[i][:,:,::-1]), return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.dino_model(**inputs_feature)
                image_feature = outputs.last_hidden_state
                image_feature = image_feature.mean(dim=1)
                image_feature /= image_feature.norm(dim=-1, keepdim=True)
                image_features.append(image_feature[0].cpu().numpy())
        return np.asarray(image_features)
    
    def set_encoder_image(self, image):
        image_features = []
        # image2 = Image.open(img_path)
        # image2 = cv2.imread(img_path)
        # image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        inputs_feature = self.processor(image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.dino_model(**inputs_feature)
            image_feature = outputs.last_hidden_state
            image_feature = image_feature.mean(dim=1)
            image_feature /= image_feature.norm(dim=-1, keepdim=True)
            image_features.append(image_feature[0].cpu().numpy())
        return np.asarray(image_features)


@torch.no_grad()
def dinov2_matching(src_images, tar_images, labels, tar_bbox):
    # Match each annotated source crop to a detected target crop via DINOv2 features
    pytorch_gc()
    dino_extractor = DinoExtractor()
    tar_images = np.asarray(tar_images)
    tar_features = dino_extractor.set_encoder_images(tar_images)
    label_to_bbox = []
    used_indices = set()
    for idx in range(len(labels)):
        src_image = src_images[idx]
        src_features = dino_extractor.set_encoder_image(src_image)
        similarities = src_features @ tar_features.T
        for used_idx in used_indices:
            similarities[0, used_idx] = -float('inf')
        best_match_idx = similarities.argmax().item()
        used_indices.add(best_match_idx)
        top_box = tar_bbox[best_match_idx]
        label_to_bbox.append(top_box)
    pytorch_gc()
    return label_to_bbox

def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(
        clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model

def load_np_image(image: np.array) -> Tuple[np.array, torch.Tensor]:
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_source = Image.fromarray(image).convert("RGB")
    image_transformed, _ = transform(image_source, None)
    return image, image_transformed

def preprocess_caption(caption: str) -> str:
    result = caption.lower().strip()
    if result.endswith("."):
        return result
    return result + "."

def predict(
        model,
        image: torch.Tensor,
        caption: str,
        box_threshold: float,
        text_threshold: float,
        device: str = "cuda",
        remove_combined: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    caption = preprocess_caption(caption=caption)

    model = model.to(device)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image[None], captions=[caption])

    prediction_logits = outputs["pred_logits"].cpu().sigmoid()[0]  # prediction_logits.shape = (nq, 256)
    prediction_boxes = outputs["pred_boxes"].cpu()[0]  # prediction_boxes.shape = (nq, 4)

    mask = prediction_logits.max(dim=1)[0] > box_threshold
    logits = prediction_logits[mask]  # logits.shape = (n, 256)
    boxes = prediction_boxes[mask]  # boxes.shape = (n, 4)

    tokenizer = model.tokenizer
    tokenized = tokenizer(caption)
    
    if remove_combined:
        sep_idx = [i for i in range(len(tokenized['input_ids'])) if tokenized['input_ids'][i] in [101, 102, 1012]]
        
        phrases = []
        for logit in logits:
            max_idx = logit.argmax()
            insert_idx = bisect.bisect_left(sep_idx, max_idx)
            right_idx = sep_idx[insert_idx]
            left_idx = sep_idx[insert_idx - 1]
            phrases.append(get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer, left_idx, right_idx).replace('.', ''))
    else:
        phrases = [
            get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer).replace('.', '')
            for logit
            in logits
        ]

    return boxes, logits.max(dim=1)[0], phrases

def str_to_list(camera_names_str):
    # Split a comma-separated string and strip whitespace around each item
    camera_names = [name.strip() for name in camera_names_str.split(',')]
    return camera_names

def track_with_sam2(video_path, ann_frame_idx, input_boxes, labels):
    # Collect all frame file names in order
    frame_names = [p for p in os.listdir(video_path) if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    inference_state = video_predictor.init_state(video_path=video_path)

    # Register the annotation boxes on the base frame
    for object_id, (box, label) in enumerate(zip(input_boxes, labels), start=1):
        _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=object_id,
            box=box,
        )

    # Propagate masks through the whole video segment
    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    mask_values = list(video_segments.values())
    return mask_values

def mask_image_replace(image, mask, original_image, bgr=False):
    # Keep generated pixels inside the mask, restore the original pixels elsewhere
    image = np.array(image)
    mask = np.array(mask)
    original_image = np.array(original_image)
    if bgr:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    image[mask == 0] = original_image[mask == 0]
    return image

def init_pipline(cur_pipe, sd_guidance_scale):
    pipe_config = dict()
    pipe_config['target_width'] = 640
    pipe_config['target_height'] = 480

    if cur_pipe == 'sd3':
        # Stable Diffusion 3 Medium (text-to-image) pipeline
        pipe = SD3Pipeline.from_pretrained(
            config.SD3_MEDIUM_PATH,
            torch_dtype=torch.float16)
        pipe.text_encoder.to(torch.float16)
        pipe.to("cuda")
        pipe.enable_attention_slicing()
        pipe.enable_model_cpu_offload()
        if sd_guidance_scale is not None:
            pipe_config['sd_guidance_scale'] = sd_guidance_scale
        pipe_config['sd_num_inference_steps'] = 30

    return pipe, pipe_config

def run_pipe(pipe, pipe_config, image_pil, mask_pil, sd_prompt):

    if cur_pipe == 'sd3':
        image = pipe(
            prompt=sd_prompt,
            image=image_pil,
            mask_image=mask_pil,
            negative_prompt="no objects, no items, no decorations, no accessories, no hands, no people, no chairs, no background, no walls, no clutter, no shadows, no lighting effects, no reflections of surroundings, no non-desk elements, no external context, only the surface of the desk, minimalism, focus on texture and material only.",
            width=pipe_config['target_width'],
            height=pipe_config['target_height'],
            num_images_per_prompt=1,
            guidance_scale=pipe_config['sd_guidance_scale'],
            num_inference_steps=pipe_config['sd_num_inference_steps'],
        ).images
    return image

def background_synthesis_process_batch(masks_batch, images_batch, batch_size=14, bgr=False):
    """
    Process multiple images in batch for background synthesis

    Args:
        masks_batch: List of masks [N, H, W]
        images_batch: List of images [N, H, W, 3]
        batch_size: Size of each processing batch
    Returns:
        List of augmented images
    """
    n_samples = len(masks_batch)
    n_batches = (n_samples + batch_size - 1) // batch_size
    augmented_images = []

    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_samples)

        # Prepare current batch
        curr_masks = masks_batch[start_idx:end_idx]
        curr_images = images_batch[start_idx:end_idx]

        # Convert to PIL images and resize
        mask_pils = [Image.fromarray(mask) for mask in curr_masks]
        image_pils = [Image.fromarray(img) for img in curr_images]

        mask_pils = [mask.resize((640, 480)) for mask in mask_pils]
        image_pils = [img.resize((640, 480)) for img in image_pils]

        # Generate one random surface prompt per sample
        sd_prompts = [random.choice(list(desktop_prompt_dict.values()))[0] for _ in range(len(curr_masks))]

        # Run pipeline in batch
        images = run_pipe(pipe, pipe_config, image_pils, mask_pils, sd_prompts)

        # Post-process each image: keep original pixels outside the mask
        for image, mask_pil, image_pil in zip(images, mask_pils, image_pils):
            image = image.resize((640, 480))
            image = mask_image_replace(image, mask_pil, image_pil, bgr=bgr)
            augmented_images.append(image)

    return augmented_images

desktop_prompt_dict = {
    '1': ['a sleek wooden desk surface with fine grain patterns running horizontally, offering a clean and uncluttered appearance, detailed, high quality'],
    '2': ['a polished walnut desk surface with deep, rich grain patterns running diagonally, minimalistic and smooth'],
    '3': ['a light pine desk surface with soft, straight grain lines, warm and natural, semi-matte finish'],
    '4': ['a reclaimed wood desk surface with weathered, faint grain lines and a matte texture, subtle yet rustic'],
    '5': ['a bamboo desk surface featuring tightly packed linear patterns, smooth and eco-friendly'],
    '6': ['a dark mahogany desk surface with subtle grain swirls and a polished sheen, elegant and luxurious'],
    '7': ['a birch wood desk surface with pale tones and faint linear grain, offering a clean and simple look'],
    '8': ['a natural maple desk surface with light wavy grain patterns and a smooth, semi-gloss finish'],
    '9': ['a teak desk surface with rich horizontal grain lines, polished and warm in tone'],
    '10': ['a whitewashed wooden desk surface with faint horizontal grain, clean and airy, semi-matte finish'],
    '11': ['a driftwood-inspired desk surface with faded, weathered grain patterns and a soft matte texture'],
    '12': ['a cherry wood desk surface with gentle grain swirls and a glossy finish, vibrant and warm'],
    '13': ['a black walnut desk surface with darker, flowing grain patterns, polished and sophisticated'],
    '14': ['a rosewood desk surface with intricate darker grain streaks against a reddish-brown background, smooth finish'],
    '15': ['a laminated wood desk surface with natural grain patterns, durable and smooth'],
    '16': ['a smooth ashwood desk surface with faint linear grain and a light, natural tone'],
    '17': ['a cedar desk surface with soft horizontal grain lines and a warm reddish hue, semi-matte finish'],
    '18': ['a handcrafted acacia wood desk surface with flowing grain patterns and a polished texture, detailed and natural'],
    '19': ['a hickory wood desk surface with gentle grain lines and a smooth, semi-gloss finish'],
    '20': ['a larch wood desk surface with simple, pale grain lines running lengthwise, matte texture'],
    '21': ['a sleek black marble desk surface with fine white veining, polished and elegant'],
    '22': ['a white granite desk surface with tiny specks of gray and beige, smooth and glossy'],
    '23': ['a travertine desk surface with subtle parallel lines and a matte finish, natural and earthy'],
    '24': ['a dark slate desk surface with faint natural streaks, smooth and understated'],
    '25': ['a quartz desk surface with soft speckled patterns and a semi-gloss finish, modern and durable'],
    '26': ['a polished onyx desk surface with dark tones and faint translucent veining, sophisticated and sleek'],
    '27': ['a white Carrara marble desk surface with fine gray veins running softly across, polished and clean'],
    '28': ['a smooth alabaster desk surface with faint swirls of white and cream, lightly polished and elegant'],
    '29': ['a beige sandstone desk surface with fine parallel grain lines and a smooth surface, matte finish'],
    '30': ['a terrazzo desk surface with muted multicolored chips set in a smooth surface, semi-matte texture'],
    '31': ['a polished granite desk surface with intricate natural speckles, glossy and refined'],
    '32': ['a soapstone desk surface with soft flowing textures and a silky matte finish, subtle and modern'],
    '33': ['a dark quartzite desk surface with gentle streaks of gray and white, polished and minimalist'],
    '34': ['a smooth concrete desk surface with light speckled textures and a matte surface, industrial and clean'],
    '35': ['a travertine desk surface with small pits and soft tonal variations, matte and earthy'],
    '36': ['a lightly polished basalt desk surface with subtle dark streaks, minimal and contemporary'],
    '37': ['a sandstone desk surface with soft beige tones and faint parallel patterns, smooth and natural'],
    '38': ['a matte white quartz desk surface with tiny gray specks evenly distributed, subtle and clean'],
    '39': ['a green marble desk surface with delicate white veining and a polished finish, luxurious and classic'],
    '40': ['a sandy beige marble desk surface with faint swirling patterns and a semi-matte finish'],
    '41': ['a smooth agate desk surface with soft flowing patterns and a glossy finish, refined and natural'],
    '42': ['a calcite desk surface with subtle cloudy textures and a faint shimmer, polished and elegant'],
    '43': ['a wooden and marble desk surface, combining teak wood with a central panel of polished black marble'],
    '44': ['a reclaimed wood desk surface with a slight mix of marble inlays, soft and natural patterns'],
    '45': ['a travertine stone and walnut wood hybrid desk surface, showcasing natural textures from both materials'],
    '46': ['a composite desk surface of smooth light oak and speckled quartz, modern and sleek'],
    '47': ['a desk surface combining light birch wood edges with a smooth dark slate center, clean and sophisticated'],
    '48': ['a polished teak wood desk surface with subtle travertine inlays for a refined finish'],
    '49': ['a walnut desk surface with soft marble inlay accents, blending natural textures harmoniously'],
    '50': ['a wood and quartz composite desk surface featuring smooth surfaces and minimal patterns'],
    '51': ['a pale pine wood desk surface with faint natural grain patterns running horizontally, simple and matte'],
    '52': ['a laminated beech wood desk surface with evenly spaced grain lines, light and smooth in appearance'],
    '53': ['a dark-stained oak desk surface with pronounced linear grain patterns and a semi-matte finish'],
    '54': ['a weathered wood desk surface with soft grain lines and a matte finish, rustic and minimal'],
    '55': ['a teak desk surface with clean horizontal lines of grain and a warm tone, polished and natural'],
    '56': ['a bamboo desk surface with compact linear grain and a smooth, eco-friendly finish'],
    '57': ['a black walnut desk surface with contrasting grain tones and a high-gloss finish, refined and elegant'],
    '58': ['a handcrafted cherry wood desk surface with subtle swirling grain, polished and vibrant'],
    '59': ['a cedar desk surface with faint horizontal grain patterns, semi-matte and natural'],
    '60': ['a lightly stained ashwood desk surface with faint linear grain and a natural look'],
    '61': ['a whitewashed wooden desk surface with soft, pale grain and a matte finish, simple and clean'],
    '62': ['a reclaimed wood desk surface with faint natural imperfections and weathered grain lines, matte finish'],
    '63': ['a polished rosewood desk surface with intricate darker streaks against a reddish-brown background'],
    '64': ['a travertine desk surface with soft linear patterns and a smooth surface, semi-matte finish'],
    '65': ['a quartzite desk surface with subtle, soft streaks of white, modern and polished'],
    '66': ['a basalt desk surface with a dark matte finish and faint natural streaks, simple and industrial'],
    '67': ['a smooth sandstone desk surface with parallel beige grain and a matte texture, minimalist design'],
    '68': ['a matte white quartz desk surface with faint speckles evenly spread, clean and understated'],
    '69': ['a pale granite desk surface with fine speckling, polished for a soft, modern finish'],
    '70': ['a polished black marble desk surface with soft, delicate white veining, glossy and sophisticated'],
    '71': ['a travertine desk surface featuring parallel lines and subtle pits, matte and natural in tone'],
    '72': ['a slate desk surface with faint streaks and a semi-matte finish, perfect for industrial settings'],
    '73': ['a terrazzo desk surface with muted multicolored chips and a smooth, polished surface, modern appeal'],
    '74': ['a limestone desk surface with subtle, pale textures and a matte surface, natural and elegant'],
    '75': ['a marble composite desk surface with soft flowing patterns and a polished finish, refined and minimalist'],
    '76': ['a light oak desk surface with clean horizontal grain and a natural semi-matte finish'],
    '77': ['a dark walnut desk surface with deep linear grain, polished for a high-end, classic feel'],
    '78': ['a reclaimed wood desk surface with weathered grain details and a smooth, matte surface, rustic yet modern'],
    '79': ['a birch wood desk surface with pale tones and faint linear grain, offering a clean and bright look'],
    '80': ['a bamboo desk surface with fine, compact grain patterns and a smooth finish, eco-friendly design'],
    '81': ['a cedar desk surface with warm, reddish grain patterns running horizontally, semi-gloss finish'],
    '82': ['a cherry wood desk surface with soft swirls in the grain, polished for a vibrant, elegant touch'],
    '83': ['a teak wood desk surface with dark horizontal grain lines and a glossy finish, high-quality appearance'],
    '84': ['a rosewood desk surface featuring intricate darker streaks against a reddish tone, polished and luxurious'],
    '85': ['a whitewashed pine wood desk surface with faint horizontal grain, semi-matte and simple'],
    '86': ['a quartzite desk surface with soft, flowing streaks and a polished surface, modern and sleek'],
    '87': ['a travertine stone desk surface with faint natural pits and soft parallel lines, matte and refined'],
    '88': ['a smooth alabaster desk surface with subtle swirling patterns and a lightly polished surface, minimalist'],
    '89': ['a terrazzo desk surface with neutral-tone chips set in a matte surface, perfect for contemporary designs'],
    '90': ['a black granite desk surface with fine speckling, polished and refined'],
    '91': ['a polished white marble desk surface with elegant veining, glossy and luxurious'],
    '92': ['a matte-finish basalt desk surface with faint natural textures, clean and minimalist'],
    '93': ['a light oak desk surface with soft grain patterns, natural matte finish'],
    '94': ['a concrete desk surface with smooth and simple texture, industrial yet modern'],
    '95': ['a soft walnut wood desk surface with gentle grain patterns, lightly polished'],
    '96': ['a black slate desk surface with faint gray streaks, polished and refined'],
    '97': ['a whitewashed birch desk surface with subtle linear grain, matte finish'],
    '98': ['a warm cherry wood desk surface with gentle swirls of grain, polished and soft'],
    '99': ['a sand-colored marble desk surface with faint patterns of beige veining'],
    '100': ['a polished rosewood desk surface with intricate dark streaks, luxurious and modern'],
    '101': ['a smooth walnut surface with tight, horizontal grain and a glossy finish, elegant and modern'],
    '102': ['a light oak surface with subtle vertical grain lines and a semi-gloss finish, simple and clean'],
    '103': ['a polished cherry wood surface with soft, wavy grain and a deep reddish hue, sophisticated and warm'],
    '104': ['a light-colored maple surface with fine, straight grain and a smooth, semi-matte finish'],
    '105': ['a reclaimed wood surface with visible knots and an uneven grain pattern, rustic and natural'],
    '106': ['a dark walnut surface with subtle, flowing grain and a semi-glossy finish, sleek and timeless'],
    '107': ['a smooth bamboo surface with compact linear grain, eco-friendly and polished'],
    '108': ['a grayish stone surface with faint marbling, smooth and matte, perfect for a modern space'],
    '109': ['a polished mahogany surface with intricate grain swirls and a high-gloss finish, luxurious and refined'],
    '110': ['a birch wood surface with soft grain lines and a natural matte finish, understated and minimal'],
    '111': ['a sleek stone surface with light gray tones and faint speckling, subtle and refined'],
    '112': ['a natural pine surface with light golden hues and visible grain patterns, smooth and simple'],
    '113': ['a smooth granite surface with small black speckles, polished and contemporary'],
    '114': ['a medium-brown oak surface with a glossy finish, showcasing horizontal grain and a sleek look'],
    '115': ['a polished marble surface with soft gray veins and a glossy finish, elegant and smooth'],
    '116': ['a simple ashwood surface with soft grain lines, matte finish, and natural texture'],
    '117': ['a sleek white quartz surface with subtle light gray streaks, smooth and durable'],
    '118': ['a dark cherry wood surface with rich grain patterns and a satin finish, elegant and timeless'],
    '119': ['a clean maple surface with fine, even grain, minimalistic and smooth'],
    '120': ['a polished black granite surface with intricate gray and white veining, luxurious and sleek'],
    '121': ['a light-colored birch surface with straight grain lines and a smooth, matte finish, clean and natural'],
    '122': ['a reclaimed oak surface with rustic grain lines, worn edges, and a matte finish'],
    '123': ['a white Carrara marble surface with delicate gray veining, polished to a glossy finish'],
    '124': ['a polished black marble surface with faint white streaks and a high-gloss surface'],
    '125': ['a sleek stone surface with subtle grain lines and a polished, glossy finish'],
    '126': ['a dark walnut surface with vertical grain patterns, smooth and satin finish'],
    '127': ['a light-colored maple surface with minimal grain, smooth and modern'],
    '128': ['a rough stone surface with natural, exposed texture, matte finish, and organic appeal'],
    '129': ['a polished teak surface with horizontal grain, smooth finish, and warm tones'],
    '130': ['a natural stone surface with intricate, light gray veining, polished and smooth'],
    '131': ['a light ashwood surface with minimal grain, subtle texture, and a matte finish'],
    '132': ['a reclaimed wood surface with deep grain patterns and a matte finish, organic and rustic'],
    '133': ['a smooth stone surface with faint white speckles, polished for a sleek and modern look'],
    '134': ['a polished bamboo surface with vertical grain, smooth and eco-friendly'],
    '135': ['a dark walnut surface with pronounced grain patterns, semi-glossy finish'],
    '136': ['a light pine surface with faint grain lines and a soft matte finish, natural and warm'],
    '137': ['a smooth marble surface with fine gray veining and a polished surface, elegant and luxurious'],
    '138': ['a simple white quartz surface with faint gray streaks and a smooth, glossy finish'],
    '139': ['a reclaimed stone surface with organic texture and soft natural tones, matte finish'],
    '140': ['a polished birch surface with smooth grain and a semi-gloss finish, fresh and clean'],
    '141': ['a gray stone surface with minimal veining, smooth and matte for a contemporary feel'],
    '142': ['a light walnut surface with soft, subtle grain patterns and a smooth finish'],
    '143': ['a light ashwood surface with parallel grain lines and a clean matte finish'],
    '144': ['a polished granite surface with natural gray streaks and a smooth, shiny surface'],
    '145': ['a warm reclaimed wood surface with deep, natural grain and a matte texture'],
    '146': ['a sleek black marble surface with faint streaks of white veining, polished and refined'],
    '147': ['a polished rosewood surface with bold grain lines and a semi-gloss finish, elegant and rich'],
    '148': ['a dark oak surface with visible grain and a soft, satin finish, understated and elegant'],
    '149': ['a smooth quartz surface with subtle streaks of gray and a polished finish, sleek and modern'],
    '150': ['a reclaimed wood surface with weathered grain, knots, and a matte surface'],
    '151': ['a polished teak wood surface with fine horizontal grain patterns, smooth and elegant'],
    '152': ['a polished marble surface with subtle gray veins, smooth and luxurious'],
    '153': ['a light-colored oak surface with soft grain lines and a matte finish, clean and natural'],
    '154': ['a sleek stone surface with smooth texture, fine gray streaks, and a polished look'],
    '155': ['a dark walnut surface with intricate grain patterns and a glossy finish, refined and rich'],
    '156': ['a light ashwood surface with fine grain lines and a smooth, natural matte finish'],
    '157': ['a polished travertine surface with soft, parallel grain and a matte finish'],
    '158': ['a smooth quartzite surface with natural, subtle veining and a high-gloss finish'],
    '159': ['a light-colored pine surface with faint grain lines, matte finish, and a soft appearance'],
    '160': ['a sleek black granite surface with faint gray streaks, polished and refined'],
    '161': ['a polished oak wood surface with straight grain and a smooth, semi-gloss finish'],
    '162': ['a smooth stone surface with faint streaks of white and light gray, polished and subtle'],
    '163': ['a polished walnut wood surface with fine grain patterns and a satin finish, elegant and warm'],
    '164': ['a reclaimed bamboo surface with tightly packed vertical grain lines, eco-friendly and smooth'],
    '165': ['a natural slate surface with faint texture and gray streaks, matte finish and industrial feel'],
    '166': ['a polished black marble surface with slight veining and a glossy finish, sleek and contemporary'],
    '167': ['a white Carrara marble surface with soft gray veining and a polished smooth surface'],
    '168': ['a polished pine wood surface with soft grain and a semi-gloss finish, clean and modern'],
    '169': ['a dark stone surface with subtle veining, matte finish and smooth texture'],
    '170': ['a reclaimed walnut surface with visible grain patterns and a satin matte finish'],
    '171': ['a polished quartz surface with light gray streaks, sleek and minimalistic'],
    '172': ['a smooth teak wood surface with intricate grain patterns and a polished semi-gloss finish'],
    '173': ['a smooth stone surface with delicate white streaks, polished and sleek'],
    '174': ['a warm oak wood surface with soft grain and a polished smooth surface'],
    '175': ['a polished granite surface with soft speckles and a smooth finish'],
    '176': ['a reclaimed wood surface with natural, weathered grain and matte finish, rustic'],
    '177': ['a smooth gray quartz surface with fine natural veining and a sleek finish'],
    '178': ['a dark walnut wood surface with intricate horizontal grain and a satin finish'],
    '179': ['a light stone surface with fine gray streaks, smooth and glossy, minimalistic'],
    '180': ['a reclaimed pine wood surface with weathered, natural grain and matte finish'],
    '181': ['a polished teak surface with fine horizontal grain patterns, sleek and elegant'],
    '182': ['a smooth stone surface with delicate white streaks, polished and sleek'],
    '183': ['a light-colored walnut wood surface with soft grain patterns, matte finish'],
    '184': ['a polished quartz surface with subtle streaks of gray, sleek and modern'],
    '185': ['a smooth travertine surface with soft, parallel lines and matte finish'],
    '186': ['a reclaimed wood surface with deep grain patterns and natural weathering, rustic'],
    '187': ['a polished marble surface with gray streaks and a luxurious glossy finish'],
    '188': ['a sleek black granite surface with faint gray streaks, polished and refined'],
    '189': ['a dark reclaimed wood surface with visible grain and natural texture'],
    '190': ['a polished rosewood surface with smooth horizontal grain and a glossy surface'],
    '191': ['a light-colored stone surface with natural texture and faint streaks'],
    '192': ['a smooth slate surface with subtle veining and matte finish, sleek'],
    '193': ['a polished bamboo surface with smooth texture and fine linear grain'],
    '194': ['a dark stone surface with gray streaks and matte finish, sleek and elegant'],
    '195': ['a light-colored oak surface with straight, minimal grain and a smooth finish'],
    '196': ['a rich mahogany surface with deep grain and a polished gloss'],
    '197': ['a sleek stone surface with gentle gray streaks, subtle and polished'],
    '198': ['a smooth walnut surface with intricate grain and a deep glossy finish'],
    '199': ['a reclaimed oak surface with a worn, rustic charm, matte and natural'],
    '200': ['a soft maple surface with fine straight grain and a smooth matte finish'],
    '201': ['a smooth walnut wood surface with subtle grain patterns and a soft, matte finish, clean and refined'],
    '202': ['a light-colored oak surface with minimal grain and a natural matte finish, simple and modern'],
    '203': ['a dark maple surface with intricate horizontal grain and a glossy finish, elegant and polished'],
    '204': ['a smooth cherry wood surface with a satin finish, showcasing soft grain patterns and a refined look'],
    '205': ['a reclaimed wood surface with natural grain and visible knots, matte finish, rustic and warm'],
    '206': ['a polished teak wood surface with fine straight grain, rich and warm tones'],
    '207': ['a natural birch wood surface with soft, even grain and a smooth, semi-gloss finish, minimalistic'],
    '208': ['a smooth bamboo surface with vertical grain, eco-friendly and polished to a sleek finish'],
    '209': ['a reclaimed oak surface with visible grain and rustic charm, matte finish for an organic look'],
    '210': ['a simple maple wood surface with soft grain patterns, polished finish, clean and contemporary'],
    '211': ['a dark walnut surface with smooth texture and elegant horizontal grain patterns'],
    '212': ['a light pine wood surface with faint grain patterns, matte finish for a fresh and natural look'],
    '213': ['a polished black granite surface with faint speckling and a smooth, high-gloss finish'],
    '214': ['a light-colored stone surface with fine grain texture and a natural matte finish'],
    '215': ['a sleek stone surface with delicate gray streaks, polished for a refined and modern look'],
    '216': ['a smooth marble surface with light veining, glossy surface for a clean, sophisticated feel'],
    '217': ['a polished Carrara marble surface with soft gray veins and a smooth, elegant finish'],
    '218': ['a smooth dark stone surface with faint veining and a polished matte surface'],
    '219': ['a light stone surface with subtle gray and white streaks, smooth surface for a contemporary design'],
    '220': ['a reclaimed wood surface with deep grain patterns, rustic charm and natural texture'],
    '221': ['a polished walnut wood surface with fine horizontal grain, soft satin finish'],
    '222': ['a simple stone surface with natural gray hues and a smooth texture, sleek and modern'],
    '223': ['a polished marble surface with delicate veining, smooth surface and a luxurious finish'],
    '224': ['a warm oak wood surface with intricate grain patterns and a glossy finish, natural elegance'],
    '225': ['a sleek bamboo surface with tight grain, eco-friendly and modern with a smooth matte finish'],
    '226': ['a smooth ashwood surface with clean, subtle grain lines and a matte finish'],
    '227': ['a polished teak surface with fine, vertical grain patterns and a rich, dark finish'],
    '228': ['a reclaimed wood surface with weathered grain, natural matte finish, rustic and authentic'],
    '229': ['a polished black marble surface with soft gray veins, sleek and glossy for a contemporary touch'],
    '230': ['a smooth light-colored stone surface with fine veining and a polished finish'],
    '231': ['a light oak wood surface with visible grain patterns, smooth matte finish for a natural look'],
    '232': ['a polished granite surface with black and gray speckling, smooth and sophisticated'],
    '233': ['a natural reclaimed wood surface with irregular grain and a matte finish, organic and earthy'],
    '234': ['a polished rosewood surface with a deep, rich hue and visible grain patterns'],
    '235': ['a light-colored ashwood surface with clean, fine grain and a semi-gloss finish'],
    '236': ['a dark granite surface with intricate gray and white veins, polished and refined'],
    '237': ['a reclaimed wood surface with worn edges and visible grain, matte finish for rustic appeal'],
    '238': ['a polished marble surface with soft gray veins and a glossy finish, smooth and sleek'],
    '239': ['a natural oak wood surface with straight grain and a satin finish, simple and elegant'],
    '240': ['a dark stone surface with faint gray veining, smooth polished texture for a contemporary design'],
    '241': ['a reclaimed walnut wood surface with visible grain, natural texture and a matte finish'],
    '242': ['a smooth quartz surface with soft gray streaks and a polished, glossy texture'],
    '243': ['a sleek black granite surface with subtle speckling, polished to a high-gloss finish'],
    '244': ['a smooth light wood surface with fine grain lines, matte finish for a clean and natural look'],
    '245': ['a polished teak wood surface with smooth horizontal grain patterns and a rich finish'],
    '246': ['a natural stone surface with faint streaks and smooth texture, perfect for a modern space'],
    '247': ['a reclaimed oak wood surface with visible grain, natural texture, and a matte finish'],
    '248': ['a smooth granite surface with black and gray speckles, polished for a refined look'],
    '249': ['a light-colored stone surface with fine streaks, polished to a smooth finish'],
    '250': ['a polished mahogany wood surface with intricate grain patterns and a glossy finish'],
    '251': ['a sleek bamboo surface with tightly packed grain, eco-friendly and polished to perfection'],
    '252': ['a light-colored walnut surface with fine grain lines and a semi-gloss finish, minimalist'],
    '253': ['a natural marble surface with subtle veining and a smooth, glossy texture'],
    '254': ['a polished birch wood surface with fine, even grain and a semi-gloss finish, fresh and natural'],
    '255': ['a reclaimed wood surface with intricate grain patterns and visible knots, matte finish'],
    '256': ['a dark stone surface with gray streaks and a sleek, polished finish'],
    '257': ['a smooth maple surface with subtle grain and a soft matte finish'],
    '258': ['a reclaimed bamboo wood surface with vertical grain and a smooth, glossy finish'],
    '259': ['a polished granite surface with intricate gray and white speckling, sleek and sophisticated'],
    '260': ['a reclaimed wood surface with visible grain and a matte finish, natural and earthy'],
    '261': ['a polished walnut wood surface with fine grain and a satin finish, sleek and elegant'],
    '262': ['a smooth travertine stone surface with soft veins, polished finish and contemporary appeal'],
    '263': ['a light-colored birch wood surface with minimal grain and a natural finish, simple and clean'],
    '264': ['a sleek marble surface with delicate gray veining and a smooth, polished texture'],
    '265': ['a smooth stone surface with light gray hues and faint streaks, polished for a modern touch'],
    '266': ['a reclaimed pine wood surface with rough texture and visible knots, natural and matte finish'],
    '267': ['a polished rosewood surface with visible grain and a glossy finish, elegant and refined'],
    '268': ['a light-colored ashwood surface with visible, subtle grain and a soft matte finish'],
    '269': ['a dark granite surface with intricate gray and white veins, polished and refined'],
    '270': ['a reclaimed wood surface with rich texture, visible grain, and a matte finish'],
    '271': ['a polished maple wood surface with fine horizontal grain, semi-gloss finish'],
    '272': ['a sleek stone surface with faint gray veining, smooth and polished for a modern look'],
    '273': ['a smooth reclaimed wood surface with natural grain patterns, matte finish'],
    '274': ['a dark reclaimed stone surface with gray streaks and a sleek, matte texture'],
    '275': ['a smooth light wood surface with subtle grain, soft matte finish for a natural look'],
    '276': ['a polished rosewood surface with visible grain and a glossy finish, elegant and refined'],
    '277': ['a light-colored marble surface with fine veining, glossy and polished for a sleek look'],
    '278': ['a polished walnut wood surface with intricate grain patterns, smooth finish'],
    '279': ['a light pine wood surface with smooth texture and faint grain lines'],
    '280': ['a smooth stone surface with fine gray streaks, polished and smooth'],
    '281': ['a reclaimed wood surface with visible knots, natural texture, and a matte finish'],
    '282': ['a polished maple wood surface with fine horizontal grain, semi-gloss finish'],
    '283': ['a sleek bamboo wood surface with vertical grain, polished and eco-friendly'],
    '284': ['a polished mahogany surface with fine, straight grain and a glossy finish, luxurious'],
    '285': ['a smooth quartz surface with gray veins and a glossy, smooth texture'],
    '286': ['a light-colored stone surface with fine streaks, polished to a smooth finish'],
    '287': ['a smooth reclaimed wood surface with natural grain patterns, matte finish'],
    '288': ['a dark reclaimed stone surface with gray streaks and a sleek, matte texture'],
    '289': ['a smooth light wood surface with subtle grain, soft matte finish for a natural look'],
    '290': ['a polished rosewood surface with visible grain and a glossy finish, elegant and refined'],
    '291': ['a light-colored marble surface with fine veining, glossy and polished for a sleek look'],
    '292': ['a polished walnut wood surface with intricate grain patterns, smooth finish'],
    '293': ['a light pine wood surface with smooth texture and faint grain lines'],
    '294': ['a smooth stone surface with fine gray streaks, polished and smooth'],
    '295': ['a reclaimed wood surface with visible knots, natural texture, and a matte finish'],
    '296': ['a polished maple wood surface with fine horizontal grain, semi-gloss finish'],
    '297': ['a sleek bamboo wood surface with vertical grain, polished and eco-friendly'],
    '298': ['a polished mahogany surface with fine, straight grain and a glossy finish, luxurious'],
    '299': ['a smooth quartz surface with gray veins and a glossy, smooth texture'],
    '300': ['a light-colored stone surface with fine streaks, polished to a smooth finish'],
    '301': ['a smooth birch wood surface with fine, straight grain and a matte finish, modern and minimalistic'],
    '302': ['a polished oak wood surface with subtle grain patterns, smooth texture, and warm natural tones'],
    '303': ['a rich mahogany surface with deep grain and a glossy finish, elegant and bold'],
    '304': ['a reclaimed pine surface with weathered grain and visible imperfections, rustic with a matte finish'],
    '305': ['a sleek maple surface with horizontal grain and a satin finish, refined and contemporary'],
    '306': ['a polished teak surface with tight grain and warm tones, smooth and sophisticated'],
    '307': ['a walnut surface with light horizontal grain and a natural matte finish, clean and modern'],
    '308': ['a glossy rosewood surface with intricate grain patterns, refined and luxurious'],
    '309': ['a bamboo surface with vertical grain, eco-friendly design and a smooth matte finish'],
    '310': ['a light oak surface with visible grain and a semi-gloss finish, simple and elegant'],
    '311': ['a cherry wood surface with soft grain patterns and a satin finish, timeless and warm'],
    '312': ['a stone surface with light gray veining, polished for a sleek and modern appeal'],
    '313': ['a polished white marble surface with soft gray streaks and a high-gloss finish'],
    '314': ['a smooth granite surface with dark speckling and a polished finish'],
    '315': ['a dark stone surface with delicate veins, sleek polished texture'],
    '316': ['a smooth black marble surface with soft white veining, glossy and refined'],
    '317': ['a light-colored stone surface with fine veins, polished for a clean and modern look'],
    '318': ['a simple gray stone surface with smooth texture and subtle veining'],
    '319': ['a polished Carrara marble surface with fine gray veining and smooth finish'],
    '320': ['a smooth granite surface with dark gray streaks, glossy and refined'],
    '321': ['a reclaimed wood surface with visible knots and grain patterns, rustic with a matte finish'],
    '322': ['a polished stone surface with subtle gray veins and smooth texture'],
    '323': ['a smooth travertine stone surface with delicate streaks, polished finish'],
    '324': ['a light stone surface with faint veins and a polished texture'],
    '325': ['a reclaimed oak surface with visible grain and a matte finish'],
    '326': ['a smooth stone surface with faint gray streaks and polished finish'],
    '327': ['a light-colored marble surface with delicate veining and a glossy finish'],
    '328': ['a simple stone surface with soft gray streaks and a sleek polished finish'],
    '329': ['a polished quartz surface with soft white veins, smooth and glossy'],
    '330': ['a reclaimed pine surface with rough texture, matte finish'],
    '331': ['a smooth black granite surface with subtle veins, high-gloss finish'],
    '332': ['a light-colored stone surface with gentle gray veins and a polished texture'],
    '333': ['a simple ashwood surface with soft grain and smooth matte finish'],
    '334': ['a polished walnut surface with fine grain and a satin finish'],
    '335': ['a sleek stone surface with faint gray streaks and a smooth, glossy finish'],
    '336': ['a smooth marble surface with delicate veining and a polished texture'],
    '337': ['a smooth oak surface with light grain and a matte finish'],
    '338': ['a dark stone surface with subtle gray streaks, polished texture'],
    '339': ['a polished rosewood surface with dark grain and glossy finish'],
    '340': ['a simple reclaimed wood surface with visible grain and matte finish'],
    '341': ['a sleek stone surface with subtle streaks and smooth texture'],
    '342': ['a smooth teak surface with fine grain and a glossy finish'],
    '343': ['a light-colored marble surface with faint gray veins, high-gloss texture'],
    '344': ['a polished black stone surface with faint veins, sleek finish'],
    '345': ['a natural stone surface with faint veins and a polished smooth finish'],
    '346': ['a simple granite surface with faint gray streaks and a polished texture'],
    '347': ['a reclaimed walnut surface with visible grain, matte finish'],
    '348': ['a polished marble surface with soft gray veins and a sleek, glossy texture'],
    '349': ['a reclaimed wood surface with rustic texture and natural finish'],
    '350': ['a light-colored stone surface with subtle streaks and smooth finish'],
    '351': ['a dark stone surface with faint white veining, sleek polished finish'],
    '352': ['a polished marble surface with soft gray and white streaks, glossy texture'],
    '353': ['a sleek stone surface with delicate veins and polished finish'],
    '354': ['a polished ashwood surface with smooth texture and fine grain'],
    '355': ['a reclaimed oak surface with visible grain and matte finish'],
    '356': ['a polished rosewood surface with soft grain patterns, high-gloss finish'],
    '357': ['a sleek bamboo surface with fine vertical grain, matte finish'],
    '358': ['a dark stone surface with faint streaks and a smooth polished texture'],
    '359': ['a smooth marble surface with fine gray veining and glossy finish'],
    '360': ['a reclaimed pine surface with rustic texture, matte finish'],
    '361': ['a polished teak wood surface with fine grain, glossy texture'],
    '362': ['a light-colored stone surface with fine streaks, polished texture'],
    '363': ['a smooth walnut wood surface with deep, rich grain and matte finish'],
    '364': ['a simple stone surface with soft gray veins and smooth texture'],
    '365': ['a reclaimed wood surface with visible grain and a natural matte finish'],
    '366': ['a sleek granite surface with soft gray speckles, polished texture'],
    '367': ['a smooth travertine surface with light streaks, matte finish'],
    '368': ['a polished stone surface with delicate gray veins, smooth finish'],
    '369': ['a reclaimed oak surface with visible grain and subtle texture, matte finish'],
    '370': ['a sleek bamboo surface with vertical grain, soft matte finish'],
    '371': ['a smooth marble surface with faint gray streaks and glossy finish'],
    '372': ['a polished walnut wood surface with smooth texture, glossy finish'],
    '373': ['a simple reclaimed wood surface with natural grain and matte finish'],
    '374': ['a light stone surface with fine streaks and polished texture'],
    '375': ['a polished Carrara marble surface with soft gray veining and smooth texture'],
    '376': ['a smooth bamboo wood surface with vertical grain, eco-friendly finish'],
    '377': ['a light-colored granite surface with fine gray streaks, polished texture'],
    '378': ['a reclaimed wood surface with rough edges, visible knots and matte finish'],
    '379': ['a polished maple wood surface with subtle grain and high-gloss finish'],
    '380': ['a smooth stone surface with faint gray streaks and a refined texture'],
    '381': ['a reclaimed oak wood surface with visible grain and natural texture'],
    '382': ['a sleek stone surface with delicate gray streaks and a polished finish'],
    '383': ['a polished teak wood surface with fine horizontal grain and a glossy finish'],
    '384': ['a smooth light stone surface with fine veins, high-gloss finish'],
    '385': ['a polished granite surface with fine gray streaks, sleek texture'],
    '386': ['a reclaimed wood surface with visible knots and matte finish, rustic charm'],
    '387': ['a smooth dark stone surface with faint streaks and polished texture'],
    '388': ['a sleek bamboo surface with fine vertical grain, polished finish'],
    '389': ['a polished marble surface with soft gray streaks and high-gloss texture'],
    '390': ['a simple reclaimed wood surface with fine grain patterns, matte finish'],
    '391': ['a dark stone surface with subtle gray veins, smooth texture'],
    '392': ['a reclaimed walnut wood surface with visible grain and a natural matte finish'],
    '393': ['a smooth rosewood surface with fine grain and a polished texture'],
    '394': ['a light stone surface with delicate streaks and high-gloss finish'],
    '395': ['a sleek marble surface with fine gray veins and polished texture'],
    '396': ['a polished walnut surface with smooth finish and fine grain'],
    '397': ['a smooth oak surface with subtle grain and polished texture'],
    '398': ['a sleek stone surface with subtle streaks, high-gloss texture'],
    '399': ['a polished rosewood surface with rich grain and glossy finish'],
    '400': ['a reclaimed stone surface with subtle gray streaks and smooth, polished texture'],
    '401': ['a smooth walnut wood surface with subtle horizontal grain patterns, matte finish'],
    '402': ['a light oak wood surface with visible grain lines and a satin finish'],
    '403': ['a polished mahogany surface with deep rich grain patterns and glossy finish'],
    '404': ['a simple bamboo surface with vertical grain, eco-friendly matte finish'],
    '405': ['a reclaimed wood surface with visible knots and natural grain patterns, matte finish'],
    '406': ['a smooth dark stone surface with faint gray streaks, polished finish'],
    '407': ['a light-colored marble surface with delicate veining, smooth polished surface'],
    '408': ['a polished granite surface with fine speckling, sleek finish'],
    '409': ['a light stone surface with subtle gray streaks and a smooth, high-gloss finish'],
    '410': ['a reclaimed pine wood surface with soft grain lines, matte finish'],
    '411': ['a sleek oak wood surface with soft grain and a satin finish'],
    '412': ['a smooth rosewood surface with fine grain and polished finish'],
    '413': ['a light-colored stone surface with soft gray veins, polished finish'],
    '414': ['a polished marble surface with delicate white veins, high-gloss finish'],
    '415': ['a smooth granite surface with dark speckles, sleek finish'],
    '416': ['a simple stone surface with subtle gray veins, matte finish'],
    '417': ['a dark reclaimed wood surface with visible grain, matte finish'],
    '418': ['a smooth teak wood surface with fine grain patterns, high-gloss finish'],
    '419': ['a polished walnut wood surface with fine horizontal grain, smooth finish'],
    '420': ['a sleek black marble surface with fine gray streaks, polished finish'],
    '421': ['a smooth travertine stone surface with light gray veins, glossy finish'],
    '422': ['a reclaimed maple wood surface with visible grain patterns and matte finish'],
    '423': ['a light-colored stone surface with delicate gray streaks, polished finish'],
    '424': ['a smooth bamboo wood surface with vertical grain and matte finish'],
    '425': ['a polished rosewood surface with dark grain and high-gloss finish'],
    '426': ['a light-colored oak wood surface with subtle grain, semi-gloss finish'],
    '427': ['a reclaimed walnut wood surface with visible grain patterns and matte finish'],
    '428': ['a smooth stone surface with faint veins and glossy finish'],
    '429': ['a polished marble surface with gray and white veining, smooth finish'],
    '430': ['a sleek bamboo surface with vertical grain, eco-friendly matte finish'],
    '431': ['a smooth maple wood surface with soft grain and a satin finish'],
    '432': ['a dark stone surface with subtle gray streaks, polished finish'],
    '433': ['a polished granite surface with light speckling, smooth finish'],
    '434': ['a reclaimed oak wood surface with visible grain and natural texture'],
    '435': ['a smooth cherry wood surface with fine grain and a glossy finish'],
    '436': ['a light-colored marble surface with soft gray veins, smooth polished finish'],
    '437': ['a polished rosewood surface with fine horizontal grain and high-gloss finish'],
    '438': ['a sleek stone surface with subtle gray veins, polished finish'],
    '439': ['a reclaimed wood surface with rustic grain and matte finish'],
    '440': ['a smooth travertine surface with light veins and high-gloss finish'],
    '441': ['a light-colored oak surface with subtle grain, satin finish'],
    '442': ['a polished black stone surface with faint streaks, sleek finish'],
    '443': ['a light wood surface with visible fine grain and a natural matte finish'],
    '444': ['a smooth marble surface with delicate veins, glossy finish'],
    '445': ['a sleek bamboo wood surface with fine vertical grain, polished finish'],
    '446': ['a polished teak wood surface with fine grain patterns, glossy finish'],
    '447': ['a reclaimed pine wood surface with subtle grain, matte finish'],
    '448': ['a smooth dark stone surface with subtle streaks, polished finish'],
    '449': ['a simple reclaimed wood surface with visible grain, matte finish'],
    '450': ['a polished granite surface with soft gray streaks, smooth finish'],
    '451': ['a light-colored stone surface with delicate streaks, polished finish'],
    '452': ['a reclaimed wood surface with visible grain and knots, matte finish'],
    '453': ['a polished black marble surface with fine gray veins, glossy finish'],
    '454': ['a light-colored oak wood surface with visible grain, matte finish'],
    '455': ['a smooth rosewood surface with soft horizontal grain and polished finish'],
    '456': ['a polished stone surface with subtle gray veins, smooth finish'],
    '457': ['a sleek black granite surface with faint speckling, polished finish'],
    '458': ['a smooth walnut wood surface with subtle grain and satin finish'],
    '459': ['a polished stone surface with soft gray streaks, smooth finish'],
    '460': ['a reclaimed oak wood surface with visible grain and matte finish'],
    '461': ['a smooth bamboo wood surface with fine vertical grain and high-gloss finish'],
    '462': ['a light-colored stone surface with fine veins and glossy finish'],
    '463': ['a dark wood surface with deep grain patterns, polished finish'],
    '464': ['a smooth maple wood surface with soft grain and satin finish'],
    '465': ['a light-colored marble surface with soft veining and glossy finish'],
    '466': ['a reclaimed walnut wood surface with visible grain patterns, matte finish'],
    '467': ['a smooth granite surface with fine speckling and polished finish'],
    '468': ['a polished cherry wood surface with subtle grain patterns, glossy finish'],
    '469': ['a light-colored stone surface with faint streaks, high-gloss finish'],
    '470': ['a polished black stone surface with fine gray streaks, sleek finish'],
    '471': ['a smooth bamboo wood surface with vertical grain, matte finish'],
    '472': ['a smooth rosewood surface with fine grain and glossy finish'],
    '473': ['a polished granite surface with delicate gray veins, sleek finish'],
    '474': ['a simple reclaimed wood surface with visible grain patterns, matte finish'],
    '475': ['a polished marble surface with fine gray veins and smooth finish'],
    '476': ['a smooth reclaimed wood surface with visible knots, matte finish'],
    '477': ['a sleek stone surface with subtle gray streaks and smooth finish'],
    '478': ['a polished rosewood surface with intricate grain and glossy finish'],
    '479': ['a smooth travertine surface with fine gray streaks, polished finish'],
    '480': ['a reclaimed oak surface with visible grain and natural texture'],
    '481': ['a polished marble surface with soft veining and glossy finish'],
    '482': ['a smooth granite surface with subtle speckling, high-gloss finish'],
    '483': ['a reclaimed walnut wood surface with visible grain and matte finish'],
    '484': ['a polished stone surface with fine gray veins and smooth finish'],
    '485': ['a smooth bamboo wood surface with fine vertical grain, eco-friendly matte finish'],
    '486': ['a light-colored stone surface with gentle veins, polished finish'],
    '487': ['a dark stone surface with delicate gray veins, polished finish'],
    '488': ['a smooth marble surface with subtle gray veining and glossy finish'],
    '489': ['a polished walnut surface with fine grain, smooth matte finish'],
    '490': ['a reclaimed wood surface with natural grain patterns and matte finish'],
    '491': ['a light-colored oak surface with fine grain, polished finish'],
    '492': ['a smooth black granite surface with fine speckling, glossy finish'],
    '493': ['a sleek stone surface with subtle gray streaks, smooth and polished'],
    '494': ['a polished rosewood surface with rich grain and glossy finish'],
    '495': ['a reclaimed wood surface with visible grain, matte finish'],
    '496': ['a polished marble surface with fine veining, smooth glossy finish'],
    '497': ['a smooth stone surface with soft gray streaks, polished finish'],
    '498': ['a light-colored wood surface with soft grain and natural matte finish'],
    '499': ['a sleek granite surface with fine gray speckling, glossy finish'],
    '500': ['a polished black marble surface with fine streaks, high-gloss finish']
}

def load_annotations_from_json(json_path, selected_labels=None):
    """
    Load annotation data from a LabelMe-style JSON file, ordered by selected_labels.

    Args:
        json_path: path to the annotation JSON file
        selected_labels: labels to keep; their order defines the output order

    Returns:
        input_boxes: list of boxes [[x0, y0, x1, y1], ...] ordered by selected_labels
        labels: matching label list ordered by selected_labels
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Temporary dict mapping each label to its boxes
    label_to_box = {}

    for shape in data['shapes']:
        label = shape['label']
        if selected_labels is not None and label not in selected_labels:
            continue

        [[x0, y0], [x1, y1]] = shape['points']
        box = [
            min(x0, x1),  # x0
            min(y0, y1),  # y0
            max(x0, x1),  # x1
            max(y0, y1)   # y1
        ]

        if label not in label_to_box:
            label_to_box[label] = []
        label_to_box[label].append(box)

    # Reassemble results following the selected_labels order
    input_boxes = []
    labels = []

    if selected_labels:
        for label in selected_labels:
            if label in label_to_box:
                for box in label_to_box[label]:
                    input_boxes.append(box)
                    labels.append(label)
    else:
        # No order specified, use all labels
        for label, boxes in label_to_box.items():
            for box in boxes:
                input_boxes.append(box)
                labels.append(label)

    return input_boxes, labels

def create_masked_rgb(image, mask):
    """Create RGB image showing only the masked region"""
    masked_img = image.copy()
    masked_img[~mask] = 0  # Set non-mask regions to black
    return masked_img

def crop_image(image, boxes):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    cropped_images = []
    w, h = image.size
    for box in boxes:
        x0, y0, x1, y1 = map(int, box)
        # Add padding while keeping within image bounds
        x0 = max(0, x0 - 10)
        y0 = max(0, y0 - 10)
        x1 = min(w, x1 + 10)
        y1 = min(h, y1 + 10)

        cropped = image.crop((x0, y0, x1, y1))
        # Resize to a uniform size
        cropped = cropped.resize((224, 224))
        cropped_images.append(cropped)

    return cropped_images

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Region-contrastive data augmentation", add_help=True)

    parser.add_argument("--input", type=str, required=True, help="path to input folder")
    parser.add_argument("--output", type=str, required=True, help="path to output folder")
    parser.add_argument("--pipe", type=str, required=True, help="name of diffuser pipeline: sd3")
    parser.add_argument("--sd_guidance_scale", type=float, required=True)
    parser.add_argument("--camera_name", type=str, required=True, help="choose camera to adjust parameter")
    parser.add_argument("--task_name", type=str, required=True, help="name of the task")
    parser.add_argument("--selected_labels", type=str, required=True,
                    help="comma-separated list of labels to track, e.g., 'table,robot arm,pot'")
    parser.add_argument("--text_prompt", type=str, required=True,)
    parser.add_argument("--bgr", type=ast.literal_eval, default=False, help="whether to use bgr or not")
    parser.add_argument("--stride", type=int, default=3, help="process every N-th frame (default 3)")
    args = parser.parse_args()

    cur_pipe = args.pipe
    input_path = args.input
    output_path = args.output
    sd_guidance_scale = args.sd_guidance_scale
    camera_name = args.camera_name
    task_name = args.task_name
    bgr = args.bgr
    stride = args.stride
    selected_labels = [label.strip() for label in args.selected_labels.split(',')]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Init SAM2 video predictor
    sam2_checkpoint = config.SAM2_CHECKPOINT
    model_cfg = config.SAM2_MODEL_CFG
    video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)

    # Init GroundingDINO
    grounding_model = load_model(
        model_config_path=config.GROUNDING_DINO_CONFIG,
        model_checkpoint_path=config.GROUNDING_DINO_CHECKPOINT,
        device=device
    )
    pipe, pipe_config = init_pipline(cur_pipe, sd_guidance_scale=sd_guidance_scale)
    camera_name = str_to_list(camera_name)
    robot_infor = {'camera_names': camera_name,
                    'camera_sensors': ['rgb_images', 'depth_images'],
                    'arms': ['master', 'puppet'],
                    'controls': ['joint_position']}
    read_h5files = ReadH5Files(robot_infor)
    rgb_images, _, _ = read_h5files.execute(file_path=input_path)
    new_rgb_dict = {cam_name: [] for cam_name in camera_name}

    key_frames = [0]
    all_masks_dict = {cam_name: {} for cam_name in camera_name}
    all_masked_images_dict = {cam_name: {} for cam_name in camera_name}
    for cam_name in camera_name:
        frame_masks_by_label = {label: [] for label in selected_labels}
        frame_masked_images_by_label = {label: [] for label in selected_labels}

        get_keyframe_boxes = {}
        get_keyframe_labels = {}

        # Load the human-annotated bbox of this camera's keyframe
        base_path = config.ANNOTATION_BASE_PATH
        json_path = f"{base_path}/{task_name}/{cam_name}_keyframe_0.json"
        source_image_path = f"{base_path}/{task_name}/{cam_name}_keyframe_0.jpg"
        source_image = cv2.imread(source_image_path)
        source_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
        input_boxes, labels = load_annotations_from_json(json_path, selected_labels=selected_labels)

        src_images = crop_image(source_image, input_boxes)

        rgb_image = rgb_images[cam_name]
        frame_names = rgb_image.tolist()

        # Temporary directory to store video frames for SAM2
        with tempfile.TemporaryDirectory() as temp_dir:
            tar_image_source, tar_image = load_np_image(rgb_image[0])
            tar_image_pil = Image.fromarray(tar_image_source)

            # Detect candidate boxes with GroundingDINO, then match them to the annotated bbox via DINOv2
            boxes, _, _ = predict(model=grounding_model, image=tar_image, caption=args.text_prompt, box_threshold=0.15, text_threshold=0.15,)
            h, w, _ = tar_image_source.shape
            boxes = boxes * torch.Tensor([w, h, w, h])
            gd_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
            tar_images = crop_image(tar_image_pil, gd_boxes)
            get_keyframe_boxes[0] = dinov2_matching(src_images, tar_images, labels, gd_boxes)
            get_keyframe_labels[0] = labels

            keyframe_boxes = {frame_idx: get_keyframe_boxes[frame_idx] for frame_idx in key_frames}

            combined_mask_dict = {}
            # Propagate masks with SAM2 over each video segment
            for i in range(len(key_frames)):
                curr_keyframe = key_frames[i]
                next_keyframe = key_frames[i+1] if i < len(key_frames)-1 else len(frame_names)

                segment_dir = os.path.join(temp_dir, f"segment_{i}")
                os.makedirs(segment_dir, exist_ok=True)

                for frame_idx in range(curr_keyframe, next_keyframe):
                    frame = rgb_image[frame_idx]
                    if isinstance(frame, np.ndarray):
                        frame = Image.fromarray(frame)
                    frame.save(os.path.join(segment_dir, f"{frame_idx}.jpg"))

                segment_mask = track_with_sam2(video_path=segment_dir, ann_frame_idx=0, input_boxes=keyframe_boxes[curr_keyframe], labels=labels)
                for frame_idx, mask_value in enumerate(segment_mask, start=curr_keyframe):
                    if frame_idx < next_keyframe:
                        combined_mask_dict[frame_idx] = mask_value

            print(f"Total frames with masks: {len(combined_mask_dict)}")

            mask_batch = []
            aug_indices = []
            for frame_idx in range(len(frame_names)):
                if frame_idx % stride != 0:
                    continue
                aug_indices.append(frame_idx)

                img_src = rgb_image[frame_idx]
                img_src = cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB)
                mask_value = combined_mask_dict[frame_idx]

                combined_mask = np.zeros_like(list(mask_value.values())[0], dtype=np.uint8)
                mask_shape = list(mask_value.values())[0].shape[1:]
                table_label_ids = []
                robot_arm_label_ids = []
                found_labels = set()

                # First pass: collect table / robot arm label ids
                for label_id, obj_mask in mask_value.items():
                    original_idx = label_id - 1
                    if original_idx < len(get_keyframe_labels[key_frames[0]]):
                        original_label = get_keyframe_labels[key_frames[0]][original_idx]
                        if original_label in ["table","floor"]:
                            table_label_ids.append(label_id)
                        elif original_label in ["left robot arm", "right robot arm", "robot arm"]:
                            robot_arm_label_ids.append(label_id)
                        if original_label in selected_labels:
                            found_labels.add(original_label)

                # Second pass: build per-label masks, excluding robot arm from the table mask
                for label_id, obj_mask in mask_value.items():
                    original_idx = label_id - 1
                    if original_idx < len(get_keyframe_labels[key_frames[0]]):
                        original_label = get_keyframe_labels[key_frames[0]][original_idx]
                        if original_label in selected_labels:
                            obj_mask = obj_mask.squeeze(0)

                            if original_label in ["table","floor"]:
                                cleaned_mask = obj_mask.copy()
                                for robot_arm_id in robot_arm_label_ids:
                                    robot_arm_mask = mask_value[robot_arm_id].squeeze(0)
                                    cleaned_mask[robot_arm_mask] = False
                                obj_mask = cleaned_mask

                            frame_masks_by_label[original_label].append(obj_mask)

                            masked_rgb = create_masked_rgb(img_src, obj_mask)
                            frame_masked_images_by_label[original_label].append(masked_rgb)

                # Fill zeros for labels not detected in this frame
                for label in selected_labels:
                    if label not in found_labels:
                        zero_mask = np.zeros(mask_shape, dtype=bool)
                        frame_masks_by_label[label].append(zero_mask)
                        zero_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
                        frame_masked_images_by_label[label].append(zero_rgb)

                # Only synthesize the table region background, exclude the robot arm region
                for label_id in table_label_ids:
                    combined_mask[mask_value[label_id]] = 255
                for label_id in robot_arm_label_ids:
                    combined_mask[mask_value[label_id]] = 0

                masks = combined_mask.squeeze(0)  # sam2
                mask_batch.append(masks)

            # Full RGB stack (non-augmented frames are kept as-is)
            full_rgb_images = rgb_image[..., ::-1].copy()
            images_to_aug = [full_rgb_images[i] for i in aug_indices]

            if len(images_to_aug) > 0:
                aug_images = background_synthesis_process_batch(mask_batch, images_to_aug, bgr=bgr)
            else:
                aug_images = []

            check_aug_img = Image.fromarray(aug_images[0])
            check_aug_img.save(os.path.join(output_path, 'aug.jpg'))

            # Merge augmented frames with original frames (unify all frames to 640x480)
            aug_iter = iter(aug_images)
            for i in range(len(frame_names)):
                if i % stride == 0:
                    try:
                        new_rgb_dict[cam_name].append(next(aug_iter))
                    except StopIteration:
                        new_rgb_dict[cam_name].append(cv2.resize(full_rgb_images[i], (640, 480)))
                else:
                    new_rgb_dict[cam_name].append(cv2.resize(full_rgb_images[i], (640, 480)))

        # Stack mask / masked-image arrays and write them back
        for label in selected_labels:
            if frame_masks_by_label[label]:
                resized_masks = []
                resized_masked_images = []
            for mask, masked_img in zip(frame_masks_by_label[label], frame_masked_images_by_label[label]):
                resized_mask = cv2.resize(mask.astype(np.uint8), (640, 480), interpolation=cv2.INTER_NEAREST)
                resized_masks.append(resized_mask.astype(bool))

                resized_masked_img = cv2.resize(masked_img, (640, 480), interpolation=cv2.INTER_LINEAR)
                resized_masked_images.append(resized_masked_img)

            masks_array = np.stack(resized_masks, axis=0)
            masked_images_array = np.stack(resized_masked_images, axis=0)

            all_masks_dict[cam_name][label] = masks_array
            all_masked_images_dict[cam_name][label] = masked_images_array
    read_h5files.process_aug_images_matrix(file_path=input_path, new_rgb=new_rgb_dict, output_dir=output_path)
    read_h5files.process_masks_and_masked_images(masks_dict=all_masks_dict, masked_images_dict=all_masked_images_dict, output_dir=output_path)
    del all_masks_dict
    del new_rgb_dict



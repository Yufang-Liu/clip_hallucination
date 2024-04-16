import argparse
import json
import os

import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPTokenizer, CLIPModel, CLIPImageProcessor
from torchvision.transforms import InterpolationMode, Resize, CenterCrop, ConvertImageDtype, Normalize
from torchvision.transforms.functional import pil_to_tensor

class Transform(torch.nn.Module):
    def __init__(self, image_size, mean, std):
        super().__init__()
        self.transforms = torch.nn.Sequential(
            Resize([image_size], interpolation=InterpolationMode.BICUBIC, antialias=True),
            CenterCrop(image_size),
            ConvertImageDtype(torch.float),
            Normalize(mean, std),
        )

    def forward(self, x) -> torch.Tensor:
        """`x` should be an instance of `PIL.Image.Image`"""
        with torch.no_grad():
            x = self.transforms(x)
        return x



def load_model(args, device):
    model = CLIPModel.from_pretrained(args.model_name).to(device)
    tokenizer = CLIPTokenizer.from_pretrained(args.model_name)
    image_preprocess = CLIPImageProcessor.from_pretrained(args.model_name)
    image_transformations = Transform(
        model.config.vision_config.image_size, image_preprocess.image_mean, image_preprocess.image_std
    )
    transform = torch.jit.script(image_transformations)
    model.eval()
    return model, tokenizer, transform


@torch.no_grad()
def text_retrieval(pos_text, neg_text, image, model, tokenizer, transform, device):
    image = transform(pil_to_tensor(image).unsqueeze(0).to(device))
    caption = [pos_text, neg_text]
    caption_embeddings = [tokenizer(c, max_length=77,
                                    padding="max_length", truncation=True) for c in caption]
    input_ids = torch.tensor([c["input_ids"] for c in caption_embeddings], device=device)
    attention_mask = torch.tensor([c["attention_mask"] for c in caption_embeddings], device=device)
    image_features = model.visual_projection(model.vision_model(image)[1])
    text_features = model.text_projection(model.text_model(input_ids=input_ids,
                                                           attention_mask=attention_mask)[1])
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    logits_per_image = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    return 1 if logits_per_image.softmax(dim=1).cpu()[0][0] > 0.5 else 0


def evaluate(image_root, dataset, model, tokenizer, transform, device):
    metrics = {}
    for c, data_dict in dataset.items():
        correct_cnt = 0
        for i, data in tqdm(data_dict.items(), desc=f'evaluating {c}'):
            image_path = os.path.join(image_root, data['filename'])
            image = Image.open(image_path).convert('RGB')
            correct = text_retrieval(data['caption'], data['negative_caption'], image, model, tokenizer, transform, device)
            correct_cnt += correct
        count = len(data_dict)
        metrics[c] = correct_cnt / count
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str, default="../../cache/model/clip-vit-base-patch32", help="Model checkpoint name to use from OpenCLIP")
    parser.add_argument('--output', type=str, default='../../output/clip_results/',
                        help="Directory to where results are saved")

    parser.add_argument('--coco_image_root', type=str, default='../data/coco/val2017')
    parser.add_argument('--data_root', type=str, default='./data')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_dict = {
        'add_obj'    : f'{args.data_root}/add_obj.json',
        'add_att'    : f'{args.data_root}/add_att.json',
        'replace_obj': f'{args.data_root}/replace_obj.json',
        'replace_att': f'{args.data_root}/replace_att.json',
        'replace_rel': f'{args.data_root}/replace_rel.json',
        'swap_obj'   : f'{args.data_root}/swap_obj.json',
        'swap_att'   : f'{args.data_root}/swap_att.json',
    }
    dataset = {}
    for c, data_path in data_dict.items():
        dataset[c] = json.load(open(data_path, 'r', encoding='utf-8'))

    os.makedirs(args.output, exist_ok=True)
    print(f"Evaluating {args.model_name}")

    model, tokenizer, transform = load_model(args, device)
    metrics = evaluate(args.coco_image_root, dataset, model, tokenizer, transform, device)
    print(metrics)
    print("avg_add", (metrics['add_obj'] + metrics['add_att']) / 2)
    print("avg_replace", (metrics['replace_obj'] + metrics['replace_att'] + metrics['replace_rel']) / 3)
    print("avg_swap", (metrics['swap_obj'] + metrics['swap_att']) / 2)
    #print(f"Dump results to: {os.path.join(args.output, f'{args.model_name.split()}.json')}")
    #json.dump(metrics, open(os.path.join(args.output, f'{args.model_name}.json'), 'w'), indent=4)

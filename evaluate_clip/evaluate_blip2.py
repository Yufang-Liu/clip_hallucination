from PIL import Image
from tqdm import tqdm
import json, os
import torch
from lavis.models import load_model_and_preprocess
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="coco", type=str)

    args = parser.parse_args()

    repo_id = "blip2_image_text_matching"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, vis_processors, text_processors = load_model_and_preprocess(repo_id,
                                                                       "pretrain",
                                                                       is_eval=True,
                                                                       device=device)
    cnt = 0
    itm_correct = 0
    itc_correct = 0
    if args.dataset == "coco":
        input_file_path = '../data/OHD-Caps/test/coco_out_label.json'
    elif args.dataset == "flickr":
        input_file_path = '../data/OHD-Caps/test/flickr_out_label.json'
    elif args.dataset == 'nocaps':
        input_file_path = '../data/OHD-Caps/test/nocaps_out_label.json'
    else:
        raise ValueError("Invalid dataset name")

    print("load dataset from {}".format(input_file_path))

    with open(input_file_path, 'r') as f:
        for idx, line in enumerate(tqdm(f)):
            data = json.loads(line)
            image = Image.open(data["image"]).convert("RGB")
            caption, label = data['caption'], data['label']

            img = vis_processors["eval"](image).unsqueeze(0).to(device)
            itm_score_list = []
            itc_score_list = []
            for cap in caption:
                txt = text_processors["eval"](cap)
                itm_output = model({"image": img, "text_input": txt}, match_head="itm")
                itm_scores = torch.nn.functional.softmax(itm_output, dim=1)[:, 1].item()
                itm_score_list.append(itm_scores)

                itc_score = model({"image": img, "text_input": txt}, match_head='itc')
                itc_score_list.append(itc_score.item())
            itm_predict = itm_score_list.index(max(itm_score_list))
            itc_predict = itc_score_list.index(max(itc_score_list))


            if itm_predict == label:
                itm_correct += 1
            if itc_predict == label:
                itc_correct += 1
            cnt += 1
    print("itm acc is {}".format(itm_correct * 1.0 / cnt))
    print("itc acc is {}".format(itc_correct * 1.0 / cnt))


if __name__ == '__main__':
    main()

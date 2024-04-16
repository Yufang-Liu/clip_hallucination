import argparse
import json
import os
from llava_utils import generate_output

pope_root = '../data/POPE'
coco_img_root = '../data/coco_val2014/val2014'
flickr_img_root = '../data/flickr30k/flickr30k-images'
nocaps_img_root = '../data/nocaps/data'


def calc_accuracy(ans_file):

    answers = [json.loads(q) for q in open(ans_file, 'r')]
    label_list = [json.loads(q)['label'] for q in open(ans_file, 'r')]

    for answer in answers:
        text = answer['answer']

        # Only keep the first sentence
        if text.find('.') != -1:
            text = text.split('.')[0]

        text = text.replace(',', '')
        words = text.split(' ')
        if 'No' in words or 'not' in words or 'no' in words:
            answer['answer'] = 'no'
        else:
            answer['answer'] = 'yes'

    for i in range(len(label_list)):
        if label_list[i] == 'no':
            label_list[i] = 0
        else:
            label_list[i] = 1

    pred_list = []
    for answer in answers:
        if answer['answer'] == 'no':
            pred_list.append(0)
        else:
            pred_list.append(1)

    pos = 1
    neg = 0
    yes_ratio = pred_list.count(1) / len(pred_list)

    TP, TN, FP, FN = 0, 0, 0, 0
    for pred, label in zip(pred_list, label_list):
        if pred == pos and label == pos:
            TP += 1
        elif pred == pos and label == neg:
            FP += 1
        elif pred == neg and label == neg:
            TN += 1
        elif pred == neg and label == pos:
            FN += 1

    print('TP\tFP\tTN\tFN\t')
    print('{}\t{}\t{}\t{}'.format(TP, FP, TN, FN))

    precision = float(TP) / float(TP + FP)
    recall = float(TP) / float(TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    acc = (TP + TN) / (TP + TN + FP + FN)
    print('Accuracy: {}'.format(acc))
    print('Precision: {}'.format(precision))
    print('Recall: {}'.format(recall))
    print('F1 score: {}'.format(f1))
    print('Yes ratio: {}'.format(yes_ratio))


def get_dataset(dataset_name, dataset_type):
    if dataset_name == "coco_pope":
        file_path = pope_root + '/coco_pope_{}.json'.format(dataset_type)
        image_dir = coco_img_root
    elif dataset_name == "flickr_pope":
        file_path = pope_root + '/flickr_pope_{}.json'.format(dataset_type)
        image_dir = flickr_img_root
    elif dataset_name == "nocaps_pope":
        file_path = pope_root + '/nocaps_pope_{}.json'.format(dataset_type)
        image_dir = nocaps_img_root
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    print("Loading data from input_file {}".format(file_path))
    all_data = []
    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            data['image'] = os.path.join(os.path.abspath(image_dir), data['image'])
            all_data.append(data)
    return all_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="../cache/model/llava-v1.5-7b",
                        help="The path to the model")
    parser.add_argument("--type", default="popular", type=str,
                        choices=["popular", "adversarial", "random", "attribute", "existence", "relation"],
                        help="The type of the dataset")
    parser.add_argument('--dataset', type=str,
                        choices=['coco_pope', 'flickr_pope', 'nocaps_pope', ],
                        default='coco_pope', help='The dataset to evaluate on')
    parser.add_argument("--output_dir", type=str, default='../output/llava_results/',
                        help="The path to the output file")
    args = parser.parse_args()

    all_data = get_dataset(args.dataset, args.type)

    output_file = args.output_dir + '{}_{}_{}.json'.format(args.model_path.split('/')[-1],
                                                           args.dataset, args.type)
    generate_output(args.model_path, all_data, output_file)

    print("evaluating the accuracy of the output file {}".format(output_file))
    calc_accuracy(output_file)


if __name__ == '__main__':
    main()
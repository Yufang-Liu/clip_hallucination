import json
import openai
from openai import OpenAI
import os
from tqdm import tqdm
from time import sleep
import argparse
import backoff

os.environ["OPENAI_API_KEY"] = "xxx"
client = OpenAI()

@backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError))
def completions_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs)


def rewrite_text_by_add(text, objects):
    text_prompt = ("Given a sentence '{}', generate a new sentence and "
                   "includes each object from the list '{}'. "
                   "Make the changes to the original sentence as minimal as possible. "
                   "Ensure that the new sentence is coherent, natural, semantically smooth "
                   "and free of grammatical errors. ".format(text, objects))

    try :
        completion = completions_with_backoff(
            model="gpt-4-0125-preview",
            messages=[
                {"role": "user",
                 "content": text_prompt}
            ]
        )
        return completion.choices[0].message.content
    except Exception as e:
        print("Error : {}".format(e))
        sleep(5)
        rewrite_text_by_add(text, objects)


def rewrite_text_by_delete(text, objects):
    object_list = objects.split(', ')
    if all(obj in text for obj in object_list):
        text_prompt = ("Given a sentence '{}', generate a new sentence and remove each object from list '{}' "
                       "to make the semantics of the sentence different. "
                       "Ensure that the new sentence is coherent, natural, semantically smooth and free of grammatical errors."
                       .format(text, objects))
    else:
        text_prompt = ("Given a sentence '{}', choose to modify the objects, colors, attributes, etc., "
                       "within the sentence to make the semantics of the sentence different. "
                       "Make the changes to the original sentence as minimal as possible. "
                       "Ensure that the new sentence is coherent, natural, semantically smooth and free of grammatical errors."
                       .format(text, objects))

    try:
        completion = completions_with_backoff(
            model="gpt-4-0125-preview",
            messages=[
                {"role": "user",
                 "content": text_prompt}
            ]
        )
        return completion.choices[0].message.content
    except Exception as e:
        print("Error : {}".format(e))
        sleep(5)
        rewrite_text_by_delete(text, objects)

def get_object_string(object_list, add=True):
    if add: assert len(object_list) == 3
    if not add and len(object_list) <= 1:
        return []
    string_list = []
    for obj in object_list:
        string_list.append(obj)
    if not add and len(string_list) < 3:
        return string_list
    string_list.append(', '.join(object_list[:2]))
    string_list.append(', '.join(object_list[1:]))
    string_list.append(', '.join([object_list[0], object_list[2]]))
    if add:
        string_list.append(', '.join(object_list))
    return string_list



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str,
                        default="images.json",
                        help="The path to the input json file")
    parser.add_argument("--output_file", type=str,
                        default="../data/coco_train2017/coco_annotations_8k.json",
                        help="The path to the output json file")
    args = parser.parse_args()

    input_file = args.input_file
    output_file = args.output_file
    fwrite = open(output_file, 'w')
    with open(input_file, 'r') as f:
        for idx, line in enumerate(tqdm(f)):
            line_dict = json.loads(line)

            if 'adversarial_samples' not in line_dict:
                line_dict['adversarial'] = [obj.strip() for obj in line_dict['adversarial']]
                line_dict['popular'] = [obj.strip() for obj in line_dict['popular']]
                line_dict['random'] = [obj.strip() for obj in line_dict['random']]

                true_adversarial_objects = [obj for obj in line_dict['adversarial'] if obj not in line_dict['ground_truth']]
                true_popular_objects = [obj for obj in line_dict['popular'] if obj not in line_dict['ground_truth']]
                true_random_objects = [obj for obj in line_dict['random'] if obj not in line_dict['ground_truth']]

                adversarial_samples = {}
                for obj_str in get_object_string(true_adversarial_objects[:3]):
                    sample = rewrite_text_by_add(line_dict['positive_sample'], obj_str)
                    adversarial_samples[obj_str] = sample
                line_dict['adversarial_samples'] = adversarial_samples

                popular_objects = {}
                for obj_str in get_object_string(true_popular_objects[:3]):
                    sample = rewrite_text_by_add(line_dict['positive_sample'], obj_str)
                    popular_objects[obj_str] = sample
                line_dict['popular_samples'] = popular_objects
    
                random_objects = {}
                for obj_str in get_object_string(true_random_objects[:3]):
                    sample = rewrite_text_by_add(line_dict['positive_sample'], obj_str)
                    random_objects[obj_str] = sample
                line_dict['random_samples'] = random_objects

                delete_samples = {}
                for obj_str in get_object_string(line_dict['ground_truth'][:3], add=False):
                    sample = rewrite_text_by_delete(line_dict['positive_sample'], obj_str)
                    delete_samples[obj_str] = sample
                line_dict['delete_samples'] = delete_samples

            json.dump(line_dict, fwrite)
            fwrite.write('\n')

    fwrite.close()
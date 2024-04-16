import os
import torch
import clip
from PIL import Image
from torchvision import transforms
from .constants import CACHE_DIR


def get_model(model_name, device, root_dir=CACHE_DIR):
    """
    Helper function that returns a model and a potential image preprocessing function.
    """
    if "openai-clip" in model_name:
        from .clip_models import CLIPWrapper
        variant = model_name.split(":")[1]
        model, image_preprocess = clip.load(variant, device=device, download_root=root_dir)
        model = model.eval()
        clip_model = CLIPWrapper(model, device) 
        return clip_model, image_preprocess

    elif 'hf-clip' in model_name:
        from transformers import CLIPModel, CLIPTokenizer, CLIPImageProcessor
        from .hf_clip_models import CLIPWrapper
        path = model_name.split(":")[1]
        model = CLIPModel.from_pretrained(path).to(device).eval()
        tokenizer = CLIPTokenizer.from_pretrained(path)
        feature_extractor = CLIPImageProcessor.from_pretrained(path)
        clip_model = CLIPWrapper(model, tokenizer, feature_extractor, device)
        return clip_model, None

    elif 'spatial_ft' in model_name:
        from .clip_models import CLIPWrapper
        variant = 'ViT-B/32'
        model, image_preprocess = clip.load(variant, device=device, download_root=root_dir)
        print('Getting model weights from {}'.format('data/{}.pt'.format(model_name)))
        state = torch.load('data/{}.pt'.format(model_name))
        state['model_state_dict'] = {k.replace('module.clip_model.', '') : v for k, v in state['model_state_dict'].items()}
        model.load_state_dict(state['model_state_dict'])
        model = model.eval()
        clip_model = CLIPWrapper(model, device)
        return clip_model, image_preprocess

    elif model_name == "blip-flickr-base":
        from .blip_models import BLIPModelWrapper
        blip_model = BLIPModelWrapper(root_dir=root_dir, device=device, variant="blip-flickr-base")
        image_preprocess = transforms.Compose([
                        transforms.Resize((384, 384),interpolation=transforms.functional.InterpolationMode.BICUBIC),
                        transforms.ToTensor(),
                        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])  
        return blip_model, image_preprocess
    
    elif model_name == "blip-coco-base":
        from .blip_models import BLIPModelWrapper
        blip_model = BLIPModelWrapper(root_dir=root_dir, device=device, variant="blip-coco-base")
        image_preprocess = transforms.Compose([
                        transforms.Resize((384, 384),interpolation=transforms.functional.InterpolationMode.BICUBIC),
                        transforms.ToTensor(),
                        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])  
        return blip_model, image_preprocess
    
    elif model_name == "blip-base-14m":
        from .blip_models import BLIPModelWrapper
        blip_model = BLIPModelWrapper(root_dir=root_dir, device=device, variant="blip-base-14m")
        image_preprocess = transforms.Compose([
                        transforms.Resize((224, 224),interpolation=transforms.functional.InterpolationMode.BICUBIC),
                        transforms.ToTensor(),
                        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])  
        return blip_model, image_preprocess
    
    elif model_name == "blip-base-129m":
        from .blip_models import BLIPModelWrapper
        blip_model = BLIPModelWrapper(root_dir=root_dir, device=device, variant="blip-base-129m")
        image_preprocess = transforms.Compose([
                        transforms.Resize((224, 224),interpolation=transforms.functional.InterpolationMode.BICUBIC),
                        transforms.ToTensor(),
                        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])  
        return blip_model, image_preprocess
    
    elif model_name == "xvlm-flickr":
        from .xvlm_models import XVLMWrapper
        xvlm_model = XVLMWrapper(root_dir=root_dir, device=device, variant="xvlm-flickr")
        image_preprocess = transforms.Compose([
                            transforms.Resize((384, 384), interpolation=Image.BICUBIC),
                            transforms.ToTensor(),
                            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),])
        return xvlm_model, image_preprocess
    
    elif model_name == "xvlm-coco":
        from .xvlm_models import XVLMWrapper
        xvlm_model = XVLMWrapper(root_dir=root_dir, device=device, variant="xvlm-coco")
        image_preprocess = transforms.Compose([
                            transforms.Resize((384, 384), interpolation=Image.BICUBIC),
                            transforms.ToTensor(),
                            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),])
        return xvlm_model, image_preprocess
    
    elif model_name == "xvlm-pretrained-16m":
        from .xvlm_models import XVLMWrapper
        xvlm_model = XVLMWrapper(root_dir=root_dir, device=device, variant="xvlm-pretrained-16m")
        image_preprocess = transforms.Compose([
                            transforms.Resize((224, 224), interpolation=Image.BICUBIC),
                            transforms.ToTensor(),
                            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),])
        return xvlm_model, image_preprocess
    
    elif model_name == "xvlm-pretrained-4m":
        from .xvlm_models import XVLMWrapper
        xvlm_model = XVLMWrapper(root_dir=root_dir, device=device, variant="xvlm-pretrained-4m")
        image_preprocess = transforms.Compose([
                            transforms.Resize((224, 224), interpolation=Image.BICUBIC),
                            transforms.ToTensor(),
                            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),])
        return xvlm_model, image_preprocess
    
    elif model_name == "flava":
        from .flava import FlavaWrapper
        flava_model = FlavaWrapper(root_dir=root_dir, device=device)
        image_preprocess = None
        return flava_model, image_preprocess

    elif model_name == "NegCLIP":
        import open_clip
        from .clip_models import CLIPWrapper
        
        path = os.path.join(root_dir, "negclip.pth")
        if not os.path.exists(path):
            print("Downloading the NegCLIP model...")
            import gdown
            gdown.download(id="1ooVVPxB-tvptgmHlIMMFGV3Cg-IrhbRZ", output=path, quiet=False)
        model, _, image_preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained=path, device=device, cache_dir=root_dir)
        model = model.eval()
        clip_model = CLIPWrapper(model, device) 
        return clip_model, image_preprocess

    
    elif 'negclip' in model_name:
        import open_clip
        from .clip_models import CLIPWrapper
        
        path = os.path.join(root_dir, "../cache/clip/{}.pt".format(model_name))
        print("Loading a negclip model from {}".format(path))
        model, _, image_preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained=path, device=device, cache_dir=root_dir)
        model = model.eval()
        clip_model = CLIPWrapper(model, device) 
        return clip_model, image_preprocess

    elif 'open_clip/' in model_name or 'neg_clip/' in model_name or 'neg_clip_v2/' in model_name:
        import open_clip
        from .clip_models import CLIPWrapper
        path = model_name
        print("Loading an openclip model from {}".format(path))
        model, _, image_preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained=path, device=device, cache_dir=root_dir)
        model = model.eval()
        clip_model = CLIPWrapper(model, device) 
        return clip_model, image_preprocess

    elif model_name == "CE-CLIP":
        import open_clip
        from .clip_models import CLIPWrapper

        path = os.path.join(root_dir, "ceclip.pt")
        if not os.path.exists(path):
            print("Downloading the CE-CLIP model...")
            import gdown
            gdown.download(id="1DWPw3CtGh5cHz9bW_-iXRSG7BBUVl13K", output=path, quiet=False)
        model, _, image_preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained=path, device=device, cache_dir=root_dir)
        model = model.eval()
        clip_model = CLIPWrapper(model, device)
        return clip_model, image_preprocess

    elif model_name == "coca":
        import open_clip
        from .clip_models import CLIPWrapper
        model, _, image_preprocess = open_clip.create_model_and_transforms(
            model_name="coca_ViT-B-32", pretrained="laion2B-s13B-b90k", device=device, cache_dir=root_dir)
        model = model.eval()
        clip_model = CLIPWrapper(model, device) 
        return clip_model, image_preprocess
    
    elif model_name == "coca-cap":
        import open_clip
        from .clip_models import CLIPWrapper
        model, _, image_preprocess = open_clip.create_model_and_transforms(
            model_name="coca_ViT-B-32", pretrained="mscoco_finetuned_laion2b_s13b_b90k",
            device=device, cache_dir=root_dir)
        model = model.eval()
        clip_model = CLIPWrapper(model, device)
        return clip_model, image_preprocess
        
    elif "laion-clip" in model_name:
        import open_clip
        from .clip_models import CLIPWrapper
        variant = model_name.split(":")[1]
        if variant == 'ViT-B/32':
            model, _, image_preprocess = open_clip.create_model_and_transforms(
                model_name=variant, pretrained="laion2b_s34b_b79k", device=device, cache_dir=root_dir)
        elif variant == 'ViT-L/14':
            model, _, image_preprocess = open_clip.create_model_and_transforms(
                model_name=variant, pretrained="laion2b_s32b_b82k", device=device, cache_dir=root_dir)
        elif variant == 'ViT-H/14':
            model, _, image_preprocess = open_clip.create_model_and_transforms(
                model_name=variant, pretrained="laion2b_s32b_b79k", device=device, cache_dir=root_dir)
        elif variant == 'roberta-ViT-B/32':
            model, _, image_preprocess = open_clip.create_model_and_transforms(
                model_name=variant, pretrained="laion2b_s12b_b32k", device=device, cache_dir=root_dir)
        else:
            print("Didn't recognize the LAION model type, sorry!")
            exit()
        model = model.eval()
        clip_model = CLIPWrapper(model, device) 
        return clip_model, image_preprocess
    
        
    else:
        raise ValueError(f"Unknown model {model_name}")

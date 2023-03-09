print('Importing')
import PIL
import requests
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
print("Starting Pipeline")
model_id = "timbrooks/instruct-pix2pix"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
print('Piping to MPS')
pipe.to("mps")
print('Scheduling')
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
print("Downloading Image")
url = input("URL: > ")
def download_image(url):
    image = PIL.Image.open(requests.get(url, stream=True).raw)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image
image = download_image(url)
#image = PIL.Image.open("orig.jpg")
#image = PIL.ImageOps.exif_transpose(image)
#image = image.convert("RGB")
print("Generating")
prompt = input("Prompt: > ")
images = pipe(prompt, image=image, num_inference_steps=10, image_guidance_scale=1).images
print('Done')
images[0].save('result.png')
print('Saved')

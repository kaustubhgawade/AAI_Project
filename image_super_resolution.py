import requests
from PIL import Image
from io import BytesIO
from diffusers import LDMSuperResolutionPipeline
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "CompVis/ldm-super-resolution-4x-openimages"

pipeline = LDMSuperResolutionPipeline.from_pretrained(model_id)
pipeline = pipeline.to(device)

url = "https://user-images.githubusercontent.com/38061659/199705896-b48e17b8-b231-47cd-a270-4ffa5a93fa3e.png"
# url = "https://imgs.search.brave.com/_ObgR9OZ0FAG3YLuhdd3Ml4jnI6YfthTmqm31xIltVw/rs:fit:860:0:0/g:ce/aHR0cHM6Ly9pbWFn/ZW9ubGluZS5jby9p/bWFnZW9ubGluZS1p/bWFnZS5qcGc"
response = requests.get(url)
low_res_img = Image.open(BytesIO(response.content)).convert("RGB")
low_res_img = low_res_img.resize((128, 128))

upscaled_image = pipeline(low_res_img, num_inference_steps=100, eta=1).images[0]

upscaled_image.save("ldm_generated_image.png")

import requests
from PIL import Image
import io
import numpy as np
import torch
import os
import configparser
import time
import base64
from enum import Enum
from urllib.parse import urljoin

class Status(Enum):
    PENDING = "Pending"
    READY = "Ready"
    ERROR = "Error"

class ConfigLoader:
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config.ini")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}. Please ensure config.ini exists in the same directory as the script.")
            
        self.config = configparser.ConfigParser()
        self.config.read(config_path)

class FluxPro11WithFinetune:
    RETURN_TYPES = ("IMAGE", "STRING")
    FUNCTION = "process"
    CATEGORY = "BFL"

    def __init__(self):
        try:
            self.config_loader = ConfigLoader()
        except Exception as e:
            print(f"[FLUX API] Initialization Error: {str(e)}")
            print("[FLUX API] Please ensure config.ini is properly set up with API credentials")
            raise

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["generate", "finetune", "inference"], {"default": "generate"}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "ultra_mode": ("BOOLEAN", {"default": True}),
                "aspect_ratio": ([
                    "21:9", "16:9", "4:3", "1:1", "3:4", "9:16", "9:21"
                ], {"default": "16:9"}),
                "safety_tolerance": ("INT", {"default": 6, "min": 0, "max": 6}),
                "output_format": (["jpeg", "png"], {"default": "png"}),
                "raw": ("BOOLEAN", {"default": False}),
                "x_key": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                "seed": ("INT", {"default": -1}),
                "finetune_zip": ("STRING", {"default": "", "multiline": False}),
                "finetune_comment": ("STRING", {"default": ""}),
                "finetune_id": ("STRING", {"default": ""}),
                "trigger_word": ("STRING", {"default": "TOK"}),
                "finetune_mode": (["character", "product", "style", "general"], {"default": "general"}),
                "iterations": ("INT", {"default": 300, "min": 100}),
                "learning_rate": ("FLOAT", {"default": 0.00001, "min": 0.00001, "max": 0.0001, "step": 0.00001}),
                "captioning": ("BOOLEAN", {"default": True}),
                "priority": (["speed", "quality"], {"default": "quality"}),
                "finetune_type": (["full", "lora"], {"default": "full"}),
                "lora_rank": ("INT", {"default": 32}),
                "finetune_strength": ("FLOAT", {"default": 1.2})
            }
        }

    def process(self, mode, x_key, **kwargs):
        try:
            print(f"[FLUX API] Processing in {mode} mode")
            if mode == "generate":
                img = self.generate_image(**kwargs, x_key=x_key)
                return (*img, "")
            elif mode == "finetune":
                return self.request_finetuning(**kwargs, x_key=x_key)
            elif mode == "inference":
                return self.finetune_inference(**kwargs, x_key=x_key)
            else:
                print(f"[FLUX API] Error: Unknown mode {mode}")
                return (*self.create_blank_image(), "")
        except Exception as e:
            print(f"[FLUX API] Process Error: {str(e)}")
            return (*self.create_blank_image(), "")

    def generate_image(self, prompt, ultra_mode, aspect_ratio, 
                      safety_tolerance, output_format, raw, x_key, seed=-1, **kwargs):
        if not prompt:
            print("[FLUX API] Error: Prompt cannot be empty")
            return self.create_blank_image()

        try:
            if ultra_mode:
                arguments = {
                    "prompt": prompt,
                    "aspect_ratio": aspect_ratio,
                    "safety_tolerance": safety_tolerance,
                    "output_format": output_format,
                    "raw": raw
                }
                if seed != -1:
                    arguments["seed"] = seed
                    
                url = "https://api.bfl.ai/v1/flux-pro-1.1-ultra"
            else:
                width, height = self.get_dimensions_from_ratio(aspect_ratio)
                arguments = {
                    "prompt": prompt,
                    "width": width,
                    "height": height,
                    "safety_tolerance": safety_tolerance,
                    "output_format": output_format
                }
                if seed != -1:
                    arguments["seed"] = seed
                    
                url = "https://api.bfl.ai/v1/flux-pro-1.1"

            headers = {"x-key": x_key}
            
            print(f"[FLUX API] Sending request to: {url}")
            print(f"[FLUX API] Arguments: {arguments}")
            
            response = requests.post(url, json=arguments, headers=headers, timeout=30)
            print(f"[FLUX API] Response Status: {response.status_code}")
            
            if response.status_code == 200:
                response_data = response.json()
                if not response_data:
                    print("[FLUX API] Error: Empty response received from server")
                    return self.create_blank_image()
                    
                task_id = response_data.get("id")
                if not task_id:
                    print("[FLUX API] Error: No task ID received in response")
                    print(f"[FLUX API] Response data: {response_data}")
                    return self.create_blank_image()
                    
                print(f"[FLUX API] Task ID received: {task_id}")
                return self.get_result(task_id, output_format, x_key)
            else:
                print(f"[FLUX API] Server Error: {response.status_code}")
                print(f"[FLUX API] Response: {response.text}")
                return self.create_blank_image()
                
        except requests.exceptions.RequestException as e:
            print(f"[FLUX API] Network Error: {str(e)}")
            return self.create_blank_image()
        except Exception as e:
            print(f"[FLUX API] Unexpected Error: {str(e)}")
            print(f"[FLUX API] Error Type: {type(e).__name__}")
            return self.create_blank_image()

    def request_finetuning(self, finetune_zip, finetune_comment, trigger_word="TOK",
                          finetune_mode="general", iterations=300, learning_rate=0.00001,
                          captioning=True, priority="quality", finetune_type="full", 
                          lora_rank=32, x_key='', **kwargs):
        try:
            print("[FLUX API] Starting finetuning process")
            if not finetune_comment:
                print("[FLUX API] Error: finetune_comment is required")
                return (*self.create_blank_image(), "")

            if not os.path.exists(finetune_zip):
                print(f"[FLUX API] Error: ZIP file not found at {finetune_zip}")
                return (*self.create_blank_image(), "")

            with open(finetune_zip, "rb") as file:
                encoded_zip = base64.b64encode(file.read()).decode("utf-8")

            url = "https://api.bfl.ai/v1/finetune"
            headers = {
                "Content-Type": "application/json",
                "X-Key": x_key,
            }
            
            payload = {
                "finetune_comment": finetune_comment,
                "trigger_word": trigger_word,
                "file_data": encoded_zip,
                "iterations": iterations,
                "mode": finetune_mode,
                "learning_rate": learning_rate,
                "captioning": captioning,
                "priority": priority,
                "lora_rank": lora_rank,
                "finetune_type": finetune_type,
            }

            print(f"[FLUX API] Sending finetuning request to {url}")
            response = requests.post(url, headers=headers, json=payload)
            print(f"[FLUX API] Response status: {response.status_code}")
            print(f"[FLUX API] Response text: {response.text}")
            
            response.raise_for_status()
            result = response.json()
            
            finetune_id = result.get("finetune_id", "")
            print(f"[FLUX API] Finetuning initiated. ID: {finetune_id}")
            return (*self.create_blank_image(), finetune_id)

        except Exception as e:
            print(f"[FLUX API] Finetuning Error: {str(e)}")
            print(f"[FLUX API] Error Type: {type(e).__name__}")
            return (*self.create_blank_image(), "")

    def finetune_inference(self, finetune_id, prompt, ultra_mode=True, 
                         finetune_strength=1.2, x_key='', **kwargs):
       try:
           print(f"[FLUX API] Starting inference with finetune_id: {finetune_id}")
           endpoint = "flux-pro-1.1-ultra-finetuned" if ultra_mode else "flux-pro-finetuned"
           url = f"https://api.bfl.ai/v1/{endpoint}"
           
           headers = {
               "Content-Type": "application/json",
               "X-Key": x_key,
           }
           
           payload = {
               "finetune_id": finetune_id,
               "finetune_strength": finetune_strength,
               "prompt": prompt,
               **kwargs
           }

           print(f"[FLUX API] Sending inference request to {url}")
           print(f"[FLUX API] Payload: {payload}")
           response = requests.post(url, headers=headers, json=payload)
           print(f"[FLUX API] Response Status: {response.status_code}")
           print(f"[FLUX API] Response Text: {response.text}")
           
           response.raise_for_status()
           result = response.json()
           
           task_id = result.get("id")
           if not task_id:
               print("[FLUX API] Error: No task ID received for inference")
               return (*self.create_blank_image(), finetune_id)
               
           print(f"[FLUX API] Inference task ID: {task_id}")
           return (*self.get_result(task_id, kwargs.get("output_format", "png"), x_key), finetune_id)

       except Exception as e:
           print(f"[FLUX API] Inference Error: {str(e)}")
           print(f"[FLUX API] Error Type: {type(e).__name__}")
           return (*self.create_blank_image(), finetune_id)

    def get_dimensions_from_ratio(self, aspect_ratio):
        regular_dimensions = {
            "1:1":  (1024, 1024),
            "4:3":  (1408, 1024),
            "3:4":  (1024, 1408),
            "16:9": (1408, 800),
            "9:16": (800, 1408),
            "21:9": (1408, 608),
            "9:21": (608, 1408)
        }
        return regular_dimensions.get(aspect_ratio, (1408, 800))

    def create_blank_image(self):
        blank_img = Image.new('RGB', (512, 512), color='black')
        img_array = np.array(blank_img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array)[None,]
        return (img_tensor,)

    def get_result(self, task_id, output_format, x_key, attempt=1, max_attempts=15):
       if attempt > max_attempts:
           print(f"[FLUX API] Max attempts reached for task_id {task_id}")
           return self.create_blank_image()

       try:
           get_url = f"https://api.bfl.ai/v1/get_result?id={task_id}"
           headers = {"x-key": x_key}
           
           wait_time = min(2 ** attempt + 5, 30)
           print(f"[FLUX API] Waiting {wait_time} seconds before attempt {attempt}")
           time.sleep(wait_time)
           
           print(f"[FLUX API] Attempt {attempt}: Checking result for task {task_id}")
           response = requests.get(get_url, headers=headers, timeout=30)
           print(f"[FLUX API] Response Status: {response.status_code}")
           
           if response.status_code == 200:
               result = response.json()
               status = result.get("status")
               print(f"[FLUX API] Task Status: {status}")
               
               if status == Status.READY.value:
                   sample_url = result.get('result', {}).get('sample')
                   if not sample_url:
                       print("[FLUX API] Error: No sample URL in response")
                       print(f"[FLUX API] Response data: {result}")
                       return self.create_blank_image()
                       
                   img_response = requests.get(sample_url, timeout=30)
                   if img_response.status_code != 200:
                       print(f"[FLUX API] Error downloading image: {img_response.status_code}")
                       return self.create_blank_image()
                   
                   img = Image.open(io.BytesIO(img_response.content))
                   
                   with io.BytesIO() as output:
                       img.save(output, format=output_format.upper())
                       output.seek(0)
                       img_converted = Image.open(output)
                       img_array = np.array(img_converted).astype(np.float32) / 255.0
                       print(f"[FLUX API] Successfully generated image for task {task_id}")
                       return (torch.from_numpy(img_array)[None,],)
                       
               elif status == Status.PENDING.value:
                   print(f"[FLUX API] Attempt {attempt}: Image not ready. Retrying...")
                   return self.get_result(task_id, output_format, x_key, attempt + 1, max_attempts)
               else:
                   print(f"[FLUX API] Unexpected status: {status}")
                   print(f"[FLUX API] Full response: {result}")
                   return self.create_blank_image()
                   
           else:
               print(f"[FLUX API] Error retrieving result: {response.status_code}")
               print(f"[FLUX API] Response: {response.text}")
               if attempt < max_attempts:
                   return self.get_result(task_id, output_format, x_key, attempt + 1, max_attempts)
               
       except Exception as e:
           print(f"[FLUX API] Error retrieving result: {str(e)}")
           print(f"[FLUX API] Error Type: {type(e).__name__}")
           if attempt < max_attempts:
               return self.get_result(task_id, output_format, x_key, attempt + 1, max_attempts)
               
       return self.create_blank_image()

NODE_CLASS_MAPPINGS = {
    "FluxPro11WithFinetune": FluxPro11WithFinetune
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxPro11WithFinetune": "Flux Pro 1.1 Ultra & Raw with Finetuning"
}
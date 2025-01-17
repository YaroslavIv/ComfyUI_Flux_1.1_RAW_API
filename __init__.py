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
        self.set_x_key()

    def get_key(self, section, key):
        if not self.config.has_section(section):
            raise KeyError(f"Section '{section}' not found in config file. Please check your config.ini")
        if not self.config.has_option(section, key):
            raise KeyError(f"Key '{key}' not found in section '{section}'. Please check your config.ini")
        return self.config[section][key]

    def set_x_key(self):
        try:
            x_key = self.get_key('API', 'X_KEY')
            if not x_key:
                raise KeyError("X_KEY cannot be empty")
            os.environ["X_KEY"] = x_key
        except KeyError as e:
            print(f"Error setting X_KEY: {str(e)}")
            print("Please ensure your config.ini contains a valid X_KEY under the [API] section")
            raise

class FluxPro11WithFinetune:
    RETURN_TYPES = ("IMAGE", "STRING")  # Added STRING return type for finetune_id
    FUNCTION = "process"
    CATEGORY = "BFL"

    def __init__(self):
        try:
            self.config_loader = ConfigLoader()
        except Exception as e:
            print(f"Initialization Error: {str(e)}")
            print("Please ensure config.ini is properly set up with API credentials")
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
                "raw": ("BOOLEAN", {"default": False})
            },
            "optional": {
                "seed": ("INT", {"default": -1}),
                # Finetuning parameters
                "finetune_zip": ("STRING", {"default": ""}),
                "finetune_comment": ("STRING", {"default": ""}),
                "finetune_id": ("STRING", {"default": ""}),
                "trigger_word": ("STRING", {"default": "TOK"}),
                "finetune_mode": (["character", "product", "style", "general"], {"default": "general"}),
                "iterations": ("INT", {"default": 300, "min": 100}),
                "learning_rate": ("FLOAT", {"default": 0.00001}),
                "captioning": ("BOOLEAN", {"default": True}),
                "priority": (["speed", "quality"], {"default": "quality"}),
                "finetune_type": (["full", "lora"], {"default": "full"}),
                "lora_rank": ("INT", {"default": 32}),
                "finetune_strength": ("FLOAT", {"default": 1.2})
            }
        }

    def process(self, mode, **kwargs):
        if mode == "generate":
            img = self.generate_image(**kwargs)
            return (*img, "")  # Return image and empty string for finetune_id
        elif mode == "finetune":
            return self.request_finetuning(**kwargs)
        elif mode == "inference":
            return self.finetune_inference(**kwargs)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def generate_image(self, prompt, ultra_mode, aspect_ratio, 
                      safety_tolerance, output_format, raw, seed=-1, **kwargs):
        if not prompt:
            print("Error: Prompt cannot be empty")
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

            x_key = os.environ.get("X_KEY")
            if not x_key:
                raise ValueError("X_KEY not found in environment variables")

            headers = {"x-key": x_key}
            response = requests.post(url, json=arguments, headers=headers, timeout=30)
            
            if response.status_code == 200:
                response_data = response.json()
                if not response_data:
                    raise ValueError("Empty response received from server")
                    
                task_id = response_data.get("id")
                if not task_id:
                    raise ValueError("No task ID received in response")
                    
                print(f"Task ID received: {task_id}")
                return self.get_result(task_id, output_format)
            else:
                print(f"Server Error: {response.status_code}")
                print(f"Response: {response.text}")
                return self.create_blank_image()
                
        except requests.exceptions.RequestException as e:
            print(f"Network Error: {str(e)}")
            return self.create_blank_image()
        except Exception as e:
            print(f"Unexpected Error: {str(e)}")
            return self.create_blank_image()

    def request_finetuning(self, finetune_zip, finetune_comment, trigger_word="TOK",
                          finetune_mode="general", iterations=300, learning_rate=0.00001,
                          captioning=True, priority="quality", finetune_type="full", 
                          lora_rank=32, **kwargs):
        try:
            if not os.path.exists(finetune_zip):
                raise FileNotFoundError(f"ZIP file not found at {finetune_zip}")

            with open(finetune_zip, "rb") as file:
                encoded_zip = base64.b64encode(file.read()).decode("utf-8")

            url = "https://api.bfl.ai/v1/finetune"
            headers = {
                "Content-Type": "application/json",
                "X-Key": os.environ["X_KEY"],
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

            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            
            # Return a blank image and the finetune_id
            return (*self.create_blank_image(), result.get("id", ""))

        except Exception as e:
            print(f"Finetuning Error: {str(e)}")
            return (*self.create_blank_image(), "")

    def finetune_inference(self, finetune_id, prompt, ultra_mode=True, 
                          finetune_strength=1.2, **kwargs):
        try:
            endpoint = "flux-pro-1.1-ultra-finetuned" if ultra_mode else "flux-pro-finetuned"
            url = f"https://api.bfl.ai/v1/{endpoint}"
            
            headers = {
                "Content-Type": "application/json",
                "X-Key": os.environ["X_KEY"],
            }
            
            payload = {
                "finetune_id": finetune_id,
                "finetune_strength": finetune_strength,
                "prompt": prompt,
                **kwargs
            }

            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            
            task_id = result.get("id")
            if not task_id:
                raise ValueError("No task ID received in response")
                
            return (*self.get_result(task_id, kwargs.get("output_format", "png")), finetune_id)

        except Exception as e:
            print(f"Inference Error: {str(e)}")
            return (*self.create_blank_image(), "")

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

    def get_result(self, task_id, output_format, attempt=1, max_attempts=10):
        if attempt > max_attempts:
            print(f"Max attempts reached for task_id {task_id}")
            return self.create_blank_image()

        try:
            get_url = f"https://api.bfl.ai/v1/get_result?id={task_id}"
            headers = {"x-key": os.environ["X_KEY"]}
            
            response = requests.get(get_url, headers=headers, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                status = result.get("status")
                
                if status == Status.READY.value:
                    sample_url = result.get('result', {}).get('sample')
                    if not sample_url:
                        raise ValueError("No sample URL in response")
                        
                    img_response = requests.get(sample_url, timeout=30)
                    img = Image.open(io.BytesIO(img_response.content))
                    
                    with io.BytesIO() as output:
                        img.save(output, format=output_format.upper())
                        output.seek(0)
                        img_converted = Image.open(output)
                        img_array = np.array(img_converted).astype(np.float32) / 255.0
                        return (torch.from_numpy(img_array)[None,],)
                        
                elif status == Status.PENDING.value:
                    print(f"Attempt {attempt}: Image not ready. Retrying in 5 seconds...")
                    time.sleep(5)
                    return self.get_result(task_id, output_format, attempt + 1)
                else:
                    print(f"Unexpected status: {status}")
                    return self.create_blank_image()
                    
        except Exception as e:
            print(f"Error retrieving result: {str(e)}")
            
        return self.create_blank_image()

NODE_CLASS_MAPPINGS = {
    "FluxPro11WithFinetune": FluxPro11WithFinetune
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxPro11WithFinetune": "Flux Pro 1.1 Ultra & Raw with Finetuning"
}
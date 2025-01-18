# ComfyUI Flux Node - Flux Pro 1.1 Ultra & Raw with Finetuning

A custom node for ComfyUI that integrates with [Black Forest Labs](https://blackforestlabs.ai/) FLUX API, providing access to FLUX's image generation and finetuning capabilities.

![BFL](https://github.com/user-attachments/assets/5da6c879-cb17-4c18-9e53-051ea6421fa1)

## Features

Core Features:
* Support for FLUX 1.1 [pro] regular and Ultra modes
* Optional Raw mode for more natural-looking images
* Multiple aspect ratios support
* Configurable safety tolerance
* Support for both JPEG and PNG output formats
* Seed support for reproducible results

Additional Capabilities:
- Complete finetuning system integration
- Model customization options
- Training mode selection (character/product/style/general)
- Inference with adjustable strength
- Comprehensive error handling

## Requirements
* ComfyUI
* Black Forest Labs API key
* Python packages:
   * requests
   * Pillow
   * numpy
   * torch

## Installation

1. Create a folder in your `ComfyUI/custom_nodes/` directory named `ComfyUI_Flux_1.1_RAW_API`
2. Copy the files from this repository into that folder
3. Create a `config.ini` file in the same directory with your FLUX API key

Example `config.ini`:
```ini
[API]
X_KEY=your_api_key_here
BASE_URL=https://api.bfl.ai/
```

## Useful Links

- Main Website: [Black Forest Labs](https://blackforestlabs.ai/)
- API Portal: https://api.us1.bfl.ai/auth/login 

## Getting Started

1. Create an account on the API Portal
2. Generate your API key from the dashboard
3. Create a `config.ini` file in your node directory with your API key (see Installation section)

## Pricing

### Finetuning Training
- **Short** ($2): < 150 steps - For fast exploration
- **Medium** ($4): 150-500 steps - For standard use cases
- **Long** ($6): > 500 steps - For difficult tasks and extreme precision

### Image Generation
- **FLUX 1.1 Pro Ultra Finetuned**: $0.07 per image
- **FLUX 1.1 Pro Ultra**: $0.06 per image - Best for photo-realistic images at 2k resolution
- **FLUX 1.1 Pro**: $0.04 per image - Efficient for large-scale generation
- **FLUX.1 Pro**: $0.05 per image - Original pro model
- **FLUX.1 Dev**: $0.025 per image - Distilled model

## Usage

### Node Parameters
* **prompt**: Text prompt describing the desired image
* **ultra_mode**: Enable Ultra mode for higher resolution output
* **aspect_ratio**: Choose from multiple aspect ratios:
   * 21:9 (Ultrawide)
   * 16:9 (Widescreen)
   * 4:3 (Standard)
   * 1:1 (Square)
   * 3:4 (Portrait)
   * 9:16 (Vertical)
   * 9:21 (Tall)
* **safety_tolerance**: 0-6 (0 being most strict, 6 being least strict)
* **output_format**: Choose between JPEG and PNG
* **raw**: Enable Raw mode for less processed, more natural-looking images
* **seed**: Optional seed for reproducible results (-1 for random)

### Mode Differences

**Regular Mode**
* Resolution based on aspect ratio (up to 1440px)
* Standard image processing

**Ultra Mode**
* Higher resolution output (up to 4MP)
* Support for Raw mode
* Advanced image processing

### Basic Workflow
1. Add the "Flux Pro 1.1 Ultra & Raw with Finetuning" node to your workflow
2. Set mode to "generate"
3. Enter your prompt
4. Configure desired settings (aspect ratio, safety tolerance, etc.)
5. Connect to a Preview Image node to see results

### Finetuning
1. Prepare your training data:
   - Create a folder with 5-20 images (JPG, JPEG, PNG, or WebP)
   - Optionally add text descriptions (same filename but .txt extension)
   - Compress the folder into a ZIP file

2. Training:
   - Set mode to "finetune"
   - Provide path to your ZIP file
   - Set finetune parameters:
     - `finetune_comment`: Description of your model
     - `finetune_mode`: character/product/style/general
     - `trigger_word`: Word to reference your concept (default: "TOK")
     - Adjust other parameters as needed

3. Using Finetuned Models:
   - Set mode to "inference"
   - Input your finetune_id
   - Use the trigger word in your prompt
   - Adjust finetune_strength if needed (default: 1.2)

## Parameters

### Generation Parameters
- `mode`: generate/finetune/inference
- `prompt`: Text description of desired image
- `ultra_mode`: Enable/disable ultra mode
- `aspect_ratio`: 21:9, 16:9, 4:3, 1:1, 3:4, 9:16, 9:21
- `safety_tolerance`: 0-6
- `output_format`: jpeg/png
- `raw`: Enable raw mode for extra realism
- `seed`: Set specific seed (-1 for random)

### Finetuning Parameters
- `finetune_zip`: Path to training data ZIP
- `finetune_comment`: Model description
- `finetune_id`: ID for inference mode
- `trigger_word`: Concept reference word
- `finetune_mode`: character/product/style/general
- `iterations`: Training steps (min: 100)
- `learning_rate`: Training rate (default: 0.00001)
- `captioning`: Auto-caption images
- `priority`: speed/quality
- `finetune_type`: full/lora
- `lora_rank`: 16 or 32
- `finetune_strength`: 0-2 (default: 1.2)

## Best Practices

### Finetuning Tips
- Use high-quality training images
- For character training, avoid multiple characters in single images
- For style training, try higher finetune_strength values
- If concept isn't showing strongly, increase finetune_strength
- If seeing artifacts, decrease finetune_strength

### Image Generation Tips
- Be specific in your prompts
- Use raw mode for more photorealistic results
- Adjust safety_tolerance based on your needs
- Consider aspect ratio for your use case

## Troubleshooting

Common issues:
1. "Error: X_KEY not found in section API of config file"
   * Check if your `config.ini` file exists and is properly formatted
   * Verify the API key is correctly copied without extra spaces
   * Make sure the file is in the correct directory

2. "404 Not Found" error
   * Verify your API key is valid and active
   * Check if you have access to the FLUX 1.1 API
   * Make sure your account is in good standing

### Error Handling
The node includes comprehensive error handling:
- Network errors return a blank image
- Invalid API keys are caught and reported
- Missing files generate appropriate errors
- Training/inference errors are logged with details

## Contributing

Feel free to submit issues and enhancement requests!

## Acknowledgements

- FLUX API team for their powerful image generation capabilities
- ComfyUI community for the node framework

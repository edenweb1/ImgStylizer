import os
import uuid
import shutil
import gradio as gr
import replicate
from PIL import Image, ImageDraw
from io import BytesIO
import requests
import torch
from diffusers import (
    StableDiffusionPipeline, 
    StableDiffusionControlNetPipeline, 
    ControlNetModel,
    UniPCMultistepScheduler
)

from safetensors.torch import load_file as load_safetensor
from controlnet_aux.processor import Processor as ControlNetAuxProcessor

# --- Configuration & Setup ---
REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN", "")
if REPLICATE_API_TOKEN:
    os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN
    try:
        replicate.Client(api_token=REPLICATE_API_TOKEN)
        print("Replicate client initialized.")
    except Exception as e:
        print(f"Failed to initialize Replicate client: {e}. Replicate backend may not work.")
else:
    print("Warning: REPLICATE_API_TOKEN environment variable not set. Replicate backend will be unavailable.")

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODELS_DIR = os.path.join(BASE_DIR, "models")
LORA_DIR = os.path.join(BASE_DIR, "lora")
CONTROLNET_MODELS_DIR = os.path.join(BASE_DIR, "controlnet_models")
IPADAPTER_MODELS_DIR = os.path.join(BASE_DIR, "ipadapter_models")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LORA_DIR, exist_ok=True)
os.makedirs(CONTROLNET_MODELS_DIR, exist_ok=True)
os.makedirs(IPADAPTER_MODELS_DIR, exist_ok=True)

controlnet_preprocessors_cache = {}

DEFAULT_BASE_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"

# --- ControlNet Configuration ---
CONTROLNET_PROCESSOR_IDS = {
    "Lineart": "lineart_realistic", 
    "Canny Edge": "canny",          
    "Scribble (HED)": "scribble_hed", # Holistically-Nested Edge Detection based scribble
    "Scribble (PIDI)": "scribble_pidinet", # PiDiNet based scribble
    # Add more: "OpenPose": "openpose_full" 
}

CONTROLNET_MODEL_INFO = {
    "Lineart": {
        "hf_id": "lllyasviel/control_v11p_sd15_lineart",
        "local_files": ["control_v11p_sd15_lineart.safetensors", "control_v11p_sd15_lineart.pth"]
    },
    "Canny Edge": {
        "hf_id": "lllyasviel/control_v11p_sd15_canny",
        "local_files": ["control_v11p_sd15_canny.safetensors", "control_v11p_sd15_canny.pth"]
    },
    "Scribble (HED)": { # Both HED and PIDI scribble preprocessors can use the same scribble ControlNet model
        "hf_id": "lllyasviel/control_v11p_sd15_scribble", 
        "local_files": ["control_v11p_sd15_scribble.safetensors", "control_v11p_sd15_scribble.pth"]
    },
    "Scribble (PIDI)": { # Both HED and PIDI scribble preprocessors can use the same scribble ControlNet model
        "hf_id": "lllyasviel/control_v11p_sd15_scribble",
        "local_files": ["control_v11p_sd15_scribble.safetensors", "control_v11p_sd15_scribble.pth"]
    },
    # Removed Depth Map
}
# --- End ControlNet Configuration ---


def get_controlnet_preprocessor(processor_id: str):
    if processor_id not in controlnet_preprocessors_cache:
        print(f"Initializing ControlNet preprocessor: {processor_id}...")
        try:
            controlnet_preprocessors_cache[processor_id] = ControlNetAuxProcessor(processor_id=processor_id)
            print(f"Successfully initialized {processor_id}.")
        except Exception as e:
            print(f"Failed to initialize ControlNet preprocessor {processor_id}: {e}")
            controlnet_preprocessors_cache[processor_id] = None
            return None
    return controlnet_preprocessors_cache[processor_id]

def get_model_list(model_dir: str, default_hf_id: str = DEFAULT_BASE_MODEL_ID):
    try:
        models = [f.name for f in os.scandir(model_dir) if f.is_file() and f.name.endswith((".safetensors", ".ckpt", ".pt"))]
        available_choices = models
        if default_hf_id not in models: 
            available_choices = [default_hf_id] + models
        
        if not models: 
            print(f"No local models found in '{model_dir}'. Using default '{default_hf_id}'.")
            return [default_hf_id]
            
        print(f"Available models (local and default): {available_choices}")
        return available_choices
    except FileNotFoundError:
        print(f"Model directory '{model_dir}' not found. Using default '{default_hf_id}'.")
        return [default_hf_id]

def get_lora_list(lora_dir: str):
    try:
        loras = [f.name for f in os.scandir(lora_dir) if f.is_file() and f.name.endswith((".safetensors", ".pt", ".ckpt"))]
        print(f"Available LoRAs in '{lora_dir}': {loras}")
        return loras
    except FileNotFoundError:
        print(f"LoRA directory '{lora_dir}' not found. No LoRAs will be listed.")
        return []

def resize_for_processor(image: Image.Image, max_side: int = 768) -> Image.Image:
    """Resizes an image to have its longest side as max_side, preserving aspect ratio."""
    w, h = image.size
    if w > max_side or h > max_side:
        if w > h:
            new_w = max_side
            new_h = int(h * (max_side / w))
        else:
            new_h = max_side
            new_w = int(w * (max_side / h))
        print(f"Resizing image from {image.size} to {(new_w, new_h)} for processor input.")
        return image.resize((new_w, new_h), Image.LANCZOS)
    return image


@torch.inference_mode()
def run_local_model(
    local_model_name_or_path: str,
    lora_name: str,
    prompt: str,
    negative_prompt: str,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
    control_image_for_processing: Image.Image = None, 
    controlnet_type: str = "None", 
    controlnet_conditioning_scale: float = 1.0,
    ip_adapter_image_pil: Image.Image = None,
    ip_adapter_scale: float = 0.6,
    target_width: int = 512,
    target_height: int = 768
):
    if local_model_name_or_path != DEFAULT_BASE_MODEL_ID and os.path.exists(os.path.join(MODELS_DIR, local_model_name_or_path)):
        model_id_or_path = os.path.join(MODELS_DIR, local_model_name_or_path)
    else: 
        model_id_or_path = local_model_name_or_path

    print(f"Attempting to load base model: {model_id_or_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cpu": print("Warning: Running on CPU will be very slow.")

    controlnet_model_instance = None
    processed_conditioning_image_map = None 
    pipe = None

    # --- ControlNet Setup ---
    if controlnet_type and controlnet_type.lower() != "none" and control_image_for_processing:
        print(f"Setting up ControlNet with type: {controlnet_type}")
        
        processor_id = CONTROLNET_PROCESSOR_IDS.get(controlnet_type)
        model_info = CONTROLNET_MODEL_INFO.get(controlnet_type)

        if processor_id and model_info:
            processor = get_controlnet_preprocessor(processor_id)
            if processor:
                print(f"Processing control image with {processor_id} for {controlnet_type}...")
                try:
                    print(f"Original control source image size: {control_image_for_processing.size}")
                    control_source_for_processor = resize_for_processor(control_image_for_processing.convert("RGB"), 768) 
                    print(f"Resized control source for processor to: {control_source_for_processor.size}")

                    raw_conditioning_map = processor(control_source_for_processor).convert("RGB")
                    print(f"Raw conditioning map size from processor: {raw_conditioning_map.size}")

                    processed_conditioning_image_map = raw_conditioning_map.resize((target_width, target_height), Image.LANCZOS)
                    print(f"Final ControlNet conditioning map size: {processed_conditioning_image_map.size}")
                    
                    debug_map_path = os.path.join(OUTPUT_DIR, f"debug_control_map_{controlnet_type.lower().replace(' ', '_').replace('(', '').replace(')', '')}_{uuid.uuid4()}.png")
                    processed_conditioning_image_map.save(debug_map_path)
                    print(f"Saved debug ControlNet map to: {debug_map_path}")

                    huggingface_controlnet_id = model_info["hf_id"]
                    controlnet_model_to_load = huggingface_controlnet_id 

                    for local_filename in model_info["local_files"]:
                        local_path = os.path.join(CONTROLNET_MODELS_DIR, local_filename)
                        if os.path.exists(local_path):
                            controlnet_model_to_load = local_path
                            print(f"Using local ControlNet {controlnet_type} model: {controlnet_model_to_load}")
                            break
                    else: 
                        print(f"Local ControlNet {controlnet_type} model not found in {CONTROLNET_MODELS_DIR}. Will attempt to download from Hugging Face: {huggingface_controlnet_id}")
                    
                    controlnet_model_instance = ControlNetModel.from_pretrained(
                        controlnet_model_to_load, 
                        torch_dtype=torch.float16 if device == "cuda" else torch.float32
                    )
                    print(f"Loaded ControlNet model from: {controlnet_model_to_load}")
                except Exception as e:
                    print(f"Error during ControlNet {controlnet_type} setup: {e}")
                    controlnet_model_instance = None
                    processed_conditioning_image_map = None
            else:
                print(f"Could not get processor for {controlnet_type}. Skipping ControlNet.")
        else:
            print(f"ControlNet type '{controlnet_type}' not configured in CONTROLNET_PROCESSOR_IDS or CONTROLNET_MODEL_INFO. Skipping.")
            controlnet_type = "None"
    else:
        if controlnet_type and controlnet_type.lower() != "none":
            print("Debug: ControlNet type selected, but no control_image_for_processing provided.")


    # --- Initialize Diffusion Pipeline ---
    print(f"Debug: Before pipeline init - controlnet_model_instance is {'VALID' if controlnet_model_instance else 'None'}")
    print(f"Debug: Before pipeline init - processed_conditioning_image_map is {'VALID' if processed_conditioning_image_map else 'None'}")
    try:
        intended_torch_dtype = torch.float16 if device == "cuda" else torch.float32
        pipeline_args = {"torch_dtype": intended_torch_dtype, "safety_checker": None, "requires_safety_checker": False}
        pipeline_class = StableDiffusionPipeline 

        if controlnet_model_instance and processed_conditioning_image_map:
            print(f"Debug: Initializing WITH ControlNet: StableDiffusionControlNetPipeline")
            pipeline_class = StableDiffusionControlNetPipeline
            pipeline_args["controlnet"] = controlnet_model_instance
        else:
            print(f"Debug: Initializing WITHOUT ControlNet: StableDiffusionPipeline")
            
        if model_id_or_path.endswith((".safetensors", ".ckpt")) and os.path.isfile(model_id_or_path):
            pipe = pipeline_class.from_single_file(model_id_or_path, **pipeline_args)
        else:
            pipe = pipeline_class.from_pretrained(model_id_or_path, **pipeline_args)
        
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.to(device)
    except Exception as e:
        print(f"Fatal error initializing diffusion pipeline: {e}")
        error_img = Image.new("RGB", (target_width, target_height), color="black")
        ImageDraw.Draw(error_img).text((10, 10), f"Pipeline Init Error:\n{str(e)[:100]}...", fill="red")
        return error_img

    # --- Load LoRA ---
    if lora_name and lora_name.lower() != "none":
        lora_file_path = os.path.join(LORA_DIR, lora_name)
        if os.path.exists(lora_file_path):
            print(f"Loading LoRA: {lora_file_path}")
            try:
                pipe.load_lora_weights(lora_file_path)
                print(f"Successfully loaded LoRA: {lora_name}")
            except Exception as e:
                print(f"Error loading LoRA {lora_file_path}: {e}")
        else:
            print(f"LoRA file not found: {lora_file_path}")

    # --- Load IPAdapter ---
    if ip_adapter_image_pil and ip_adapter_scale > 0: 
        print("Setting up IPAdapter...")
        try:
            ip_adapter_model_id_or_path = "h94/IP-Adapter"
            ip_adapter_filename = "ip-adapter_sd15.bin" 
            
            local_ip_adapter_config_path = os.path.join(IPADAPTER_MODELS_DIR, "ip-adapter_sd15") 
            local_ip_adapter_bin_path = os.path.join(local_ip_adapter_config_path, ip_adapter_filename)

            if os.path.exists(local_ip_adapter_bin_path) and os.path.isdir(os.path.join(local_ip_adapter_config_path, "image_encoder")):
                print(f"Using local IPAdapter model from: {local_ip_adapter_config_path}")
                pipe.load_ip_adapter(local_ip_adapter_config_path, subfolder="", weight_name=ip_adapter_filename)
            else:
                print(f"Local IPAdapter model not found. Will download from HF: {ip_adapter_model_id_or_path} (using {ip_adapter_filename})")
                pipe.load_ip_adapter(ip_adapter_model_id_or_path, subfolder="models", weight_name=ip_adapter_filename)
            
            pipe.set_ip_adapter_scale(ip_adapter_scale)
            print(f"IPAdapter loaded. Scale set to: {ip_adapter_scale}")
        except Exception as e:
            print(f"Error loading IPAdapter: {e}. Proceeding without IPAdapter.")
            ip_adapter_image_pil = None 
    elif ip_adapter_image_pil and ip_adapter_scale == 0:
        print("IPAdapter image provided but scale is 0. Skipping IPAdapter loading.")
        ip_adapter_image_pil = None 
    else:
        print("No IPAdapter style image or scale is 0, skipping IPAdapter setup.")
        ip_adapter_image_pil = None


    # --- Prepare Inputs for Pipeline ---
    generator = torch.Generator(device=device).manual_seed(seed) if seed != -1 else torch.Generator(device=device).manual_seed(torch.randint(0, 10**6, (1,)).item())

    pipeline_call_args = {
        "prompt": prompt, "negative_prompt": negative_prompt,
        "width": target_width, "height": target_height, 
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps, "generator": generator,
    }

    if isinstance(pipe, StableDiffusionControlNetPipeline) and processed_conditioning_image_map:
        pipeline_call_args["image"] = processed_conditioning_image_map 
        pipeline_call_args["controlnet_conditioning_scale"] = float(controlnet_conditioning_scale)
    
    if ip_adapter_image_pil:
        pipeline_call_args["ip_adapter_image"] = ip_adapter_image_pil.convert("RGB").resize((224,224))

    print(f"Running inference with seed: {seed if seed != -1 else 'random'}...")
    print(f"Debug: Final pipeline_call_args keys: {list(pipeline_call_args.keys())}")
    if "image" in pipeline_call_args and pipeline_call_args["image"] is not None:
        print(f"Debug: Conditioning image (for ControlNet) is present. Size: {pipeline_call_args['image'].size}")
    elif "image" in pipeline_call_args:
        print("Debug: Conditioning image (for ControlNet) key exists but value is None.")
    
    if "controlnet_conditioning_scale" in pipeline_call_args:
        print(f"Debug: controlnet_conditioning_scale = {pipeline_call_args['controlnet_conditioning_scale']}")
    else:
        if isinstance(pipe, StableDiffusionControlNetPipeline) and "image" in pipeline_call_args and pipeline_call_args["image"] is not None:
             print("Debug: WARNING - ControlNet pipeline with image, but no controlnet_conditioning_scale in args.")
        elif isinstance(pipe, StableDiffusionControlNetPipeline):
            print("Debug: ControlNet pipeline but no conditioning image or scale in args.")


    try:
        autocast_enabled = (device == "cuda") 
        print(f"Autocast enabled for CUDA: {autocast_enabled}")
        with torch.amp.autocast(device_type=device, enabled=autocast_enabled, dtype=(torch.float16 if autocast_enabled and intended_torch_dtype == torch.float16 else None)): 
            generated_image = pipe(**pipeline_call_args).images[0]
        print("Inference complete.")
    except Exception as e:
        print(f"Error during pipeline inference: {e}")
        error_img = Image.new("RGB", (target_width, target_height), color="black")
        ImageDraw.Draw(error_img).text((10, 10), f"Inference Error:\n{str(e)[:100]}...", fill="red")
        return error_img
    finally:
        if lora_name and lora_name.lower() != "none" and hasattr(pipe, "unload_lora_weights"):
            try: pipe.unload_lora_weights(); print("Unloaded LoRA weights.")
            except Exception as e: print(f"Could not unload LoRA: {e}")
        
        if hasattr(pipe, "unload_ip_adapter"): 
            try: pipe.unload_ip_adapter(); print("Unloaded IP Adapter weights.")
            except Exception as e: print(f"Could not unload IP Adapter: {e}")

    return generated_image

def stylize(
    backend_choice: str,
    sketch_image_ui: Image.Image, 
    control_net_source_image_ui: Image.Image,
    ip_adapter_style_image_ui: Image.Image,
    local_model_name_ui: str,
    lora_name_ui: str,
    prompt_ui: str,
    negative_prompt_ui: str,
    num_inference_steps_ui: int,
    guidance_scale_ui: float,
    seed_ui: int,
    controlnet_type_ui: str = "None", 
    controlnet_scale_ui: float = 1.0,
    ip_adapter_scale_ui: float = 0.6,
    output_width_ui: int = 512,
    output_height_ui: int = 768
):
    actual_control_source_image = control_net_source_image_ui if control_net_source_image_ui else sketch_image_ui

    if controlnet_type_ui.lower() != "none" and not actual_control_source_image:
        raise gr.Error(f"Structural Guide ('{controlnet_type_ui}') selected, but no Sketch Image or Alternate Source image provided.")
    
    saved_sketch_path = None 
    if sketch_image_ui: 
        file_id_main_sketch = str(uuid.uuid4())
        saved_sketch_path = os.path.join(UPLOAD_DIR, f"{file_id_main_sketch}_sketch_input.png")
        sketch_image_ui.save(saved_sketch_path)
        print(f"Saved main sketch image to: {saved_sketch_path}")
    
    if control_net_source_image_ui and control_net_source_image_ui != sketch_image_ui:
        file_id_cn_source = str(uuid.uuid4())
        saved_cn_source_path = os.path.join(UPLOAD_DIR, f"{file_id_cn_source}_cn_source_input.png")
        control_net_source_image_ui.save(saved_cn_source_path)
        print(f"Saved dedicated ControlNet source image to: {saved_cn_source_path}")
    elif not actual_control_source_image and controlnet_type_ui.lower() != "none":
         print("Warning: Structural Guide is active, but no source image found. This shouldn't happen due to earlier check.")


    if backend_choice == "replicate":
        if not REPLICATE_API_TOKEN: raise gr.Error("Replicate API token not set.")
        if not sketch_image_ui: raise gr.Error("A Sketch Image is required as the base for the Replicate image style transfer.")
        if not ip_adapter_style_image_ui: raise gr.Error("Style Reference Image is required for the Replicate image style transfer.")
        
        if sketch_image_ui and (saved_sketch_path is None or not os.path.exists(saved_sketch_path)):
            file_id_rep_sketch = str(uuid.uuid4())
            saved_sketch_path = os.path.join(UPLOAD_DIR, f"{file_id_rep_sketch}_sketch_input_replicate.png")
            sketch_image_ui.save(saved_sketch_path)
            print(f"Re-saved sketch for Replicate (should be rare): {saved_sketch_path}")


        saved_style_path_replicate = os.path.join(UPLOAD_DIR, f"{str(uuid.uuid4())}_replicate_style.png")
        ip_adapter_style_image_ui.save(saved_style_path_replicate)
        print(f"Using Replicate. Input: {saved_sketch_path}, Style for Replicate: {saved_style_path_replicate}")
        gr.Warning("Structural Guide and Style Image Strength (IPAdapter) settings are for Local backend only.")

        with open(saved_sketch_path, "rb") as input_f, open(saved_style_path_replicate, "rb") as style_f:
            try:
                output_url = replicate.run(
                    "philz1337x/style-transfer:a15407d73d9669676d623e37ee3b6d43642439beec1b99639967d215bcf42fc4",
                    input={ 
                        "image": input_f, "image_style": style_f, "style_strength": 0.7, 
                        "prompt": prompt_ui, "negative_prompt": negative_prompt_ui,
                        "num_inference_steps": num_inference_steps_ui, "guidance_scale": guidance_scale_ui, "seed": seed_ui
                    }
                )[0]
                response = requests.get(output_url); response.raise_for_status()
                output_image = Image.open(BytesIO(response.content))
                output_path_replicate = os.path.join(OUTPUT_DIR, f"{str(uuid.uuid4())}_output_replicate.png")
                output_image.save(output_path_replicate)
                print(f"Replicate output saved: {output_path_replicate}")
                return output_image
            except Exception as e: raise gr.Error(f"Replicate API Error: {e}")
    else: 
        print(f"Using Local backend.")
        if sketch_image_ui: print(f"Sketch image provided.")
        if control_net_source_image_ui: print(f"Alternate Structural Guide Source image provided.")
        if actual_control_source_image: print(f"Actual image for Structural Guide processing: {'Alternate Source' if control_net_source_image_ui else 'Main Sketch Image'}")
        
        print(f"Structural Guide Type: {controlnet_type_ui}, Guide Strength: {controlnet_scale_ui}")
        if ip_adapter_style_image_ui:
            print(f"Style Reference Image provided. Style Image Strength: {ip_adapter_scale_ui}")
        else:
            print("No Style Reference Image provided.")

        output_img = run_local_model(
            local_model_name_or_path=local_model_name_ui,
            lora_name=lora_name_ui,
            prompt=prompt_ui,
            negative_prompt=negative_prompt_ui,
            num_inference_steps=num_inference_steps_ui,
            guidance_scale=guidance_scale_ui,
            seed=seed_ui,
            control_image_for_processing=actual_control_source_image,
            controlnet_type=controlnet_type_ui, 
            controlnet_conditioning_scale=controlnet_scale_ui,
            ip_adapter_image_pil=ip_adapter_style_image_ui,
            ip_adapter_scale=ip_adapter_scale_ui,
            target_width=output_width_ui,
            target_height=output_height_ui
        )
        output_path_local = os.path.join(OUTPUT_DIR, f"{str(uuid.uuid4())}_output_local.png")
        output_img.save(output_path_local)
        print(f"Local output saved: {output_path_local}")
        return output_img

def get_initial_model_choice():
    models = get_model_list(MODELS_DIR, default_hf_id=DEFAULT_BASE_MODEL_ID)
    if DEFAULT_BASE_MODEL_ID in models:
        return DEFAULT_BASE_MODEL_ID
    return models[0] if models else DEFAULT_BASE_MODEL_ID

css = """
body { font-family: 'Inter', sans-serif; }
.gradio-container { max-width: 1280px !important; margin: auto !important; }
footer { display: none !important; }
.gr-button { background-color: #007bff; color: white; border-radius: 8px; }
.gr-button:hover { background-color: #0056b3; }
.gr-panel, .gr-group { border-radius: 12px !important; box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important; padding: 15px !important; margin-bottom: 15px !important;}
.image-column > div { display: flex; flex-direction: column; align-items: center; }
.image-column > div > .label { text-align: center; margin-bottom: 4px; font-weight: bold; }
.gr-block.gr-box { border-radius: 12px !important; } 
h3 { margin-top: 20px; margin-bottom:10px; border-bottom: 1px solid #eee; padding-bottom: 5px;}
"""

with gr.Blocks(css=css, theme=gr.themes.Soft(primary_hue="blue", secondary_hue="sky")) as gr_interface:
    gr.Markdown("## 🎨 AI Image Creator")
    gr.Markdown("Create images from text prompts. You can guide the structure with a sketch and influence the artistic style with a reference image.")

    with gr.Group():
        gr.Markdown("### ⚙️ Setup & Models")
        with gr.Row():
            backend_choice_radio = gr.Radio(choices=["local", "replicate"], value="local", label="Processing Backend", info="Local backend enables all features.")
        with gr.Row():
            local_model_dropdown_ui = gr.Dropdown(
                choices=get_model_list(MODELS_DIR, default_hf_id=DEFAULT_BASE_MODEL_ID), 
                label="Base Model",
                value=get_initial_model_choice, 
                allow_custom_value=True,
                info=f"Select a base image generation model. Default: {DEFAULT_BASE_MODEL_ID}. Place custom models in '\image-stylizer\models'. SD 1.5 models recommended for structural guides."
            )
            lora_model_dropdown_ui = gr.Dropdown(
                choices=["None"] + get_lora_list(LORA_DIR), 
                label="Style/Character Model (LoRA - Optional)", 
                value="None",
                info=f"Optional: Select a LoRA model to influence style or characters. Place custom LoRAs in \image-stylizer\lora."
            )

    with gr.Group():
        gr.Markdown("### Inputs: Images & Prompt")
        gr.Markdown("Provide a sketch for structural guidance, a style reference image, and your text prompts.")
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, elem_classes="image-column"):
                gr.HTML("<div class='label'>Sketch Input (for Structural Guide)</div>")
                sketch_image_component_ui = gr.Image(label=None, type="pil", height=300, sources=["upload", "webcam", "clipboard"], show_label=False)
            with gr.Column(scale=1, elem_classes="image-column"):
                gr.HTML("<div class='label'>Alternate Structural Guide Source (Optional)</div>")
                control_net_source_image_component_ui = gr.Image(label=None, type="pil", height=300, sources=["upload", "webcam", "clipboard"], show_label=False)
            with gr.Column(scale=1, elem_classes="image-column"):
                gr.HTML("<div class='label'>Style Reference Image (Optional)</div>")
                ip_adapter_style_image_component_ui = gr.Image(label=None, type="pil", height=300, sources=["upload", "webcam", "clipboard"], show_label=False)
        
        with gr.Row():
            prompt_textbox_ui = gr.Textbox(value="Photo of a cute cat wearing a tiny hat, studio lighting, high detail", label="Prompt", lines=3, placeholder="Describe the image you want to create...", scale=2)
        with gr.Row():
            negative_prompt_textbox_ui = gr.Textbox(value="ugly, blurry, low quality, watermark, signature, text, extra limbs, disfigured", label="Negative Prompt (what to avoid)", lines=3, placeholder="Describe what you don't want to see...", scale=2)

    with gr.Group():
        gr.Markdown("### 🛠️ Generation Settings")
        with gr.Row():
            seed_number_ui = gr.Number(value=42, label="Seed", info="Same seed + same settings = same image. -1 for random.", precision=0, interactive=True)
            num_inference_steps_slider_ui = gr.Slider(minimum=1, maximum=150, value=25, step=1, label="Quality Steps", info="More steps can improve quality but take longer.")
        with gr.Row():
            output_width_slider_ui = gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Image Width")
            output_height_slider_ui = gr.Slider(minimum=256, maximum=1024, value=768, step=64, label="Image Height")
        
        guidance_scale_slider_ui = gr.Slider(minimum=1.0, maximum=30.0, value=7.5, step=0.1, label="Prompt Strength (CFG)", info="How strongly the AI should follow your prompt.")
        
        gr.Markdown("#### 🔌 Sketch & Style Influence (Local Backend)")
        with gr.Row():
            controlnet_type_dropdown_ui = gr.Dropdown( 
                choices=["None"] + list(CONTROLNET_PROCESSOR_IDS.keys()), 
                value="None", 
                label="Structural Guide Type (from Sketch)", 
                info="Uses the sketch to guide the image structure via selected method."
            )
            controlnet_scale_slider_ui = gr.Slider(minimum=0.0, maximum=2.0, value=0.8, step=0.05, label="Structural Guide Strength", info="How much the sketch influences the structure. 0 to disable.")
        with gr.Row():
            ip_adapter_scale_slider_ui = gr.Slider(minimum=0.0, maximum=1.5, value=0.6, step=0.05, label="Style Image Strength", info="How much the Style Reference Image influences the artistic style. 0 to disable.")

    with gr.Row():
        submit_button = gr.Button("✨ Generate Image ✨", variant="primary", scale=3) 
    
    output_image_component_ui = gr.Image(label="Generated Image", type="pil", height=512, interactive=False)

    submit_button.click(
        fn=stylize,
        inputs=[
            backend_choice_radio,
            sketch_image_component_ui,
            control_net_source_image_component_ui,
            ip_adapter_style_image_component_ui,
            local_model_dropdown_ui,
            lora_model_dropdown_ui,
            prompt_textbox_ui,
            negative_prompt_textbox_ui,
            num_inference_steps_slider_ui,
            guidance_scale_slider_ui,
            seed_number_ui,
            controlnet_type_dropdown_ui,
            controlnet_scale_slider_ui,
            ip_adapter_scale_slider_ui,
            output_width_slider_ui,
            output_height_slider_ui
        ],
        outputs=output_image_component_ui
    )

    with gr.Accordion("💡 Tips for Local Image Generation", open=False):
        gr.Markdown(f"- **Base Models**: Place custom models (e.g., `.safetensors`, `.ckpt`) in the `{MODELS_DIR}` folder.")
        gr.Markdown(f"- **Style/Character Models (LoRA)**: Place LoRA files in the `{LORA_DIR}` folder.")
        gr.Markdown(f"- **Structural Guide Models (ControlNet)**: If you have local ControlNet models (e.g., `control_v11p_sd15_lineart.safetensors`, `control_v11p_sd15_canny.safetensors`, `control_v11p_sd15_scribble.safetensors`), place them in `{CONTROLNET_MODELS_DIR}` to use them instead of downloading. Refer to `CONTROLNET_MODEL_INFO` in the script for expected filenames.")
        gr.Markdown(f"- **Style Reference Model (IPAdapter)**: For local IPAdapter, place `ip-adapter_sd15.bin` and the `image_encoder` folder into `{IPADAPTER_MODELS_DIR}/ip-adapter_sd15`. Otherwise, it will be downloaded.")
        gr.Markdown(f"- **Compatibility**: The default base model ({DEFAULT_BASE_MODEL_ID}) is recommended for Structural Guide and Style Reference features.")
        gr.Markdown(f"- **Output**: Generated images and debug maps are saved in the `{OUTPUT_DIR}` folder. Uploaded images are temporarily stored in `{UPLOAD_DIR}`.")

if __name__ == "__main__":
    gr_interface.queue().launch(share=True, debug=True)

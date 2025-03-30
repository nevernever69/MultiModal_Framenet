#!/usr/bin/env python
"""
llava_batch_processing.py

This script processes images using the LLaVA-1.5 model in parallel, converting images to base64 strings
to ensure the accepted input format. It writes structured analysis to a CSV while using GPUs as much as possible.
"""

import os
os.environ["HF_HUB_CACHE"] = "Model"

import csv
import re
import time
import datetime
import json
import gc
import psutil
import numpy as np
import base64
import io

from transformers import pipeline
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Constants for optimization and model configuration
BATCH_SIZE = 4  # Adjust based on GPU memory and performance
MAX_NEW_TOKENS = 1000  # Maximum tokens generated per image
COMPILE_MODEL = True
MODEL_ID = "llava-hf/llava-1.5-13b-hf"  # LLaVA model identifier

def image_to_base64(image_path):
    """
    Loads an image from a local path, converts it to RGB,
    and returns a base64-encoded string.
    """
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise ValueError(f"Error loading image {image_path}: {e}")
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def create_prompt(image_path):
    """
    Creates the prompt structure for the LLaVA model.
    The image is provided as a base64 string.
    The prompt text remains unchanged.
    """
    image_b64 = image_to_base64(image_path)
    return {
        "image": image_b64,
        "text": """You are a multimodal visual analysis expert assisting in the creation of a structured academic dataset based on user-uploaded news photographs.

You will be provided with a photograph sourced from a news article. The complexity of the image may range from simple (e.g., one person in a room) to complex (e.g., a protest or public gathering).

Please follow these steps to generate a structured output suitable for CSV export:

### **1. Entities & Relationships**

- **Instructions:**
- Identify **all visually identifiable entities** in the image.
- For each entity, provide a natural, descriptive phrase followed by its tag in square brackets only for Entities & Relationships.
- **Every entity mentioned in a sentence must be individually tagged.** Do not skip entities; if a sentence mentions several objects, each must appear with its corresponding tag.
- Use a comma-separated format for multiple entities within a single sentence.

- **Example:**
- Use:
*"Man [person] wearing a red and black soccer jersey [clothing] , shorts [clothing], and socks [clothing], jumping and celebrating."*

- **Required Entities to Look for:**
- People [person]
- Animals [animal]
- Vehicles [vehicle]
- Objects [object]
- Places [place]
- Clothing & accessories [clothing], [accessory]
- Tools & equipment [tool], [equipment]
- Text and signs [text]
- Screens or digital displays [screen]
- Furniture & fixtures [furniture], [fixture]
- Food & drink [food], [drink]
- Logos and symbols [logo], [symbol]
- Groups or clusters [group]
- Money & payment items [money], [payment]
- Toys & play items [toy]
- Firearms [real_gun], [toy_gun]
- Musical instruments [musical_instrument]
- Cosmetics & beauty products [cosmetics]
- Art & artistic media [art]
- Plants & vegetation [plant]

- **Format:**
- List each entity as a bullet point, following the natural order they appear or interact in the image.

### **2. Scene Description:**

- Write a rich, detailed description of the scene.
- Include spatial relationships, background elements, and general setting/context relevant to identifying and localizing entities.
- Be objective and avoid culturally biased interpretations.
- Do not infer names or identities unless explicitly shown in the image (e.g., "man in red shirt" instead of "John").
- After writing the description in English, provide a Brazilian Portuguese translation of the same description.
- Maintain all entity labels (e.g., [person], [vehicle], [text]) unchanged and in their original positions within the translated version.

### **3. Event Description:**

- Describe the main activity or event depicted.
- If ambiguous, provide up to three plausible interpretations, clearly separated (e.g., "Possibility 1: ...", "Possibility 2: ...").
- Focus on the purpose, intent, or core action of the image.

### **4. Objects List:**

- Provide an explicit, exhaustive list of objects detected in the image without tags.
- Ensure all objects referenced in 'Entities & Relationships' are listed here.

**Ensure all sections are well-structured, labeled, and formatted as bullet points.**"""
    }

def create_batch_prompts(image_paths):
    """
    Creates a batch of prompts in the format expected by the LLaVA model.
    """
    batch_prompts = []
    for img_path in image_paths:
        prompt = create_prompt(img_path)
        batch_prompts.append(prompt)
    return batch_prompts

def extract_details(response_text):
    """
    Extracts structured sections from the response text.
    """
    headers = [
        ("entities_relationships", r"### \*\*1\.\s*Entities & Relationships.*?\*\*"),
        ("scene_description", r"### \*\*2\.\s*Scene Description.*?\*\*"),
        ("event_description", r"### \*\*3\.\s*Event Description.*?\*\*"),
        ("objects_list", r"### \*\*4\.\s*Objects List.*?\*\*")
    ]
    fallback_headers = [
        ("entities_relationships", r"### 1\.\s*Entities & Relationships"),
        ("scene_description", r"### 2\.\s*Scene Description"),
        ("event_description", r"### 3\.\s*Event Description"),
        ("objects_list", r"### 4\.\s*Objects List")
    ]
    
    header_starts = {}
    header_ends = {}
    for key, pat in headers:
        match = re.search(pat, response_text)
        if match:
            header_starts[key] = match.start()
            header_ends[key] = match.end()
    for key, pat in fallback_headers:
        if key not in header_starts:
            match = re.search(pat, response_text)
            if match:
                header_starts[key] = match.start()
                header_ends[key] = match.end()
    
    details = {}
    if header_starts:
        sorted_keys = sorted(header_starts, key=header_starts.get)
        for i, key in enumerate(sorted_keys):
            start = header_ends[key]
            end = header_starts[sorted_keys[i + 1]] if i + 1 < len(sorted_keys) else None
            text = response_text[start:end].strip() if end else response_text[start:].strip()
            lines = [re.sub(r"^\*\s+", "", line) for line in text.split("\n") if line.strip()]
            details[key] = "\n".join(lines)
    return {key: details.get(key, "No data available") for key, _ in headers}

def remove_markdown_bolding(text):
    """
    Removes Markdown bolding from the given text.
    """
    return re.sub(r'\*\*(.*?)\*\*', r'\1', text)

class StatsTracker:
    """
    Class to track and report processing statistics.
    """
    def __init__(self, total_images):
        self.start_time = time.time()
        self.total_images = total_images
        self.processed_images = 0
        self.success_count = 0
        self.failed_count = 0
        self.processing_times = []
        self.batch_processing_times = []
        self.batch_sizes = []
        self.gpu_utilization = []
        self.gpu_memory_used = []
        self.cpu_utilization = []
        self.ram_usage = []
        
    def update_resource_stats(self):
        self.cpu_utilization.append(psutil.cpu_percent())
        self.ram_usage.append(psutil.virtual_memory().percent)
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                try:
                    self.gpu_utilization.append(torch.cuda.utilization(i))
                    props = torch.cuda.get_device_properties(i)
                    mem_used = torch.cuda.memory_allocated(i)
                    total_mem = props.total_memory
                    self.gpu_memory_used.append(mem_used / total_mem * 100)
                except Exception:
                    pass
                    
    def add_batch_result(self, batch_size, batch_time, success_count, fail_count):
        self.batch_sizes.append(batch_size)
        self.batch_processing_times.append(batch_time)
        self.processed_images += batch_size
        self.success_count += success_count
        self.failed_count += fail_count
        self.update_resource_stats()
        
    def add_image_time(self, processing_time):
        self.processing_times.append(processing_time)
    
    def get_progress_string(self):
        elapsed = time.time() - self.start_time
        images_per_second = self.processed_images / elapsed if elapsed > 0 else 0
        eta_seconds = (self.total_images - self.processed_images) / images_per_second if images_per_second > 0 else 0
        eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
        return (f"Progress: {self.processed_images}/{self.total_images} images "
                f"({(self.processed_images/self.total_images*100):.1f}%) | "
                f"Success: {self.success_count} | Failed: {self.failed_count} | "
                f"Time elapsed: {str(datetime.timedelta(seconds=int(elapsed)))} | "
                f"ETA: {eta_str} | "
                f"Speed: {images_per_second:.2f} img/s")
    
    def get_summary_stats(self):
        elapsed = time.time() - self.start_time
        if self.processing_times:
            avg_image_time = np.mean(self.processing_times)
            min_image_time = np.min(self.processing_times)
            max_image_time = np.max(self.processing_times)
            median_image_time = np.median(self.processing_times)
            p95_image_time = np.percentile(self.processing_times, 95)
        else:
            avg_image_time = min_image_time = max_image_time = median_image_time = p95_image_time = 0
        if self.batch_processing_times and sum(self.batch_processing_times) > 0:
            avg_batch_time = np.mean(self.batch_processing_times)
            avg_batch_size = np.mean(self.batch_sizes) if self.batch_sizes else 0
            images_per_second = sum(self.batch_sizes) / sum(self.batch_processing_times)
        else:
            avg_batch_time = avg_batch_size = images_per_second = 0
        avg_cpu = np.mean(self.cpu_utilization) if self.cpu_utilization else 0
        avg_ram = np.mean(self.ram_usage) if self.ram_usage else 0
        avg_gpu_util = np.mean(self.gpu_utilization) if self.gpu_utilization else 0
        avg_gpu_mem = np.mean(self.gpu_memory_used) if self.gpu_memory_used else 0
        
        return {
            "total_runtime_seconds": elapsed,
            "total_runtime_formatted": str(datetime.timedelta(seconds=int(elapsed))),
            "total_images": self.total_images,
            "processed_images": self.processed_images,
            "success_count": self.success_count,
            "failed_count": self.failed_count,
            "success_rate": (self.success_count / self.processed_images * 100) if self.processed_images > 0 else 0,
            "image_processing_stats": {
                "average_seconds_per_image": avg_image_time,
                "min_seconds_per_image": min_image_time,
                "max_seconds_per_image": max_image_time,
                "median_seconds_per_image": median_image_time,
                "95th_percentile_seconds": p95_image_time,
                "images_per_second": images_per_second
            },
            "batch_processing_stats": {
                "average_batch_size": avg_batch_size,
                "average_seconds_per_batch": avg_batch_time,
                "total_batches": len(self.batch_processing_times)
            },
            "resource_utilization": {
                "average_cpu_percent": avg_cpu,
                "average_ram_percent": avg_ram,
                "average_gpu_utilization_percent": avg_gpu_util,
                "average_gpu_memory_percent": avg_gpu_mem
            },
            "model_config": {
                "model_id": MODEL_ID,
                "batch_size": BATCH_SIZE,
                "max_new_tokens": MAX_NEW_TOKENS,
                "model_compilation": COMPILE_MODEL
            }
        }
    
    def save_stats_to_file(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.get_summary_stats(), f, indent=2)
            
    def print_final_report(self):
        stats = self.get_summary_stats()
        img_stats = stats["image_processing_stats"]
        batch_stats = stats["batch_processing_stats"]
        res_stats = stats["resource_utilization"]
        
        print("\n" + "="*80)
        print(f"LLAVA IMAGE PROCESSING COMPLETE - PERFORMANCE REPORT")
        print("="*80)
        print(f"Total runtime: {stats['total_runtime_formatted']}")
        print(f"Images processed: {stats['processed_images']}/{stats['total_images']} ({stats['processed_images']/stats['total_images']*100:.1f}%)")
        print(f"Success rate: {stats['success_rate']:.1f}% ({stats['success_count']} succeeded, {stats['failed_count']} failed)")
        print(f"Performance: {img_stats['images_per_second']:.2f} images/second")
        print("\nIMAGE TIMING STATISTICS:")
        print(f"  Average time per image: {img_stats['average_seconds_per_image']:.2f} seconds")
        print(f"  Median time per image: {img_stats['median_seconds_per_image']:.2f} seconds")
        print(f"  95th percentile time: {img_stats['95th_percentile_seconds']:.2f} seconds")
        print(f"  Fastest image: {img_stats['min_seconds_per_image']:.2f} seconds")
        print(f"  Slowest image: {img_stats['max_seconds_per_image']:.2f} seconds")
        print("\nBATCH PROCESSING STATISTICS:")
        print(f"  Total batches: {batch_stats['total_batches']}")
        print(f"  Average batch size: {batch_stats['average_batch_size']:.1f} images")
        print(f"  Average time per batch: {batch_stats['average_seconds_per_batch']:.2f} seconds")
        print("\nRESOURCE UTILIZATION:")
        print(f"  Average CPU utilization: {res_stats['average_cpu_percent']:.1f}%")
        print(f"  Average RAM utilization: {res_stats['average_ram_percent']:.1f}%")
        print(f"  Average GPU utilization: {res_stats['average_gpu_utilization_percent']:.1f}%")
        print(f"  Average GPU memory usage: {res_stats['average_gpu_memory_percent']:.1f}%")
        print("\nCONFIGURATION:")
        print(f"  Model: {stats['model_config']['model_id']}")
        print(f"  Batch size: {stats['model_config']['batch_size']}")
        print(f"  Max new tokens: {stats['model_config']['max_new_tokens']}")
        print(f"  Model compilation: {'Enabled' if stats['model_config']['model_compilation'] else 'Disabled'}")
        print("="*80)

def create_optimized_pipeline():
    """
    Creates an optimized image-to-text pipeline for LLaVA.
    Clears GPU cache before loading and compiles the model if enabled.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    print("Loading LLaVA model with pipeline...")
    start_time = time.time()
    pipe = pipeline(
        "image-to-text",
        model=MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto"  # Adjusts model loading across available GPUs
    )
    
    if COMPILE_MODEL and hasattr(torch, 'compile') and hasattr(pipe, 'model'):
        try:
            print("Compiling model for faster execution...")
            pipe.model = torch.compile(pipe.model, mode="reduce-overhead")
            print("Model compilation successful")
        except Exception as e:
            print(f"Model compilation failed, continuing without compilation: {e}")
    
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f} seconds.")
    
    if torch.cuda.is_available():
        print("GPU Information:")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            mem_used = torch.cuda.memory_allocated(i) / 1024**3
            mem_total = props.total_memory / 1024**3
            print(f"GPU {i}: {props.name} - Memory: {mem_used:.2f}GB / {mem_total:.2f}GB")
    
    return pipe

def process_batch(batch_image_paths, pipe, stats_tracker):
    """
    Processes a batch of images using the pipeline.
    Converts prompts, sends the batch to the model, and extracts responses.
    """
    results = []
    try:
        batch_prompts = create_batch_prompts(batch_image_paths)
        with torch.inference_mode():
            batch_start_time = time.time()
            batch_responses = pipe(
                batch_prompts,
                max_new_tokens=MAX_NEW_TOKENS,
                batch_size=BATCH_SIZE
            )
            batch_duration = time.time() - batch_start_time
        
        success_count = 0
        fail_count = 0
        for i, (img_path, response) in enumerate(zip(batch_image_paths, batch_responses)):
            try:
                if response and isinstance(response, list) and len(response) > 0:
                    analysis = response[0]['generated_text'] if isinstance(response[0], dict) else response[0]
                    analysis_data = extract_details(analysis)
                    cleaned_data = {k: remove_markdown_bolding(v) for k, v in analysis_data.items()}
                    image_time = batch_duration / len(batch_image_paths)
                    stats_tracker.add_image_time(image_time)
                    results.append({
                        'image_path': img_path,
                        'status': 'success',
                        'entities_relationships': cleaned_data["entities_relationships"],
                        'scene_description': cleaned_data["scene_description"],
                        'event_description': cleaned_data["event_description"],
                        'objects_list': cleaned_data["objects_list"],
                        'duration': image_time
                    })
                    success_count += 1
                else:
                    results.append({'image_path': img_path, 'status': 'failed', 'error': 'Empty or invalid response'})
                    fail_count += 1
            except Exception as e:
                print(f"Error processing response for {img_path}: {e}")
                results.append({'image_path': img_path, 'status': 'failed', 'error': str(e)})
                fail_count += 1
        
        stats_tracker.add_batch_result(len(batch_image_paths), batch_duration, success_count, fail_count)
        
    except Exception as e:
        print(f"Error processing batch: {e}")
        for img_path in batch_image_paths:
            results.append({'image_path': img_path, 'status': 'failed', 'error': str(e)})
        stats_tracker.add_batch_result(len(batch_image_paths), 0.1, 0, len(batch_image_paths))
    
    return results

def process_images_in_parallel(image_paths, pipe, stats_tracker, batch_size=BATCH_SIZE):
    """
    Processes images in parallel batches.
    """
    results = []
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i+batch_size]
        batch_results = process_batch(batch, pipe, stats_tracker)
        results.extend(batch_results)
        print(f"Batch {(i//batch_size)+1}/{(len(image_paths)+batch_size-1)//batch_size} complete. {stats_tracker.get_progress_string()}")
        if ((i//batch_size)+1) % 5 == 0 or i+batch_size >= len(image_paths):
            stats_file = "llava_processing_stats_progress.json"
            stats_tracker.save_stats_to_file(stats_file)
    return results

def main(image_paths, output_csv, batch_size=BATCH_SIZE):
    """
    Main function: initializes tracking, loads the model, processes images in batches,
    and writes results to CSV.
    """
    stats_tracker = StatsTracker(len(image_paths))
    
    print("="*50)
    print("LLAVA IMAGE PROCESSING")
    print("="*50)
    print(f"Images to process: {len(image_paths)}")
    print(f"Batch size: {batch_size}")
    print(f"Model: {MODEL_ID}")
    print(f"GPU count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
    print(f"Using model compilation: {COMPILE_MODEL}")
    print("="*50)
    
    pipe = create_optimized_pipeline()
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Image Path", "Entities & Relationships", "Scene Description", "Event Description", "Objects List", "Processing Time(s)"])
        
        try:
            results = process_images_in_parallel(image_paths, pipe, stats_tracker, batch_size)
            for result in results:
                if result['status'] == 'success':
                    writer.writerow([
                        result['image_path'],
                        result['entities_relationships'],
                        result['scene_description'],
                        result['event_description'],
                        result['objects_list'],
                        f"{result['duration']:.2f}"
                    ])
                else:
                    print(f"Failed to process {result['image_path']}: {result.get('error', 'Unknown error')}")
            stats_tracker.print_final_report()
        except KeyboardInterrupt:
            print("\nProcessing interrupted by user. Saving partial results...")
            stats_file = f"{os.path.splitext(output_csv)[0]}_stats_partial.json"
            stats_tracker.save_stats_to_file(stats_file)
            stats_tracker.print_final_report()
    
    stats_file = f"{os.path.splitext(output_csv)[0]}_stats_final.json"
    stats_tracker.save_stats_to_file(stats_file)
    print(f"Processing complete. Results saved to {output_csv}")
    print(f"Statistics saved to {stats_file}")

def optimize_batch_size():
    """
    Determines optimal batch size based on GPU memory.
    """
    if not torch.cuda.is_available():
        print("No GPU available, using default batch size")
        return BATCH_SIZE
    total_mem = torch.cuda.get_device_properties(0).total_memory
    total_mem_gb = total_mem / (1024**3)
    if total_mem_gb > 80:
        return 8
    elif total_mem_gb > 45:
        return 6
    elif total_mem_gb > 24:
        return 4
    elif total_mem_gb > 16:
        return 2
    else:
        return 1

if __name__ == '__main__':
    image_dir = 'images'
    if not os.path.exists(image_dir):
        print(f"Warning: Directory {image_dir} not found. Creating it...")
        os.makedirs(image_dir)
    
    image_paths = [os.path.join(image_dir, f) for f in sorted(os.listdir(image_dir)) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    if not image_paths:
        print("No images found. Using sample paths instead.")
        image_paths = [f'images/{i:05d}.jpg' for i in range(1, 11)]
    
    csv_file = 'structured_image_analysis_results.csv'
    torch.set_float32_matmul_precision('high')
    
    try:
        optimal_batch_size = optimize_batch_size()
        print(f"Auto-detected optimal batch size: {optimal_batch_size}")
        if optimal_batch_size != BATCH_SIZE:
            print(f"Consider setting BATCH_SIZE to {optimal_batch_size} for better performance")
    except Exception as e:
        print(f"Error determining optimal batch size: {e}")
    
    start_time = time.time()
    print(f"Starting processing at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    main(image_paths, csv_file, BATCH_SIZE)
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"Processing finished at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total runtime: {str(datetime.timedelta(seconds=int(duration)))}")


#!/usr/bin/env python
"""
gemma_analysis.py

This script processes images using the Gemma model in parallel, saving results to a CSV.
Uses optimized batch processing and better parallelization for H100 GPUs.
Includes comprehensive performance statistics tracking.
"""

import os
os.environ["HF_HUB_CACHE"] = "Model"
from transformers import pipeline
import torch
from torch.utils.data import Dataset, DataLoader
import csv
import re
import time
import datetime
import json
import gc
import psutil
import numpy as np

# Constants for optimization
BATCH_SIZE = 4  # Adjust based on GPU memory
MAX_NEW_TOKENS = 1000  # Reduced from 1200
COMPILE_MODEL = True
MODEL_ID = "unsloth/gemma-3-12b-it-bnb-4bit"

class ImageDataset(Dataset):
    """Dataset for batch processing of images"""
    def __init__(self, image_paths):
        self.image_paths = image_paths
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        return self.image_paths[idx]

class StatsTracker:
    """Class to track and report processing statistics"""
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
        """Update system resource statistics"""
        # CPU stats
        self.cpu_utilization.append(psutil.cpu_percent())
        self.ram_usage.append(psutil.virtual_memory().percent)
        
        # GPU stats if available
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                try:
                    self.gpu_utilization.append(torch.cuda.utilization(i))
                    self.gpu_memory_used.append(torch.cuda.memory_allocated(i) / torch.cuda.get_device_properties(i).total_memory * 100)
                except:
                    pass
    
    def add_batch_result(self, batch_size, batch_time, success_count, fail_count):
        """Record statistics for a completed batch"""
        self.batch_sizes.append(batch_size)
        self.batch_processing_times.append(batch_time)
        self.processed_images += batch_size
        self.success_count += success_count
        self.failed_count += fail_count
        self.update_resource_stats()
        
    def add_image_time(self, processing_time):
        """Record processing time for an individual image"""
        self.processing_times.append(processing_time)
    
    def get_progress_string(self):
        """Get a formatted progress string"""
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
        """Generate complete statistical summary"""
        elapsed = time.time() - self.start_time
        
        # Calculate image statistics
        if self.processing_times:
            avg_image_time = np.mean(self.processing_times)
            min_image_time = np.min(self.processing_times)
            max_image_time = np.max(self.processing_times)
            median_image_time = np.median(self.processing_times)
            p95_image_time = np.percentile(self.processing_times, 95)
        else:
            avg_image_time = min_image_time = max_image_time = median_image_time = p95_image_time = 0
        
        # Calculate batch statistics
        if self.batch_processing_times:
            avg_batch_time = np.mean(self.batch_processing_times)
            avg_batch_size = np.mean(self.batch_sizes)
            images_per_second = sum(self.batch_sizes) / sum(self.batch_processing_times)
        else:
            avg_batch_time = avg_batch_size = images_per_second = 0
        
        # Calculate resource utilization
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
        """Save complete statistics to a JSON file"""
        with open(filename, 'w') as f:
            json.dump(self.get_summary_stats(), f, indent=2)
            
    def print_final_report(self):
        """Print a formatted final report to console"""
        stats = self.get_summary_stats()
        img_stats = stats["image_processing_stats"]
        batch_stats = stats["batch_processing_stats"]
        res_stats = stats["resource_utilization"]
        
        print("\n" + "="*80)
        print(f"GEMMA IMAGE PROCESSING COMPLETE - PERFORMANCE REPORT")
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
    """Creates an optimized pipeline with better memory management"""
    # Clear CUDA cache before loading model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    print(f"Loading Gemma model with pipeline...")
    start_time = time.time()
    
    # Use the standard pipeline approach which works with Gemma models
    pipe = pipeline(
        "image-text-to-text", 
        model=MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Apply model compilation if enabled and available
    if COMPILE_MODEL and hasattr(torch, 'compile') and hasattr(pipe, 'model'):
        try:
            print("Compiling model for faster execution...")
            pipe.model = torch.compile(pipe.model, mode="reduce-overhead")
            print("Model compilation successful")
        except Exception as e:
            print(f"Model compilation failed, continuing without compilation: {e}")
    
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f} seconds.")
    
    # Print GPU information
    if torch.cuda.is_available():
        print(f"GPU Information:")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            mem_used = torch.cuda.memory_allocated(i) / 1024**3
            mem_total = props.total_memory / 1024**3
            print(f"GPU {i}: {props.name} - Memory: {mem_used:.2f}GB / {mem_total:.2f}GB")
    
    return pipe

def create_prompt(image_path):
    """Creates the prompt structure for a given image"""
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": """
            You are a multimodal visual analysis expert assisting in the creation of a structured academic dataset based on user-uploaded news photographs.

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

            **Ensure all sections are well-structured, labeled, and formatted as bullet points.**
                """}
            ]
        }
    ]

def extract_details(response_text):
    """Extracts structured sections from the response."""
    headers = [
        ("entities_relationships", r"### \*\*1\.\s*Entities & Relationships.*?\*\*"),
        ("scene_description", r"### \*\*2\.\s*Scene Description.*?\*\*"),
        ("event_description", r"### \*\*3\.\s*Event Description.*?\*\*"),
        ("objects_list", r"### \*\*4\.\s*Objects List.*?\*\*")
    ]
    header_starts = {key: re.search(pat, response_text).start() for key, pat in headers if re.search(pat, response_text)}
    header_ends = {key: re.search(pat, response_text).end() for key, pat in headers if re.search(pat, response_text)}
    sorted_keys = sorted(header_starts, key=header_starts.get)
    details = {}
    for i, key in enumerate(sorted_keys):
        start = header_ends[key]
        end = header_starts[sorted_keys[i + 1]] if i + 1 < len(sorted_keys) else None
        text = response_text[start:end].strip() if end else response_text[start:].strip()
        lines = [re.sub(r"^\*\s+", "", line) for line in text.split("\n") if line.strip()]
        details[key] = "\n".join(lines)
    return {key: details.get(key, "No data available") for key, _ in headers}

def remove_markdown_bolding(text):
    """Removes Markdown bolding from text."""
    return re.sub(r'\*\*(.*?)\*\*', r'\1', text)

def process_batch(batch_image_paths, pipe, stats_tracker):
    """Process a batch of images using the pipeline"""
    results = []
    batch_prompts = [create_prompt(img_path) for img_path in batch_image_paths]
    
    try:
        # Generate responses for the entire batch
        with torch.inference_mode():
            batch_start_time = time.time()
            batch_responses = pipe(
                batch_prompts,
                max_new_tokens=MAX_NEW_TOKENS,
                batch_size=BATCH_SIZE,
                do_sample=False,  # Deterministic generation is faster
            )
            batch_duration = time.time() - batch_start_time
        
        success_count = 0
        fail_count = 0
        
        # Process each response in the batch
        for i, (img_path, response) in enumerate(zip(batch_image_paths, batch_responses)):
            if response and "generated_text" in response[0] and len(response[0]["generated_text"]) > 0:
                analysis = response[0]["generated_text"][-1]["content"]
                analysis_data = extract_details(analysis)
                cleaned_data = {k: remove_markdown_bolding(v) for k, v in analysis_data.items()}
                
                # Calculate estimated time per image
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
                results.append({'image_path': img_path, 'status': 'failed'})
                fail_count += 1
        
        # Update batch statistics
        stats_tracker.add_batch_result(len(batch_image_paths), batch_duration, success_count, fail_count)
        
    except Exception as e:
        print(f"Error processing batch: {e}")
        # Return failed status for all images in the batch
        results.extend([{'image_path': img_path, 'status': 'failed'} for img_path in batch_image_paths])
        stats_tracker.add_batch_result(len(batch_image_paths), 0, 0, len(batch_image_paths))
    
    return results

def process_images_in_parallel(image_paths, pipe, stats_tracker, batch_size=BATCH_SIZE):
    """Process images in parallel with better error handling"""
    results = []
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i+batch_size]
        batch_results = process_batch(batch, pipe, stats_tracker)
        results.extend(batch_results)
        
        # Print progress update
        print(f"Batch {(i//batch_size)+1}/{(len(image_paths)+batch_size-1)//batch_size} complete. {stats_tracker.get_progress_string()}")
        
        # Periodically save stats
        if ((i//batch_size)+1) % 5 == 0 or i+batch_size >= len(image_paths):
            stats_file = f"gemma_processing_stats_progress.json"
            stats_tracker.save_stats_to_file(stats_file)
            
    return results

def main(image_paths, output_csv, batch_size=BATCH_SIZE):
    """Main function to process images in optimized batches"""
    # Initialize statistics tracker
    stats_tracker = StatsTracker(len(image_paths))
    
    # Print configuration information
    print(f"\n{'='*50}")
    print(f"GEMMA IMAGE PROCESSING")
    print(f"{'='*50}")
    print(f"Images to process: {len(image_paths)}")
    print(f"Batch size: {batch_size}")
    print(f"Model: {MODEL_ID}")
    print(f"GPU count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
    print(f"Using model compilation: {COMPILE_MODEL}")
    print(f"{'='*50}\n")
    
    # Initialize the pipeline
    pipe = create_optimized_pipeline()
    
    # Process batches and write results
    with open(output_csv, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Image Path", "Entities & Relationships", "Scene Description", "Event Description", "Objects List", "Processing Time(s)"])
        
        try:
            # Process all images
            results = process_images_in_parallel(image_paths, pipe, stats_tracker, batch_size)
            
            # Write successful results to CSV
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
            
            # Process complete - print final report
            stats_tracker.print_final_report()
            
        except KeyboardInterrupt:
            print("\nProcessing interrupted by user. Saving partial results...")
            stats_file = f"{os.path.splitext(output_csv)[0]}_stats_partial.json"
            stats_tracker.save_stats_to_file(stats_file)
            stats_tracker.print_final_report()
    
    # Save final statistics
    stats_file = f"{os.path.splitext(output_csv)[0]}_stats_final.json"
    stats_tracker.save_stats_to_file(stats_file)
    
    print(f"Processing complete. Results saved to {output_csv}")
    print(f"Statistics saved to {stats_file}")

def optimize_batch_size():
    """Function to determine optimal batch size for the GPUs"""
    if not torch.cuda.is_available():
        print("No GPU available, using default batch size")
        return BATCH_SIZE
        
    # Get GPU memory info
    # For simplicity, we'll check the first GPU
    total_mem = torch.cuda.get_device_properties(0).total_memory
    total_mem_gb = total_mem / (1024**3)
    
    # Very rough heuristic - adjust based on model
    if total_mem_gb > 45:  # H100 likely has >45GB memory
        # For large GPUs, try larger batch sizes
        return 8
    elif total_mem_gb > 35:
        return 6
    elif total_mem_gb > 24:
        return 4
    elif total_mem_gb > 16:
        return 2
    else:
        return 1

if __name__ == '__main__':
    # Example image paths (adjust as needed)
    image_dir = 'images'
    
    # Check if directory exists
    if not os.path.exists(image_dir):
        print(f"Warning: Directory {image_dir} not found. Creating it...")
        os.makedirs(image_dir)
        
    # Get all image files from directory
    image_paths = [os.path.join(image_dir, f) for f in sorted(os.listdir(image_dir)) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    # Alternatively, use your original method
    if not image_paths:
        print("No images found. Using sample paths instead.")
        image_paths = [f'images/{i:05d}.jpg' for i in range(1, 11)]
    
    csv_file = 'structured_image_analysis_results.csv'
    
    # Set torch settings for maximum performance
    torch.set_float32_matmul_precision('high')
    
    # Try to optimize batch size based on GPU memory
    try:
        optimal_batch_size = optimize_batch_size()
        print(f"Auto-detected optimal batch size: {optimal_batch_size}")
        if optimal_batch_size != BATCH_SIZE:
            print(f"Consider setting BATCH_SIZE to {optimal_batch_size} for better performance")
    except Exception as e:
        print(f"Error determining optimal batch size: {e}")
    
    # Show start time
    start_time = time.time()
    print(f"Starting processing at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run the optimized processing
    main(image_paths, csv_file, BATCH_SIZE)
    
    # Show end time and total duration
    end_time = time.time()
    duration = end_time - start_time
    print(f"Processing finished at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total runtime: {str(datetime.timedelta(seconds=int(duration)))}")
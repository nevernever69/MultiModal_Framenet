#!/usr/bin/env python
"""
gemma_analysis.py

This script processes images using the Gemma model in parallel across multiple GPUs, saving results to a CSV.
It uses the fast image processor for improved performance and includes timing and progress stats.
"""

import os
os.environ["HF_HUB_CACHE"] = "Model"
from transformers import pipeline
import torch
import csv
import re
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np

# This will be used to distribute work across GPUs
def get_gpu_id(worker_id, num_gpus):
    """
    Maps a worker ID to a specific GPU ID.
    
    Args:
        worker_id (int): The worker's ID
        num_gpus (int): Total number of available GPUs
        
    Returns:
        int: The GPU ID to use
    """
    return worker_id % num_gpus

# Global pipeline variable, initialized in each worker
pipe = None

def initialize_worker(worker_id, num_gpus):
    """
    Initialize the worker with the correct GPU assignment.
    
    Args:
        worker_id (int): The ID of this worker
        num_gpus (int): Total number of available GPUs
    """
    global pipe
    gpu_id = get_gpu_id(worker_id, num_gpus)
    print(f"Worker {worker_id} assigned to GPU {gpu_id}")
    
    # Set the visible device for this process
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Load the pipeline with device mapping
    device = f"cuda:{0}"  # Always use cuda:0 since we've set CUDA_VISIBLE_DEVICES
    pipe = pipeline(
        "image-text-to-text", 
        model="unsloth/gemma-3-12b-it-bnb-4bit", 
        torch_dtype=torch.bfloat16, 
        use_fast=True,
        device=device
    )
    
    # Optional: Compile model for better performance if supported
    try:
        pipe.model = torch.compile(pipe.model)
        print(f"Worker {worker_id} on GPU {gpu_id}: Model successfully compiled")
    except Exception as e:
        print(f"Worker {worker_id} on GPU {gpu_id}: Model compilation skipped - {str(e)}")
    
    return worker_id

def analyze_image(image_path, pipe):
    """
    Analyzes an image using the Gemma model and returns the assistant's text response.

    Args:
        image_path (str): Path to the image file.
        pipe: The loaded Transformers pipeline for image-text-to-text tasks.

    Returns:
        str: Generated text response or an empty string if an error occurs.
    """
    # Define the prompt as a list of messages
    messages = [
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

    try:
        # Generate response from the pipeline
        response = pipe(
            text=messages,
            max_new_tokens=1200  # Limit the output tokens
        )
        # Extract the assistant's response text
        if response and "generated_text" in response[0] and len(response[0]["generated_text"]) > 0:
            return response[0]["generated_text"][-1]["content"]
        else:
            return ""
    except Exception as e:
        print(f"Error analyzing image {image_path}: {e}")
        return ""

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

def process_image(args):
    """
    Processes an image and returns structured data with timing.
    
    Args:
        args (tuple): (image_path, worker_id)
    """
    image_path, worker_id = args
    
    try:
        # The pipeline should already be initialized by initialize_worker
        start_time = time.time()
        analysis = analyze_image(image_path, pipe)
        if analysis and analysis.strip():
            analysis_data = extract_details(analysis)
            cleaned_data = {k: remove_markdown_bolding(v) for k, v in analysis_data.items()}
            duration = time.time() - start_time
            return {
                'image_path': image_path,
                'status': 'success',
                'entities_relationships': cleaned_data["entities_relationships"],
                'scene_description': cleaned_data["scene_description"],
                'event_description': cleaned_data["event_description"],
                'objects_list': cleaned_data["objects_list"],
                'duration': duration,
                'worker_id': worker_id
            }
        return {'image_path': image_path, 'status': 'failed', 'worker_id': worker_id}
    except Exception as e:
        print(f"Worker {worker_id}: Error processing {image_path}: {e}")
        return {'image_path': image_path, 'status': 'failed', 'worker_id': worker_id}

def initializer(worker_id, num_gpus):
    """Wrapper function to pass arguments to initialize_worker"""
    return initialize_worker(worker_id, num_gpus)

if __name__ == '__main__':
    # Configuration
    num_gpus = 2  # Number of H100 GPUs available
    num_workers = num_gpus * 1  # Typically 1 worker per GPU for large models like Gemma
    
    # Example image paths (adjust as needed)
    image_paths = [f'images/pipe.model{i:05d}.jpg' for i in range(1, 13001)]  # For 13k+ images
    csv_file = 'structured_image_analysis_results.csv'
    
    total_images = len(image_paths)
    processed_count = 0
    
    # Prepare worker arguments - each worker gets its own ID
    worker_args = [(i, num_gpus) for i in range(num_workers)]
    
    # Distribute images across workers
    image_worker_pairs = []
    for i, img_path in enumerate(image_paths):
        worker_id = i % num_workers
        image_worker_pairs.append((img_path, worker_id))
    
    with ProcessPoolExecutor(max_workers=num_workers, initializer=initializer, initargs=worker_args) as executor:
        with open(csv_file, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["Image Path", "Entities & Relationships", "Scene Description", "Event Description", "Objects List", "Processing Time (s)", "Worker ID", "GPU ID"])
            
            # Process images and gather results
            for result in executor.map(process_image, image_worker_pairs):
                if result['status'] == 'success':
                    writer.writerow([
                        result['image_path'],
                        result['entities_relationships'],
                        result['scene_description'],
                        result['event_description'],
                        result['objects_list'],
                        f"{result['duration']:.2f}",
                        result['worker_id'],
                        get_gpu_id(result['worker_id'], num_gpus)
                    ])
                    processed_count += 1
                    
                    # Print progress update
                    if processed_count % 10 == 0 or processed_count == total_images:
                        print(f"Progress: {processed_count}/{total_images} images ({processed_count/total_images*100:.1f}%)")
                        print(f"Last processed: {result['image_path']} by Worker {result['worker_id']} on GPU {get_gpu_id(result['worker_id'], num_gpus)} in {result['duration']:.2f} seconds")
                else:
                    print(f"Failed to process {result['image_path']} on Worker {result['worker_id']}")
    
    print(f"Processing complete. {processed_count} images successfully processed.")
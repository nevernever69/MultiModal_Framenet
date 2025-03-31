#!/usr/bin/env python
"""
gemma_analysis.py

This script processes images using the Gemma model in parallel, saving results to a CSV.
It uses the fast image processor for improved performance and includes timing and progress stats.
"""

import os
os.environ["HF_HUB_CACHE"] = "Model"
from transformers import pipeline
import torch
import csv
import re
from multiprocessing import Pool
import time

# Global pipeline variable, initialized in each worker
pipe = None
worker_device = None  # Global variable to hold the GPU id for this worker

def init_worker(gpu_id):
    """Initializer for each pool worker to set the global GPU device."""
    global worker_device, pipe
    worker_device = gpu_id
    pipe = None  # Ensure pipeline is re-initialized per worker

def analyze_image(image_path, pipe):
    """
    Analyzes an image using the Gemma model and returns the assistant's text response.
    """
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

            ### **2. Scene Description (English):**

            - Write a rich, detailed description of the scene.
            - Include spatial relationships, background elements, and general setting/context relevant to identifying and localizing entities.
            - Be objective and avoid culturally biased interpretations.
            - Do not infer names or identities unless explicitly shown in the image (e.g., "man in red shirt" instead of "John").
            
            ### **3. Scene Description (Brazilian Portuguese):**

            - Provide a Brazilian Portuguese translation of the scene description.
            - The translation should maintain all entity labels (e.g., [person], [vehicle], [text]) unchanged.
            - The content should mirror the English version in detail and context.


            ### **4. Event Description:**

            - Describe the main activity or event depicted.
            - If ambiguous, provide up to three plausible interpretations, clearly separated (e.g., "Possibility 1: ...", "Possibility 2: ...").
            - Focus on the purpose, intent, or core action of the image.


            ### **5. Objects List:**

            - Provide an explicit, exhaustive list of objects detected in the image without tags.
            - Ensure all objects referenced in 'Entities & Relationships' are listed here.

            **Ensure all sections are well-structured, labeled, and formatted as bullet points.**                
            """}
            ]
        }
    ]
    try:
        response = pipe(
            text=messages,
            max_new_tokens=1200
        )
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
        ("scene_description_eng", r"### \*\*2\.\s*Scene Description \(English\).*?\*\*"),
        ("scene_description_pt", r"### \*\*3\.\s*Scene Description \(Brazilian Portuguese\).*?\*\*"),
        ("event_description", r"### \*\*4\.\s*Event Description.*?\*\*"),
        ("objects_list", r"### \*\*5\.\s*Objects List.*?\*\*")
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


def process_image(task):
    """
    Processes an image on the designated GPU and returns structured data with timing.
    task is a tuple: (image_path, gpu_id)
    """
    image_path, gpu_id = task
    global pipe

    # Set the visible GPU for this process so that only the desired GPU is used.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    torch.cuda.set_device(0)  # In the context of this process, the chosen GPU is now device 0

    if pipe is None:
        # Load the pipeline without the device argument
        pipe = pipeline(
            "image-text-to-text",
            model="unsloth/gemma-3-12b-it-bnb-4bit",
            torch_dtype=torch.bfloat16,
            use_fast=True
        )
        # Optionally compile the model if supported
        pipe.model = torch.compile(pipe.model)
    
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
            'scene_description_eng': cleaned_data["scene_description_eng"],
            'scene_description_pt': cleaned_data["scene_description_pt"],
            'event_description': cleaned_data["event_description"],
            'objects_list': cleaned_data["objects_list"],
            'duration': duration,
            'gpu': gpu_id
        }
    return {'image_path': image_path, 'status': 'failed', 'gpu': gpu_id}

if __name__ == '__main__':
    # Generate list of image paths; adjust as needed
    image_paths = [f'images/pipe.model{i:05d}.jpg' for i in range(1, 500)]
    
    # Create tasks as tuples: (image_path, gpu_id) using round-robin assignment.
    tasks = [(img, i % 2) for i, img in enumerate(image_paths)]
    
    csv_file = 'structured_image_analysis_results.csv'
    total_images = len(tasks)
    counter = 0

    # Create a pool with 2 workers.
    with Pool(processes=2) as pool:
        with open(csv_file, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([
                "Image Path", 
                "GPU", 
                "Entities & Relationships", 
                "Scene Description (English)", 
                "Scene Description (Brazilian Portuguese)", 
                "Event Description", 
                "Objects List"
            ])
            for result in pool.imap_unordered(process_image, tasks):
                if result['status'] == 'success':
                    writer.writerow([
                        result['image_path'],
                        result['gpu'],
                        result['entities_relationships'],
                        result['scene_description_eng'],
                        result['scene_description_pt'],
                        result['event_description'],
                        result['objects_list']
                    ])
                    counter += 1
                    print(f"Processed {result['image_path']} on GPU {result['gpu']} in {result['duration']:.2f} seconds. Total processed: {counter}/{total_images}")
                else:
                    print(f"Failed to process {result['image_path']} on GPU {result['gpu']}")

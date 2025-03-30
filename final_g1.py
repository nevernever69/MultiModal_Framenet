#!/usr/bin/env python
"""
gemma_analysis_optimized.py

This script loads the Gemma model pipeline and processes images in true batches,
properly optimized for high-performance GPUs like H100, extracting structured details and
saving the results to a CSV file.
"""
import os
os.environ["HF_HUB_CACHE"] = "Model"
from transformers import pipeline, AutoConfig, AutoProcessor
import torch
import csv
import os
import re
import time
import glob
from functools import partial
import numpy as np
from tqdm import tqdm
from datasets import Dataset

def setup_pipeline():
    """
    Sets up an optimized pipeline for H100 GPU with maximum efficiency.
    """
    # Configure for maximum performance on H100
    config = AutoConfig.from_pretrained("unsloth/gemma-3-12b-it-bnb-4bit")
    
    # Load the Gemma model pipeline with optimized settings for H100
    pipe = pipeline(
        "image-text-to-text",
        model="unsloth/gemma-3-12b-it-bnb-4bit",
        torch_dtype=torch.bfloat16,
        device_map="auto",  # Automatically use all available GPUs
        config=config
    )
    
    # Optimize attention computation
    if hasattr(pipe.model, "config"):
        pipe.model.config.use_flash_attention_2 = True  # Use flash attention if available
    
    # Enable better batching
    if hasattr(pipe, "tokenizer"):
        pipe.tokenizer.pad_token = pipe.tokenizer.eos_token
        pipe.tokenizer.padding_side = "left"
    
    return pipe

def create_prompt_text():
    """
    Creates the text portion of the prompt message for the Gemma model.
    """
    return """
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
    """

def collect_image_paths(base_dir='/content/images', use_sequential=True):
    """
    Collects valid image paths in sequential numerical order.
    
    Args:
        base_dir (str): Base directory to search for images.
        use_sequential (bool): Whether to use sequential file naming pattern.
        
    Returns:
        list: List of valid image paths in numerical order.
    """
    valid_paths = []
    
    if use_sequential:
        # Try sequential pattern (00001.jpg, 00002.jpg, etc.)
        for i in range(1, 1000):  # Adjust range as needed
            path = f'{base_dir}/{i:05d}.jpg'
            if os.path.exists(path):
                valid_paths.append(path)
    else:
        # Use glob pattern as fallback
        valid_paths = sorted(glob.glob(os.path.join(base_dir, '*.jpg')))
    
    # Print summary of found images
    if valid_paths:
        print(f"Found {len(valid_paths)} images in {base_dir}")
        print(f"First few images: {valid_paths[:5]}")
    else:
        print(f"No images found in {base_dir}")
    
    return valid_paths

def extract_details(response_text):
    """
    Extracts structured sections from the analysis response.
    """
    # Define section headers and their keys
    headers = [
        ("entities_relationships", r"### \*\*1\.\s*Entities & Relationships.*?\*\*"),
        ("scene_description", r"### \*\*2\.\s*Scene Description.*?\*\*"),
        ("event_description", r"### \*\*3\.\s*Event Description.*?\*\*"),
        ("objects_list", r"### \*\*4\.\s*Objects List.*?\*\*")
    ]

    header_starts = {}
    header_ends = {}
    for key, pattern in headers:
        match = re.search(pattern, response_text)
        if match:
            header_starts[key] = match.start()
            header_ends[key] = match.end()

    # Sort sections by their starting position
    sorted_keys = sorted(header_starts, key=header_starts.get)
    details = {}
    for i, key in enumerate(sorted_keys):
        start = header_ends[key]
        end = header_starts[sorted_keys[i + 1]] if i + 1 < len(sorted_keys) else None
        section_text = response_text[start:end].strip() if end else response_text[start:].strip()

        if key == "scene_description":
            cleaned_text = section_text
        else:
            lines = section_text.split("\n")
            cleaned_lines = [re.sub(r"^\*\s+", "", line) for line in lines if line.strip()]
            if key == "event_description":
                cleaned_lines = [re.sub(r"\*\*(Possibility \d+:)\*\*", r"\1", line) for line in cleaned_lines]
            cleaned_text = "\n".join(cleaned_lines)

        details[key] = cleaned_text

    # Ensure all sections exist
    for key, _ in headers:
        if key not in details:
            details[key] = "No data available"

    return details

def remove_markdown_bolding(text):
    """
    Removes Markdown bolding from the input text.
    """
    return re.sub(r'\*\*(.*?)\*\*', r'\1', text)

def preprocess_images(image_paths):
    """
    Prepare a dataset from the image paths for efficient batch processing.
    
    Args:
        image_paths (list): List of paths to image files
        
    Returns:
        datasets.Dataset: A HuggingFace dataset ready for batch processing
    """
    prompt_text = create_prompt_text()
    
    # Create a list of processed examples
    examples = []
    for path in image_paths:
        examples.append({
            "image_path": path,
            "text": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": path},
                        {"type": "text", "text": prompt_text}
                    ]
                }
            ]
        })
    
    # Create a HuggingFace dataset for efficient batching
    return Dataset.from_list(examples)

def process_batch(batch, pipe):
    """
    Process a batch of images in parallel.
    
    Args:
        batch (dict): Batch of inputs including image paths and prompts
        pipe: The loaded Transformers pipeline
        
    Returns:
        dict: Dictionary with processing results
    """
    start_time = time.time()
    results = []
    
    try:
        # Extract required data from batch
        image_paths = batch["image_path"]
        texts = batch["text"]
        
        # Process entire batch at once
        responses = pipe(text=texts, max_new_tokens=1200)
        
        # Process each response in the batch
        for i, (path, response) in enumerate(zip(image_paths, responses)):
            try:
                if response and "generated_text" in response and len(response["generated_text"]) > 0:
                    response_text = response["generated_text"][-1]["content"]
                    analysis_data = extract_details(response_text)
                    cleaned_data = {key: remove_markdown_bolding(value) for key, value in analysis_data.items()}
                    
                    results.append({
                        "image_path": path,
                        "entities_relationships": cleaned_data["entities_relationships"],
                        "scene_description": cleaned_data["scene_description"],
                        "event_description": cleaned_data["event_description"],
                        "objects_list": cleaned_data["objects_list"],
                        "processing_time": time.time() - start_time,
                        "status": "success"
                    })
                else:
                    results.append({
                        "image_path": path,
                        "entities_relationships": "No data available",
                        "scene_description": "No data available",
                        "event_description": "No data available",
                        "objects_list": "No data available",
                        "processing_time": time.time() - start_time,
                        "status": "empty_response"
                    })
            except Exception as e:
                results.append({
                    "image_path": path,
                    "entities_relationships": f"Error: {str(e)}",
                    "scene_description": f"Error: {str(e)}",
                    "event_description": f"Error: {str(e)}",
                    "objects_list": f"Error: {str(e)}",
                    "processing_time": time.time() - start_time,
                    "status": f"error: {str(e)}"
                })
    except Exception as batch_error:
        # Handle batch-level errors
        for path in image_paths:
            results.append({
                "image_path": path,
                "entities_relationships": f"Batch Error: {str(batch_error)}",
                "scene_description": f"Batch Error: {str(batch_error)}",
                "event_description": f"Batch Error: {str(batch_error)}",
                "objects_list": f"Batch Error: {str(batch_error)}",
                "processing_time": time.time() - start_time,
                "status": f"batch_error: {str(batch_error)}"
            })
    
    return results

def main():
    # Parameters - adjust for H100 performance
    batch_size = 4  # Real batch size for GPU processing
    image_dir = '/content/images'  # Adjust to your image directory
    
    # Setup optimized pipeline for H100
    print("Setting up optimized pipeline for H100 GPU...")
    pipe = setup_pipeline()
    
    # Collect image paths in sequential order
    print("Collecting image paths...")
    image_paths = collect_image_paths(image_dir, use_sequential=True)
    
    if not image_paths:
        print("No valid images found. Exiting.")
        return
    
    print(f"Processing {len(image_paths)} images with batch size {batch_size}...")
    
    # Create dataset for efficient processing
    dataset = preprocess_images(image_paths)
    
    # Use the dataset for efficient batched processing
    all_results = []
    total_start_time = time.time()
    
    # Process in batches with progress bar
    for i in tqdm(range(0, len(dataset), batch_size), desc="Processing batches"):
        batch_end = min(i + batch_size, len(dataset))
        batch = dataset[i:batch_end]
        
        # Start timer for this batch
        batch_start_time = time.time()
        
        # Process the batch
        batch_results = process_batch(batch, pipe)
        all_results.extend(batch_results)
        
        # Print batch processing time
        batch_time = time.time() - batch_start_time
        print(f"Batch {i//batch_size + 1}/{(len(dataset) + batch_size - 1)//batch_size} "
              f"processed in {batch_time:.2f}s ({batch_time/len(batch):.2f}s per image)")
    
    # Calculate total processing time
    total_time = time.time() - total_start_time
    
    # Define CSV output file path
    csv_file = 'structured_image_analysis_results.csv'
    
    # Write results to CSV
    with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Image Path", "Entities & Relationships", "Scene Description", "Event Description", "Objects List", "Processing Time (s)", "Status"])
        
        for result in all_results:
            writer.writerow([
                result["image_path"],
                result["entities_relationships"],
                result["scene_description"],
                result["event_description"],
                result["objects_list"],
                f"{result['processing_time']:.2f}",
                result["status"]
            ])
    
    # Calculate and print statistics
    successful_results = [r for r in all_results if r["status"] == "success"]
    processing_times = [r["processing_time"] for r in successful_results]
    
    print("\n===== Performance Statistics =====")
    print(f"Total images processed: {len(all_results)}")
    print(f"Successfully processed: {len(successful_results)}")
    if processing_times:
        print(f"Average processing time: {np.mean(processing_times):.2f}s")
        print(f"Median processing time: {np.median(processing_times):.2f}s")
        print(f"Min processing time: {min(processing_times):.2f}s")
        print(f"Max processing time: {max(processing_times):.2f}s")
    print(f"Total wall time: {total_time:.2f}s")
    print(f"Theoretical throughput: {len(all_results)/total_time:.2f} images/second")
    print(f"Results saved to: {csv_file}")
    print("==================================")

if __name__ == '__main__':
    main()

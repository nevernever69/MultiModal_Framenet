#!/usr/bin/env python
"""
gemma_analysis.py

This script loads the Gemma model pipeline and processes a set of images in batches,
extracting structured details and saving the results to a CSV file.
"""
import os
os.environ["HF_HUB_CACHE"] = "Model"
from transformers import pipeline
import torch
import csv
import os
import re
import time
from tqdm import tqdm

# Load the Gemma model pipeline with GPU acceleration and 4-bit quantization
pipe = pipeline(
    "image-text-to-text",
    model="unsloth/gemma-3-12b-it-bnb-4bit",  # Instruction-tuned, quantized model by Unsloth
    torch_dtype=torch.bfloat16,              # Optimize memory usage
    device_map="auto"                        # Automatically use available GPU(s)
)

def create_prompt(image_path):
    """
    Creates the prompt message structure for the Gemma model.
    
    Args:
        image_path (str): Path to the image file.
        
    Returns:
        list: A list containing the prompt structure.
    """
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

def collect_image_paths(base_dir='/content/images', pattern='*.jpg'):
    """
    Collects valid image paths.
    
    Args:
        base_dir (str): Base directory to search for images.
        pattern (str): Glob pattern to match image files.
        
    Returns:
        list: List of valid image paths.
    """
    import glob
    
    # Define the list of image paths to process
    # Try both the provided pattern and numbered pattern
    all_paths = glob.glob(os.path.join(base_dir, pattern))
    
    if not all_paths:
        # Try numbered pattern as a fallback
        all_paths = [f'{base_dir}/{i:05d}.jpg' for i in range(1, 1000)]
    
    # Filter out non-existent paths
    valid_paths = []
    for path in all_paths:
        if os.path.exists(path):
            valid_paths.append(path)
        else:
            # Only print for the numbered pattern to avoid excessive messages
            if '/' in path and path.split('/')[-1].isdigit():
                print(f"Image path {path} does not exist. Skipping.")
    
    return valid_paths

def extract_details(response_text):
    """
    Extracts structured sections from the analysis response.

    Args:
        response_text (str): The text response from the model.

    Returns:
        dict: Dictionary with keys for each section.
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
    Removes Markdown bolding (i.e., **text**) from the input text.
    
    Args:
        text (str): Text with potential markdown bolding.
        
    Returns:
        str: Text with bolding removed.
    """
    return re.sub(r'\*\*(.*?)\*\*', r'\1', text)

def process_batch(image_paths, pipe, batch_size=2):
    """
    Process a batch of images using the pipeline.
    
    Args:
        image_paths (list): List of image paths to process.
        pipe: The loaded Transformers pipeline.
        batch_size (int): Size of batches to process.
        
    Returns:
        list: List of dictionaries with processing results.
    """
    results = []
    
    # Process in batches
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        prompts = [create_prompt(path) for path in batch_paths]
        
        print(f"Processing batch {i//batch_size + 1}/{(len(image_paths) + batch_size - 1)//batch_size} ({len(batch_paths)} images)...")
        
        try:
            # Process the batch
            start_time = time.time()
            batch_responses = pipe(text=prompts, max_new_tokens=1200)
            duration = time.time() - start_time
            
            # Process each response in the batch
            for j, path in enumerate(batch_paths):
                try:
                    response = batch_responses[j]
                    
                    if "generated_text" in response[0] and len(response[0]["generated_text"]) > 0:
                        response_text = response[0]["generated_text"][-1]["content"]
                        analysis_data = extract_details(response_text)
                        cleaned_data = {key: remove_markdown_bolding(value) for key, value in analysis_data.items()}
                        
                        results.append({
                            "image_path": path,
                            "entities_relationships": cleaned_data["entities_relationships"],
                            "scene_description": cleaned_data["scene_description"],
                            "event_description": cleaned_data["event_description"],
                            "objects_list": cleaned_data["objects_list"],
                            "processing_time": duration / len(batch_paths)
                        })
                        
                        print(f"  Processed {path} ({j+1}/{len(batch_paths)} in batch)")
                    else:
                        print(f"  Warning: No valid response for {path}. Skipping.")
                        results.append({
                            "image_path": path,
                            "entities_relationships": "No data available",
                            "scene_description": "No data available",
                            "event_description": "No data available",
                            "objects_list": "No data available",
                            "processing_time": duration / len(batch_paths)
                        })
                except Exception as e:
                    print(f"  Error processing image {path}: {e}")
                    results.append({
                        "image_path": path,
                        "entities_relationships": f"Error: {str(e)}",
                        "scene_description": f"Error: {str(e)}",
                        "event_description": f"Error: {str(e)}",
                        "objects_list": f"Error: {str(e)}",
                        "processing_time": duration / len(batch_paths)
                    })
        except Exception as e:
            print(f"Error processing batch: {e}")
            # Add error entries for all images in the failed batch
            for path in batch_paths:
                results.append({
                    "image_path": path,
                    "entities_relationships": f"Batch error: {str(e)}",
                    "scene_description": f"Batch error: {str(e)}",
                    "event_description": f"Batch error: {str(e)}",
                    "objects_list": f"Batch error: {str(e)}",
                    "processing_time": 0
                })
    
    return results

def main():
    # Parameters
    batch_size = 2  # Adjust based on GPU memory and performance testing
    image_dir = '/content/images'  # Adjust to your image directory
    
    # Collect image paths
    image_paths = collect_image_paths(image_dir)
    
    if len(image_paths) == 0:
        print("No valid images found. Exiting.")
        return
    
    print(f"Found {len(image_paths)} valid images.")
    print(f"Processing with batch size of {batch_size}...")
    
    # Process images in batches
    results = process_batch(image_paths, pipe, batch_size)
    
    # Define CSV output file path
    csv_file = 'structured_image_analysis_results.csv'
    
    # Write results to CSV
    with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Image Path", "Entities & Relationships", "Scene Description", "Event Description", "Objects List", "Processing Time (s)"])
        
        for result in results:
            writer.writerow([
                result["image_path"],
                result["entities_relationships"],
                result["scene_description"],
                result["event_description"],
                result["objects_list"],
                f"{result['processing_time']:.2f}"
            ])
    
    # Calculate and print statistics
    successful_times = [result["processing_time"] for result in results if result["processing_time"] > 0]
    avg_time = sum(successful_times) / len(successful_times) if successful_times else 0
    
    print(f"\nAnalysis complete. Results saved to {csv_file}")
    print(f"Processed {len(results)} images")
    print(f"Successfully processed {len(successful_times)} images")
    print(f"Average processing time per image: {avg_time:.2f} seconds")
    print(f"Total processing time: {sum(successful_times):.2f} seconds")

if __name__ == '__main__':
    main()

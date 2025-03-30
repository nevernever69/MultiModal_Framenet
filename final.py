#!/usr/bin/env python
"""
gemma_analysis.py

This script loads the Gemma model pipeline and processes a set of images,
extracting structured details and saving the results to a CSV file using multiprocessing
for efficient parallel processing on an H100 GPU.
"""

import os
os.environ["HF_HUB_CACHE"] = "Model"
from transformers import pipeline
import torch
import csv
import re
from multiprocessing import Pool

# Global variable for the pipeline, initialized as None in each worker process
pipe = None

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
    """
    Extracts structured sections from the analysis response.

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
        text (str): Input text with possible Markdown bolding.

    Returns:
        str: Text with bolding removed.
    """
    return re.sub(r'\*\*(.*?)\*\*', r'\1', text)

def process_image(image_path):
    """
    Processes a single image by loading the model (if not loaded), analyzing the image,
    extracting details, and returning structured data.

    Args:
        image_path (str): Path to the image file.

    Returns:
        dict or None: Structured data dictionary or None if processing fails.
    """
    global pipe
    try:
        # Load the pipeline only once per worker process
        if pipe is None:
            pipe = pipeline("image-text-to-text", model="unsloth/gemma-3-12b-it-bnb-4bit", torch_dtype=torch.bfloat16)
            # Optional: Compile the model for speed (comment out if not supported)
            pipe.model = torch.compile(pipe.model)
        # Analyze the image
        analysis = analyze_image(image_path, pipe)
        if not analysis or analysis.strip() == "":
            return None
        # Extract structured details
        analysis_data = extract_details(analysis)
        # Clean the data by removing Markdown bolding
        cleaned_data = {key: remove_markdown_bolding(value) for key, value in analysis_data.items()}
        return {
            'image_path': image_path,
            'entities_relationships': cleaned_data["entities_relationships"],
            'scene_description': cleaned_data["scene_description"],
            'event_description': cleaned_data["event_description"],
            'objects_list': cleaned_data["objects_list"]
        }
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

if __name__ == '__main__':
    # List of image paths (adjust range and path as needed)
    image_paths = [f'/content/images/{i:05d}.jpg' for i in range(1, 13001)]
    csv_file = 'structured_image_analysis_results.csv'
    num_workers = 10  # Number of worker processes; adjust based on GPU memory

    # Create a pool of worker processes and process images in parallel
    with Pool(num_workers) as pool:
        with open(csv_file, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            # Write CSV header
            writer.writerow(["Image Path", "Entities & Relationships", "Scene Description", "Event Description", "Objects List"])
            # Process images and write results as they become available
            for result in pool.imap_unordered(process_image, image_paths):
                if result:
                    writer.writerow([
                        result['image_path'],
                        result['entities_relationships'],
                        result['scene_description'],
                        result['event_description'],
                        result['objects_list']
                    ])

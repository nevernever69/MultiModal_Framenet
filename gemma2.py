#!/usr/bin/env python
"""
gemma_analysis_batch.py

This script loads the Gemma model pipeline with GPU acceleration and 4-bit quantization,
processes a set of news images in batches, extracts structured details, and saves the
results to a CSV file.
"""

import os
import csv
import re
import time
from PIL import Image
import torch
from transformers import pipeline
from torch.utils.data import Dataset, DataLoader

# Set cache location for model files
os.environ["HF_HUB_CACHE"] = "Model"

# Load the Gemma model pipeline with GPU acceleration and 4-bit quantization
pipe = pipeline(
    "image-text-to-text",
    model="unsloth/gemma-3-12b-it-bnb-4bit",  # Instruction-tuned, quantized model by Unsloth
    torch_dtype=torch.bfloat16
)

# Custom dataset for loading news images (returning PIL Images)
class NewsImageDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_files = sorted(
            [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith('.jpg')]
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        # Return the PIL Image directly
        image = Image.open(image_path).convert("RGB")
        return image, image_path

# Custom collate function that simply returns lists of PIL images and image paths.
def custom_collate_fn(batch):
    images, image_paths = zip(*batch)
    return list(images), list(image_paths)

# Create dataset and dataloader for batch processing.
dataset = NewsImageDataset('images')
dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4, collate_fn=custom_collate_fn)

# Analysis prompt text remains the same.
PROMPT_TEXT = """
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

def analyze_images_batch(batch, pipe):
    """
    Processes a batch of images with their associated paths.
    Each image is wrapped in the prompt messages and passed to the pipeline.
    Returns a list of tuples (image_path, analysis_text).
    """
    responses = []
    messages_batch = []
    image_paths = []
    
    images, paths = batch
    for image, image_path in zip(images, paths):
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT_TEXT}
            ]
        }]
        messages_batch.append(messages)
        image_paths.append(image_path)
    
    try:
        # Process the batch. The pipeline should now accept PIL images.
        batch_responses = pipe(text=messages_batch, max_new_tokens=1200)
    except Exception as e:
        print(f"Error processing batch: {e}")
        return []
    
    for resp, image_path in zip(batch_responses, image_paths):
        try:
            analysis_text = resp[0]["generated_text"][-1]["content"]
        except Exception as e:
            print(f"Error extracting response for {image_path}: {e}")
            analysis_text = ""
        responses.append((image_path, analysis_text))
    return responses

def extract_details(response_text):
    """
    Extracts structured sections from the analysis response.
    Returns a dictionary with keys for each section.
    """
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

    for key, _ in headers:
        if key not in details:
            details[key] = "No data available"

    return details

def remove_markdown_bolding(text):
    """
    Removes Markdown bolding (e.g., **text**) from the input text.
    """
    return re.sub(r'\*\*(.*?)\*\*', r'\1', text)

def main():
    csv_file = 'structured_image_analysis_results.csv'
    
    with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Image Path", "Entities & Relationships", "Scene Description", "Event Description", "Objects List"])
        
        for batch in dataloader:
            # Use the custom collate function output: (list of PIL images, list of image paths)
            batch_results = analyze_images_batch(batch, pipe)
            for image_path, analysis in batch_results:
                if not analysis or analysis.strip() == "":
                    print(f"Warning: No response for {image_path}. Skipping.")
                    continue
                try:
                    analysis_data = extract_details(analysis)
                    cleaned_data = {key: remove_markdown_bolding(value) for key, value in analysis_data.items()}
                    writer.writerow([
                        image_path,
                        cleaned_data["entities_relationships"],
                        cleaned_data["scene_description"],
                        cleaned_data["event_description"],
                        cleaned_data["objects_list"]
                    ])
                except Exception as e:
                    print(f"Error processing analysis for {image_path}: {e}")
    
    print(f"Analysis complete. Results saved to {csv_file}")

if __name__ == '__main__':
    main()


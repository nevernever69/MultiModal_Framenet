import os
import base64
import time
import torch
import pandas as pd
from PIL import Image
import io
import re
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# Configuration
MODEL_ID = "Salesforce/blip2-opt-2.7b"
BATCH_SIZE = 8  # Adjust based on GPU memory (e.g., 8 for two H100 GPUs with 80GB each)
COMPILE_MODEL = True  # Set to True to compile the model for faster inference
IMAGES_DIR = "images"  # Directory containing news photographs

# Prompt for BLIP-2 to generate structured output
PROMPT = """
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

### Helper Functions

def image_to_base64(image_path):
    """Convert an image file to a base64 string compatible with BLIP-2."""
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")  # Ensure image is in RGB format
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG")  # Save as JPEG for consistency
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"Error converting {image_path} to base64: {e}")
        return None

def get_image_paths(directory):
    """Collect paths of supported image files from the directory."""
    supported_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = [
        os.path.join(directory, f) for f in os.listdir(directory)
        if os.path.splitext(f.lower())[1] in supported_extensions
    ]
    return image_paths

def create_optimized_pipeline(model_id):
    """Load BLIP-2 model and processor, optimized for multi-GPU usage."""
    processor = Blip2Processor.from_pretrained(model_id)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    if COMPILE_MODEL and hasattr(torch, "compile"):
        model = torch.compile(model)  # Compile for faster inference
    return processor, model

def parse_response(response):
    """Parse the model's output into structured sections using regex."""
    pattern = r'(\d+\.\s\w+[\w\s]+):\s*(.*?)(?=\d+\.\s\w+[\w\s]+:|$)'  # Match numbered sections
    matches = re.findall(pattern, response, re.DOTALL)
    
    parsed = {
        "entities_relationships": "",
        "scene_description": "",
        "event_description": "",
        "objects_list": ""
    }
    
    for match in matches:
        section_title, content = match[0].strip(), match[1].strip()
        if "Entities & Relationships" in section_title:
            parsed["entities_relationships"] = content
        elif "Scene Description" in section_title:
            parsed["scene_description"] = content
        elif "Event Description" in section_title:
            parsed["event_description"] = content
        elif "Objects List" in section_title:
            parsed["objects_list"] = content
    
    return parsed

def process_batch(image_paths, processor, model, prompt):
    """Process a batch of images and return parsed responses."""
    # Convert images to base64
    base64_images = [image_to_base64(path) for path in image_paths]
    base64_images = [img for img in base64_images if img is not None]  # Filter out failures
    
    if not base64_images:
        return []
    
    # Prepare inputs for the batch
    inputs = processor(
        images=[Image.open(io.BytesIO(base64.b64decode(img))) for img in base64_images],
        text=[prompt] * len(base64_images),
        return_tensors="pt",
        padding=True
    ).to("cuda", torch.bfloat16)
    
    # Generate outputs for the batch
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=300)
    
    # Decode and parse responses
    responses = processor.batch_decode(outputs, skip_special_tokens=True)
    parsed_responses = [parse_response(resp) for resp in responses]
    
    # Pair responses with original image paths (accounting for potential failures)
    return [{"image_path": path, **parsed} for path, parsed in zip(image_paths, parsed_responses)]

### Main Execution

def main():
    """Main function to process images, save to CSV, and display execution time."""
    # Check if images directory exists
    if not os.path.exists(IMAGES_DIR):
        print(f"Directory '{IMAGES_DIR}' not found.")
        return
    
    # Get image paths
    image_paths = get_image_paths(IMAGES_DIR)
    if not image_paths:
        print(f"No images found in '{IMAGES_DIR}'.")
        return
    
    print(f"Found {len(image_paths)} images to process.")
    
    # Load model and processor once
    print("Loading BLIP-2 model and processor...")
    processor, model = create_optimized_pipeline(MODEL_ID)
    
    # Process images in chunks
    results = []
    total_start_time = time.time()
    
    for i in range(0, len(image_paths), BATCH_SIZE):
        chunk_start_time = time.time()
        chunk_paths = image_paths[i:i + BATCH_SIZE]
        
        print(f"Processing chunk {i // BATCH_SIZE + 1} ({len(chunk_paths)} images)...")
        chunk_results = process_batch(chunk_paths, processor, model, PROMPT)
        results.extend(chunk_results)
        
        chunk_time = time.time() - chunk_start_time
        print(f"Chunk {i // BATCH_SIZE + 1} completed in {chunk_time:.2f} seconds.")
    
    # Save results to CSV
    df = pd.DataFrame(results, columns=[
        "image_path", "entities_relationships", "scene_description",
        "event_description", "objects_list"
    ])
    df.to_csv("output.csv", index=False)
    print("Results saved to 'output.csv'.")
    
    # Display total execution time
    total_time = time.time() - total_start_time
    print(f"Total execution time: {total_time:.2f} seconds for {len(image_paths)} images.")

if __name__ == "__main__":
    main()
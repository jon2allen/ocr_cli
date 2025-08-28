import argparse
import google.generativeai as genai
import os
from PIL import Image
import pyheif

def load_image(image_path):
    """
    Loads an image from a file, discerning the format by its extension.
    Supports JPG, PNG, and HEIC/HEIF file types. 
    """
    ext = os.path.splitext(image_path)[1].lower()

    if ext in ('.heic', '.heif'):
        try:
            heif_file = pyheif.read(image_path)
            img = Image.frombytes(
                heif_file.mode,
                heif_file.size,
                heif_file.data,
                "raw",
                heif_file.mode,
                heif_file.stride,
            )
            return img
        except Exception as e:
            raise IOError(f"Failed to process HEIC file: {e}")
    elif ext in ('.jpg', '.jpeg', '.png'):
        try:
            return Image.open(image_path)
        except Exception as e:
            raise IOError(f"Failed to open image file: {e}")
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def extract_text_with_gemini(image_path):
    """
    Extracts text from an image using the Gemini 1.5 Flash model.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Error: GEMINI_API_KEY environment variable not set."

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash-latest')

    try:
        img = load_image(image_path)
        
        prompt = "Extract all text from this receipt, including line items, prices, and the total. Present it in a structured format that is easy to read, like the original receipt. Identify key fields like subtotal, tax, and total. Do not include any other conversational text."

        response = model.generate_content([prompt, img])
        
        return response.text
    except FileNotFoundError:
        return "Error: Image file not found."
    except Exception as e:
        return f"An error occurred: {e}"

def main():
    parser = argparse.ArgumentParser(description="A CLI tool to extract text from an image using Gemini 1.5 Flash.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to the input image file.")
    parser.add_argument("--output", "-o", type=str, help="Path to save the output text file (optional).")

    args = parser.parse_args()

    extracted_text = extract_text_with_gemini(args.input)

    print(extracted_text)

    if args.output:
        try:
            with open(args.output, "w") as f:
                f.write(extracted_text)
            print(f"\nText successfully saved to {args.output}")
        except Exception as e:
            print(f"\nError saving file: {e}")

if __name__ == "__main__":
    main()

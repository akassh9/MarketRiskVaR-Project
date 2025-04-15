# scripts/analyze_report_vertex.py
import os
from dotenv import load_dotenv
from google.cloud import aiplatform
import vertexai # Import the top-level vertexai namespace
from vertexai.generative_models import GenerativeModel, Part, GenerationConfig
import sys # Import sys to potentially exit gracefully

# --- Configuration ---
load_dotenv() # Load environment variables from .env file

# --- Vertex AI Configuration ---
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "var-risk-456908") # Replace or set in .env
GCP_REGION = os.getenv("GCP_REGION", "us-central1")       # Replace or set in .env

# --- Model Configuration ---
# Keeping the user-specified model name.
# !! Warning: If 500 errors persist, change this to a known stable model like "gemini-1.5-flash-preview-0514" !!
MODEL_NAME = "gemini-2.0-flash-lite"

# --- Input File Configuration (GCS) ---
# FIX #2 Applied: Assuming 'ak-risk-project-reports' is the bucket name based on previous logs.
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "ak-risk-project-reports") # Set your bucket name correctly
# FIX #2 Applied: Setting filename directly, removing the duplicated path.
REPORT_FILENAME = "financial-stability-report-20241122.pdf" # The actual PDF filename directly inside the bucket

GCS_URI = f"gs://{GCS_BUCKET_NAME}/{REPORT_FILENAME}" # Should now correctly be gs://ak-risk-project-reports/ch2.pdf

# --- Define Local Directory for Saving Output ---
# This specifies where the LLM's raw text output will be saved locally.
# Assumes your 'scripts' folder is parallel to a 'reports' folder.
REPORT_DIR = "Risk Reports"

# --- Initialize Vertex AI ---
try:
    print(f"Initializing Vertex AI for project='{GCP_PROJECT_ID}', location='{GCP_REGION}'...")
    vertexai.init(project=GCP_PROJECT_ID, location=GCP_REGION) # Use vertexai.init
    print("Vertex AI initialized successfully.")
except Exception as e:
    print(f"Error initializing Vertex AI: {e}")
    sys.exit(1) # Exit if initialization fails

# --- Load the Generative Model ---
try:
    print(f"Loading model: {MODEL_NAME}...")
    # Set system instruction if needed (optional, can guide overall behavior)
    # system_instruction = "You are an expert financial risk analyst."
    # model = GenerativeModel(MODEL_NAME, system_instruction=system_instruction)
    model = GenerativeModel(MODEL_NAME)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model {MODEL_NAME}: {e}")
    print("Ensure the model name is correct and available in your region.")
    sys.exit(1)

# --- Prepare PDF Input ---
try:
    print(f"Preparing PDF input from GCS URI: {GCS_URI}")
    # Create a Part object directly from the GCS URI
    # Ensure the MIME type is correctly set for PDF
    pdf_file = Part.from_uri(GCS_URI, mime_type="application/pdf")
    print("PDF Part created successfully.")
except Exception as e:
    print(f"Error creating Part from GCS URI {GCS_URI}: {e}")
    print("Please ensure the GCS URI is correct, the bucket/file exists, and permissions are set.")
    sys.exit(1)

# --- Define the Prompt for the LLM ---
# This prompt is designed for the HYBRID approach using direct PDF input.
PROMPT_TEXT = f"""
Analyze the attached PDF document: "{REPORT_FILENAME}". This document appears to be a chapter or section related to financial stability or risk. Your goal is to identify potential market risks, economic downturns, or financial stability concerns discussed anywhere within this document.

For each distinct risk or concern you identify:
1.  Provide a concise title or name for the risk (e.g., "Elevated Inflation Risk", "Geopolitical Tensions Impacting Energy", "Commercial Real Estate Vulnerabilities").
2.  Write a brief summary (2-4 sentences) describing the concern as presented in the document. Include context like the perceived severity or potential impact if mentioned.
3.  Extract 1-2 key sentences or phrases directly from the document text that best illustrate this specific risk. Try to include page numbers or section references if easily identifiable in the text context.

Focus *only* on risks mentioned within the provided PDF document. Do not invent risks or bring in outside knowledge beyond interpreting the document. Scan the entire document for relevant information.

Format your output clearly for each identified risk. Example:

**Risk Title:** Example Risk Name
**Summary:** This is a brief summary based on the text found in the PDF. It explains the core concern mentioned.
**Supporting Text:** "Direct quote from the PDF illustrating the risk (potentially near page X)." ; "Another relevant quote if available."
--- [Separate risks with a dashed line like this] ---

Please proceed with the analysis of the attached PDF.
"""
print("Prompt text defined.")

# --- Analyze PDF with Vertex AI ---
llm_output = None # Initialize llm_output to None
generation_config = GenerationConfig(
    temperature=0.2, # Lower temperature for factual extraction
    max_output_tokens=8192, # Set a generous limit for the response text
)

print(f"\n--- Starting analysis of {REPORT_FILENAME} using {MODEL_NAME} ---")
# Combine the text prompt and the PDF file Part into a list for the model input
contents = [PROMPT_TEXT, pdf_file]

try:
    # Send prompt and PDF to Vertex AI Gemini model
    print("Sending request to the model (this may take some time for large PDFs)...")
    response = model.generate_content(
        contents, # Pass the list containing prompt text and PDF Part
        generation_config=generation_config,
        stream=False # Process response once it's complete
        )
    print("Model response received.")

    # Extract the text response
    if response and response.candidates and response.candidates[0].content.parts:
         llm_output = response.candidates[0].content.parts[0].text
         print(f"Analysis successful.")
         # Optional: Print first part of response for immediate feedback
         # print(f"--- Response Snippet ---\n{llm_output[:500]}...\n--------------------")
    else:
         # Handle cases like safety blocks or empty responses
         try:
             # Attempt to print the full response object for debugging if no text part
             print(f"Warning: Received potentially non-text or blocked response.")
             print(f"Full Response Object: {response}")
             llm_output = f"[[Warning: Received non-text or blocked response. Details: {response}]]"
         except Exception as dbg_e:
             print(f"Warning: Received empty or unexpected response structure. Error during debug print: {dbg_e}")
             llm_output = f"[[Error: Empty or unexpected response structure]]"


except Exception as e:
    print(f"Error during model generation call: {e}")
    llm_output = f"[[Error during model call: {e}]]" # Store error message for saving

# --- Save Raw Response ---
# FIX #1 Applied: Using os.path.basename to create a clean output filename
base_filename = os.path.basename(REPORT_FILENAME) # Gets 'ch2.pdf'
raw_output_filename = os.path.splitext(base_filename)[0] + "_llm_directPDF_raw_output.txt" # Creates 'ch2_llm_directPDF_raw_output.txt'
raw_output_filepath = os.path.join(REPORT_DIR, raw_output_filename) # Saves locally like '../reports/ch2_llm_directPDF_raw_output.txt'

try:
    # FIX #1 Applied: Ensure the output directory exists locally
    os.makedirs(REPORT_DIR, exist_ok=True)

    print(f"Attempting to save raw LLM output to: {raw_output_filepath}")
    with open(raw_output_filepath, 'w', encoding='utf-8') as f:
        if llm_output:
            f.write(llm_output) # Write the collected response or error message
        else:
            f.write("[[No response content generated or captured]]")
    print(f"Raw LLM response/error saved to: {raw_output_filepath}")
except Exception as e:
    print(f"Error saving raw LLM response file: {e}")


# --- Placeholder for next step: Processing the response ---
print("\n--- Next Step: Process the collected LLM response ---")
# If the LLM call was successful, you would now parse the content of the saved file
# (or the llm_output variable if you prefer) to extract the structured risk information.
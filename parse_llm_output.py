# parse_llm_output.py
import re
import os
import pandas as pd # Optional: If you want to save as CSV

def parse_risk_report_output(file_path):
    """
    Parses the consolidated LLM output file containing risk analyses.

    Args:
        file_path (str): The path to the consolidated text file.

    Returns:
        list[dict]: A list of dictionaries, where each dictionary
                    represents a risk with keys 'title', 'summary',
                    and 'supporting_text'. Returns an empty list if
                    the file cannot be read or is empty.
    """
    risks = []
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return risks

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return risks

    # Split the content into individual risk blocks based on the '---' separator
    # We also handle potential leading/trailing whitespace and filter empty blocks
    risk_blocks = [block.strip() for block in content.split('---') if block.strip()]

    if not risk_blocks:
        print("Warning: No risk blocks found in the file after splitting by '---'.")
        # Check if the very first line might be a risk without a preceding '---'
        # This handles cases where the file starts directly with "**Risk Title:**"
        if content.strip().startswith("**Risk Title:**"):
             risk_blocks = [content.strip()] # Treat the whole content as one block
        else:
             # Check for the specific introductory line from the first file
             intro_line = "Here's an analysis of the PDF, identifying potential market risks, economic downturns, or financial stability concerns:"
             if content.strip().startswith(intro_line):
                 print(f"Found introductory line: '{intro_line}'. Removing it.")
                 content_after_intro = content.strip()[len(intro_line):].strip()
                 # Try splitting again after removing the intro
                 risk_blocks = [block.strip() for block in content_after_intro.split('---') if block.strip()]
                 if not risk_blocks and content_after_intro.startswith("**Risk Title:**"):
                     risk_blocks = [content_after_intro] # Treat remaining as one block


    print(f"Found {len(risk_blocks)} potential risk blocks.")

    # Regular expressions to extract the fields
    # Using re.DOTALL so '.' matches newlines within summary/supporting text
    title_re = re.compile(r"^\*\*Risk Title:\*\*\s*(.*?)\s*\n", re.MULTILINE)
    summary_re = re.compile(r"\*\*Summary:\*\*\s*(.*?)\s*\n\*\*Supporting Text:\*\*", re.DOTALL)
    support_re = re.compile(r"\*\*Supporting Text:\*\*\s*(.*)", re.DOTALL)

    for i, block in enumerate(risk_blocks):
        title_match = title_re.search(block)
        summary_match = summary_re.search(block)
        support_match = support_re.search(block)

        title = title_match.group(1).strip() if title_match else f"Unknown Title {i+1}"
        summary = summary_match.group(1).strip() if summary_match else "Summary not found."
        supporting_text = support_match.group(1).strip() if support_match else "Supporting text not found."

        # Basic cleaning for quotes within supporting text if needed
        supporting_text = supporting_text.replace('."', '"').replace('".', '"') # Example cleaning

        risks.append({
            'title': title,
            'summary': summary,
            'supporting_text': supporting_text
        })
        # print(f"Parsed block {i+1}: Title='{title}'") # Uncomment for debug

    return risks

if __name__ == "__main__":
    # --- Configuration ---
    # Assuming this script is in the root, and reports are in 'Risk Reports'
    REPORT_DIR = "Risk Reports"
    INPUT_FILENAME = "consolidated_risk_reports.txt"
    OUTPUT_CSV_FILENAME = "parsed_risks.csv" # Optional CSV output file

    input_filepath = os.path.join(REPORT_DIR, INPUT_FILENAME)

    # --- Parse the File ---
    parsed_risks = parse_risk_report_output(input_filepath)

    # --- Output Results ---
    if parsed_risks:
        print(f"\nSuccessfully parsed {len(parsed_risks)} risks.")

        # Print the first few risks as an example
        print("\n--- Example Parsed Risks (First 3) ---")
        for i, risk in enumerate(parsed_risks[:3]):
            print(f"Risk {i+1}:")
            print(f"  Title: {risk['title']}")
            print(f"  Summary: {risk['summary'][:100]}...") # Print snippet
            print(f"  Supporting Text: {risk['supporting_text'][:100]}...") # Print snippet
            print("-" * 10)

        # --- Optional: Save to CSV using Pandas ---
        try:
            df = pd.DataFrame(parsed_risks)
            output_csv_filepath = os.path.join(REPORT_DIR, OUTPUT_CSV_FILENAME)
            df.to_csv(output_csv_filepath, index=False, encoding='utf-8')
            print(f"\nParsed risks saved to CSV: {output_csv_filepath}")
        except ImportError:
            print("\nPandas library not found. Skipping CSV export.")
            print("Install pandas (`pip install pandas`) to enable CSV export.")
        except Exception as e:
            print(f"\nError saving to CSV: {e}")

    else:
        print("\nNo risks were parsed from the file.")
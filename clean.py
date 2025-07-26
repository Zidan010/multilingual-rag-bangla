import re

def clean_ocr_text(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Step 1: Remove "--- Page X ---"
    text = re.sub(r"\n*--- Page \d+ ---\n*", "\n", text)

    # Step 2: Remove standalone numbers and page numbers
    text = re.sub(r"\n\d+\n", "\n", text)

    # Step 3: Remove excessive newlines and keep paragraph separation
    text = re.sub(r"\n{3,}", "\n\n", text)  # 3+ newlines → double newlines
    text = re.sub(r"[ \t]+\n", "\n", text)  # strip spaces before line breaks

    # Step 4: Merge broken lines (lines that end without Bangla punctuation)
    lines = text.splitlines()
    merged_lines = []
    buffer = ""

    for line in lines:
        line = line.strip()
        if not line:
            if buffer:
                merged_lines.append(buffer)
                buffer = ""
            continue

        buffer += " " + line if buffer else line

        if line.endswith(("।", "!", "?", ":")):
            merged_lines.append(buffer.strip())
            buffer = ""

    if buffer:
        merged_lines.append(buffer.strip())

    cleaned_text = "\n\n".join(merged_lines)
    return cleaned_text


# Clean and save output
input_path = "bangla_ocr_output.txt"
output_path = "cleaned_bangla_ocr_output.txt"

cleaned = clean_ocr_text(input_path)

with open(output_path, "w", encoding="utf-8") as f:
    f.write(cleaned)

print(f"✅ Cleaned text saved to {output_path}")


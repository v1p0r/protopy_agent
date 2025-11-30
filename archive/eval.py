def count_tokens_for_file(encoding, file_path):
    """Count tokens for a single image (Base64 encoded)."""
    with open(file_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return len(encoding.encode(b64))

def count_total_tokens_in_folder(folder_path, encoding_name="cl100k_base"):
    """
    Loop through all image files in a folder and sum token counts.
    Supports recursive search.
    """
    encoding = tiktoken.get_encoding(encoding_name)
    total_tokens = 0
    per_file = {}

    for root, _, files in os.walk(folder_path):
        for file in sorted(files):
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                path = os.path.join(root, file)
                tokens = count_tokens_for_file(encoding, path)
                per_file[path] = tokens
                total_tokens += tokens
    
    print("\n=== Token counts per file ===")
    for f, t in per_file.items():
        print(f"{f}: {t} tokens")
    print(f"\nTotal tokens across folder: {total_tokens}")
    return total_tokens, per_file
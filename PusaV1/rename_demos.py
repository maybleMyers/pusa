import os
import re

def sanitize_filename(filename):
    """Sanitizes a filename to be URL-friendly."""
    if 't2v_output' in filename:
        return filename
        
    name, ext = os.path.splitext(filename)
    
    # Replace decimal points in noise values with 'p' to avoid URL issues
    name = re.sub(r'(\d)\.(\d)', r'\1p\2', name)
    
    # Replace brackets, commas, and spaces
    name = name.replace('[', '_').replace(']', '').replace(', ', '_').replace(',', '_')
    
    # Clean up any resulting multiple underscores
    name = re.sub(r'__+', '_', name)
    
    return name + ext

def rename_demo_files(directory):
    """Renames all demo files in a directory to be URL-friendly."""
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return

    print(f"Scanning '{directory}' for files to rename...")
    for filename in sorted(os.listdir(directory)):
        if filename.endswith((".mp4", ".gif")):
            sanitized_name = sanitize_filename(filename)
            if sanitized_name != filename:
                old_path = os.path.join(directory, filename)
                new_path = os.path.join(directory, sanitized_name)
                print(f"Renaming '{filename}' to '{sanitized_name}'")
                os.rename(old_path, new_path)

if __name__ == "__main__":
    # Assuming the script is run from the project root (Pusa-VidGen)
    outputs_directory = "./outputs"
    rename_demo_files(outputs_directory)
    print("\nRenaming finished.") 
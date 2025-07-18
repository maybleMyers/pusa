import os
from moviepy.editor import VideoFileClip

def convert_videos_to_gifs(input_dir):
    """
    Converts all .mp4 videos in the input directory to .gif files
    in the same directory.
    """
    if not os.path.exists(input_dir):
        print(f"Directory not found: {input_dir}")
        return

    for filename in os.listdir(input_dir):
        if filename.endswith(".mp4"):
            video_path = os.path.join(input_dir, filename)
            gif_filename = os.path.splitext(filename)[0] + ".gif"
            gif_path = os.path.join(input_dir, gif_filename)

            if os.path.exists(gif_path):
                print(f"GIF already exists for {filename}, skipping.")
                continue
            
            print(f"Converting {video_path} to {gif_path}...")
            
            try:
                # Load the video clip
                clip = VideoFileClip(video_path)
                
                # Write the GIF file, reducing fps to lower file size
                clip.write_gif(gif_path, fps=12)
                
                print(f"Successfully converted {filename} to GIF.")
            except Exception as e:
                print(f"Failed to convert {filename}. Error: {e}")

if __name__ == "__main__":
    # Assuming the script is run from the 'PusaV1' directory
    outputs_directory = "./outputs"
    print(f"Starting conversion in '{outputs_directory}'...")
    convert_videos_to_gifs(outputs_directory)
    print("\nAll conversions finished.")
import os
import sys
import imageio
import numpy as np

# Add VDA to path for utils
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VDA_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "Video-Depth-Anything"))
sys.path.append(VDA_PATH)

from utils.dc_utils import read_video_frames

def convert_to_gif(mp4_path, gif_path, fps=10):
    print(f"Converting {mp4_path} -> {gif_path}")
    if not os.path.exists(mp4_path):
        print(f"Error: {mp4_path} not found")
        return

    # Use read_video_frames which handles cv2/decord
    # It returns (frames, fps), where frames is [T, H, W, C] (if RGB)
    # Wait, read_video_frames returns frames as [T, H, W, C] usually?
    # Let's verify usage in visualize_quantized.py:
    # frames, target_fps = read_video_frames(...)
    # It reads original video.
    # But here we want to read the GENERATED mp4s (which are depth maps).
    # Does read_video_frames handle them? Yes, they are standard mp4s.
    
    frames, original_fps = read_video_frames(mp4_path, -1, -1, -1)
    
    # Save as GIF
    # duration = 1/fps usually for GIF
    # imageio expects 'duration' per frame in seconds
    duration = 1.0 / fps
    
    # Ensure uint8
    if frames.dtype != np.uint8:
        frames = frames.astype(np.uint8)
        
    imageio.mimsave(gif_path, frames, format='GIF', duration=duration, loop=0)
    print("Done")

def main():
    assets_dir = os.path.join(SCRIPT_DIR, "..", "assets", "demo")
    
    fp32_mp4 = os.path.join(assets_dir, "davis_rollercoaster_fp32.mp4")
    fp32_gif = os.path.join(assets_dir, "davis_rollercoaster_fp32.gif")
    
    int8_mp4 = os.path.join(assets_dir, "davis_rollercoaster_int8_sim.mp4")
    int8_gif = os.path.join(assets_dir, "davis_rollercoaster_int8_sim.gif")
    
    convert_to_gif(fp32_mp4, fp32_gif)
    convert_to_gif(int8_mp4, int8_gif)

if __name__ == "__main__":
    main()

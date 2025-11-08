import os
import shutil

# Create a temporary folder with just the good seed105 file
temp_folder = "temp_seed105_only"
os.makedirs(temp_folder, exist_ok=True)

# Copy only the working seed105 file
src_file = "samples/config.ema_0.9999_000500.pt.seed105_step0.jsonl"
dst_file = os.path.join(temp_folder, "config.ema_0.9999_000500.pt.seed105_step0.jsonl")
shutil.copy2(src_file, dst_file)

print(f"Copied {src_file} to {temp_folder}")
print("Now run: .\\venv\\Scripts\\python.exe eval_DiffuVQA.py --folder temp_seed105_only")
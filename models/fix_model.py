import zipfile
import json
import os
from pathlib import Path

# Path to your keras model
model_path = Path("models/autoencoder_final64/autoencoder_final64.keras")
temp_dir = Path("models/autoencoder_final64/temp_extracted")

print("Unpacking .keras file...")
with zipfile.ZipFile(model_path, 'r') as zip_ref:
    zip_ref.extractall(temp_dir)

# The architecture config is usually stored in config.json inside the archive
config_json_path = temp_dir / "config.json"

if config_json_path.exists():
    print("Cleaning quantization_config from config.json...")
    with open(config_json_path, 'r', encoding='utf-8') as f:
        config_data = json.load(f)

    # Recursive function to remove 'quantization_config' keys from any layer configuration dictionaries
    def remove_quantization_config(obj):
        if isinstance(obj, dict):
            obj.pop("quantization_config", None)
            for key, value in obj.items():
                remove_quantization_config(value)
        elif isinstance(obj, list):
            for item in obj:
                remove_quantization_config(item)

    remove_quantization_config(config_data)

    with open(config_json_path, 'w', encoding='utf-8') as f:
        json.dump(config_data, f)
    print("✅ Successfully stripped quantization_config!")

# Repack back into a .keras zip archive
fixed_model_path = Path("models/autoencoder_final64/autoencoder_final64_fixed.keras")
print("Repacking into a clean .keras file...")
with zipfile.ZipFile(fixed_model_path, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
    for foldername, subfolders, filenames in os.walk(temp_dir):
        for filename in filenames:
            file_path = Path(foldername) / filename
            arcname = file_path.relative_to(temp_dir)
            zip_ref.write(file_path, arcname)

print(f"✅ Done! Created clean model at: {fixed_model_path}")

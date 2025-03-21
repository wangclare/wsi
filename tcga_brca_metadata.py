import os
import openslide
import json

main_folder = '/scratch/leuven/373/vsc37341/TCGA-BRCA/data/'
output_magnification_file = os.path.join(main_folder, 'svs_magnification_info.json')
output_metadata_file = os.path.join(main_folder, 'svs_full_metadata.json')

svs_files = []
for root, dirs, files in os.walk(main_folder):
    for file in files:
        if file.lower().endswith('.svs'):
            svs_files.append(os.path.join(root, file))

magnification_info = {}
full_metadata_info = {}

for svs_path in svs_files:
    try:
        slide = openslide.OpenSlide(svs_path)
        properties = slide.properties
        filename = os.path.relpath(svs_path, main_folder)

        magnification = properties.get('aperio.AppMag') or properties.get('openslide.objective-power')
        magnification_info[filename] = magnification if magnification else "None"

        full_metadata_info[filename] = dict(properties)
        slide.close()
    except Exception as e:
        magnification_info[filename] = f"Error: {e}"
        full_metadata_info[filename] = f"Error: {e}"

with open(output_magnification_file, 'w') as f:
    json.dump(magnification_info, f, indent=2)

with open(output_metadata_file, 'w') as f:
    json.dump(full_metadata_info, f, indent=2)

print(f"完成：共处理 {len(svs_files)} 个 SVS 文件。输出保存至 {main_folder}")

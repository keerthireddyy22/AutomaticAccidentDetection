def create_non_accident_annotation_file(output_annotation_file, num_videos=500):
    with open(output_annotation_file, 'w') as output_file:
        for i in range(1, num_videos + 1):
            vidname = f"{i:06d}"
            output_file.write(f"{vidname},[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]\n")

# Specify custom path for the non-accident annotation file
non_accident_annotation_file = r"C:\Users\pavan\OneDrive\Desktop\miniproject\noncrash.txt"
create_non_accident_annotation_file(non_accident_annotation_file)

print(f"Non-accident annotation file '{non_accident_annotation_file}' created and saved.")

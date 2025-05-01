import json

#Function to test that the number of images match in the array and in the file and making sure all the file paths have a correct file path
def test_num_of_images(filepath, parsed_data_path):
  with open(filepath, 'r') as f:
    data = f.readlines()
  with open(parsed_data_path, 'r') as f:
    parsed_data = json.load(f)
    num_images_file = 0
    num_images_arr = len(parsed_data)

    if data is None:
      return f"{filepath} is empty"

    for line in data:
      if 'jpg' in line:
        num_images_file += 1
  print(f"Total images in file: {num_images_file}")
  print(f"Total images in arr: {num_images_arr}")

  return num_images_file == num_images_arr


#function loops through annotated data and checks for:
  # if image is in file
  # if bbox is in file

def test_data_in_file(file_data_path, parsed_data_path, split):
    with open(file_data_path, 'r') as f:
        file_data = [line.strip() for line in f.readlines()]
        print(f"filedata[0]: {file_data[0]}")

    with open(parsed_data_path, 'r') as f:
        parsed_data = json.load(f)

    for entry in parsed_data:
        filename = entry['filename'].replace(f"data/WIDER/WIDER_{split}/images", "").strip()
        if filename not in file_data:
            print(f"Filename not found: {entry['filename']}")
            continue

        for bbox in entry['bboxes']:
            bbox_str = ' '.join(map(str, bbox))
            if bbox_str not in file_data:
                print(f"Bbox not found: {bbox_str}")
                print(f"Matching filename: {entry['filename']}")
                return False, bbox

    return True

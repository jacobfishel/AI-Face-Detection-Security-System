# File to parse the wider dataset into the proper format to train data
import json
import tests.parse_wider_test as test
from config import TRAIN_ANNOTATIONS_PATH, PARSED_TRAIN_ANNOTATIONS_PATH, TRAIN_IMAGES_FOLDER_PATH


# ---------------- CODE -------------- #
# open the annotations file and modify data 

final_data_arr = []


with open(TRAIN_ANNOTATIONS_PATH, 'r') as f:
  data = f.readlines()


  img_path = None #jpg
  bbox_coords = [] #1, 2, 3, 4, 5, 6, 7, 78, 
  for i, line in enumerate(data):
    
    # If the line has 'jpg' its img path and parse each line until another jpg line
    if 'jpg' not in line:
      bbox_coords.append([line.strip()])
      continue
    else:
      #if img_path is not none, we are seeing jpg again:
        #push all info and reset values
      if img_path is not None:
        #Delete the num of bboxes from the coords (useless)
        del(bbox_coords[0])
        final_data_arr.append({"filename": img_path, "bboxes": bbox_coords})  #updated dict storage
        # final_data_arr.append((img_path, bbox_coords))  #old dat storage
        bbox_coords = []

      img_path = TRAIN_IMAGES_FOLDER_PATH + line.strip()

  #Once we reach the end, push the final pieces of data
  del(bbox_coords[0])
  final_data_arr.append({"filename": img_path, "bboxes": bbox_coords})

  # TEST FUNCTION LINE TO TEST PARSE
  # print(f"Testing each line in data = {test.test_data_in_file(data, final_data_arr)}")      

with open(PARSED_TRAIN_ANNOTATIONS_PATH, 'w') as f:
  json.dump(final_data_arr, f, indent=2)

# TEST FUNCTION LINE TO TEST PARSE
# print(test.test_num_of_images(train_annotations_path, final_data_arr))



    



  


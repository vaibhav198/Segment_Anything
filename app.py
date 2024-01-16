import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os
import shutil
import matplotlib
# matplotlib.use('TkAgg')
from utils import DetectionModel
from utils import SegmentationModel
from utils.helpers import *
from dotenv import load_dotenv
# Load environment variables from the .env file
load_dotenv()

model = DetectionModel(os.getenv('DETECTION_MODEL'))
sam = SegmentationModel(os.getenv("SEGMENTATION_MODEL"))
dest_folder = os.getenv("DESTINATION_FOLDER")
seg_folder = os.getenv("PROCESSED_IMG_FOLDER")

if os.getenv('MODE') == 'img':
    try:
            
        #get img folder name
        img_folder = os.getenv('IMAGES_FOLDER')
        img_files = os.listdir(img_folder)

        for img_file in img_files:
            print(f"Running for file: {img_file}")
            image = cv2.imread('./images/'+img_file)
            orig_image = image.copy()
            results, class_names = model.pred(image)
            print(f"class_names: {class_names}")
            #Plot rect
            for result in results:
                # print(">>>>>Helo>>>>")
                boxes = result.boxes
                boxs = boxes.xyxy
                print(f"boxs: {boxs}")
                cls = boxes.cls
                cls = list(map(int, cls.tolist()))
                print(f"cls###: {cls}")
            #     class_names = ['person', 'bicycle', 'car', 'bus', 'cat', 'dog']
            #     output_index = cls
            #     class_name = class_names[0]
                
            #     print(class_name)
                if len(cls)>0:
                    for c in range(len(cls)):
                        class_name = class_names[cls[c]]
                        #get the coord of bounding box
                        l = boxs[c].tolist()
                        x1, y1, x2, y2 = l[0],l[1],l[2],l[3]        
                        cv2.rectangle(image, (int(x1), int(y1), int(x2), int(y2)), (0,255,0), 2)

                        #add text
                        text = class_name
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 1.5
                        thickness = 4
                        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
                        text_x = int(x1+5)
                        text_y = int(y1+text_size[1]+5)
                        cv2.putText(image, text, (text_x, text_y), font, font_scale, (0,255,0), thickness)

                        #Segmentation Task
                        input_box = np.array(boxs[0].tolist())
                        mask = sam.pred(image, input_box)

                        plt.figure(figsize=(10,10))
                        plt.imshow(orig_image)
                        show_mask(mask[0], plt.gca())
                        show_box(input_box, plt.gca())
                    create_dir(dest_folder+'/'+class_name)
                    create_dir(seg_folder+'/'+class_name)
                    shutil.copy(img_folder+'/'+img_file, dest_folder+'/'+class_name)
                    plt.axis('off')
                    plt.savefig(seg_folder+'/'+class_name+'/'+img_file)

            print(f"Done for file: {img_file}")
    except Exception as e:
        print(f"Exception: {e}")
                

        
        
        


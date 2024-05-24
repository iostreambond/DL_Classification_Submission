import os
import cv2
import natsort

"""
train
'Audi',             1012 - no augmentation (962 + 50) ==> (962 + 150) (train + test)
'Hyundai_Creta',    336  - 2 augmentation  (286 + 50) ==> (858 + 150) (train + test)  
'Mahindra_Scorpio', 374  - 2 augmentation  (324 + 50) ==> (972 + 150) (train + test)  
'Rolls_Royce',      372  - 2 augmentation  (322 + 50) ==> (966 + 150) (train + test)  
'Swift',            518  - 1 augmentation  (468 + 50) ==> (936 + 150) (train + test)  
'Tata_Safari',      485  - 1 augmentation  (435 + 50) ==> (870 + 150) (train + test)  
'Toyota_Innova'     924  - no augmentation (874 + 50) ==> (874 + 150) (train + test) 
"""

class_names = ['Audi', 'Hyundai_Creta', 'Mahindra_Scorpio', 'Rolls_Royce', 'Swift', 'Tata_Safari', 'Toyota_Innova']

for class_name in class_names:
    root_folder = "kaggle/Cars_Dataset/test/" + class_name + "/"

    all_images = natsort.natsorted(os.listdir(root_folder))
    total_frames = len(all_images)

    for in_image in all_images:
        image_path = root_folder + in_image
        image_file_name = in_image[0:-4]

        cv_image = cv2.imread(image_path)

        flipped_cv_image = cv2.flip(cv_image, 1)
        cv2.imwrite(root_folder + image_file_name + "_hflipped.jpg", flipped_cv_image)

        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        h_image, s_image, v_image = cv2.split(hsv_image)

        clahe = cv2.createCLAHE(clipLimit=0.8, tileGridSize=(8, 8))
        v_image_clahe = clahe.apply(v_image)
        clahe_image_hsv = cv2.merge((h_image, s_image, v_image_clahe))
        clahe_image_rgb = cv2.cvtColor(clahe_image_hsv, cv2.COLOR_HSV2BGR)

        cv2.imwrite(root_folder + image_file_name + "_clahe_hsv.jpg", clahe_image_rgb)

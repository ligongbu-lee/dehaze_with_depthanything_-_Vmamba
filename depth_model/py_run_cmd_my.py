import os


def folder_trans(folder_dir):
    for folder, _, file_list in os.walk(folder_dir):
        for file in file_list:
            file_Path = os.path.join(folder, file)

            # print("orign path:   ", file_Path)
            aa = f'python test_simple.py --image_path {file_Path} --model_name RA-Depth'
            print(aa)
            a = os.system(aa)

# G:/0000_data/railway_dataset/dataset/GT
# python test_simple.py --image_path /test.png --model_name RA-Depth

folder_trans("G:/0000_data/roadway_dataset/dataset_rain_snow_fog/fog/")
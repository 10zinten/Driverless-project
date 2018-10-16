'''
Creating dataset for our cone object detection model for donkey car.
'''

import os
from PIL import Image
from tqdm import tqdm

data_dir = 'data/'
tub_src = os.path.join(data_dir, 'tub_17_18-08-03')
tub_des_1 = os.path.join(data_dir, 'dataset-01')
tub_des_2 = os.path.join(data_dir, 'dataset-02')

not_images = []
def save(filaname, output_dir):
    try:
        image = Image.open(filename)
        image.save(os.path.join(output_dir, filename.split('/')[-1]))
        return True
    except:
        not_images.append(filename)

def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    else:
        print('Warning: output dir {} already exists'.format(dir_name))

if __name__ == '__main__':
    filenames = os.listdir(tub_src)
    filenames = [os.path.join(tub_src, f) for f in filenames if f.endswith('.jpg')]

    mid = len(filenames) // 2

    # create the destination dir
    create_dir(tub_des_1)
    create_dir(tub_des_2)

    # create first dataset
    count = 0
    print("Creating {}".format(tub_des_1))
    for filename in tqdm(filenames[:mid]):
        if save(filename, tub_des_1):
            count += 1
    print("[INFO] Created {} with {} images".format(tub_des_1, count))

    count = 0
    print("Creating {}".format(tun_des_2))
    for filename in tqdm(filenames[mid:]):
        if save(filename, tub_des_2):
            count += 1
    print('[INFO] Created {} with {} images'.format(tub_des_2, count))

    # save all the not image file into file
    with open('not_images', 'w') as f:
        f.writelines(not_image)

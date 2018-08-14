import os

data_dir = 'data/'

tubs = os.listdir(data_dir)
tub_ids = [int(tub.split('_')[1]) for tub in tubs]

# find the last data id in tub_17
dest_tub = os.path.join(data_dir, tubs[tub_ids.index(17)])
last_data_id = 0
for data in os.listdir(dest_tub):
    try:
        data_id = int(data.split('_')[0])
    except:
        continue
    if data_id > last_data_id:
        last_data_id = data_id


def move(src, dest):
    '''Move file from src to dest.'''
    os.system("mv " + src + " " + dest)

# Merge data from all the tubs into tub_17
for tub_id in tub_ids:
    if tub_id == 17:
        continue

    print("[INFO] Moving tub_{}".format(tub_id))

    for data in os.listdir(data_dir + '/' + tubs[tub_ids.index(tub_id)]):
        try:
            data_id = int(data.split("_")[0])
        except:
            continue

        img_src = os.path.join('data', tubs[tub_ids.index(tub_id)],
                               "{}_cam-image_array_.jpg".format(data_id))
        label_src = os.path.join('data', tubs[tub_ids.index(tub_id)],
                                 "record_{}.json".format(data_id))

        img_dest = os.path.join(dest_tub,
                                "{}_cam_image_array_.jpg".format(last_data_id))
        label_dest = os.path.join(dest_tub,
                                  "record_{}.json".format(last_data_id))

        if os.path.exists(img_src) and os.path.exists(label_src):
            move(img_src, img_dest)
            move(label_src, label_dest)

        last_data_id += 1

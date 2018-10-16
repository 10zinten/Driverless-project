import os

tub_dir = 'data/tub_17_18-08-03'

for data in os.listdir(tub_dir):
    datas = data.split('_')
    if datas[0] != "record":
        continue

    if not open(os.path.join(tub_dir, data)).read():
        data_id = datas[1].split('.')[0]

        img_src = os.path.join(tub_dir,
                                "{}_cam-image_array_.jpg".format(data_id))
        label_src = os.path.join(tub_dir,
                                 "record_{}.json".format(data_id))

        os.system('rm ' + img_src)
        os.system('rm ' + label_src)




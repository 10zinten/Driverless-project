# Cone Detection with SSD-Mobilenet

### Why SSD-Mobilenet
- **Choice of SDD over YOLO**: For us detecting far away cone is more important then one which is near by and far away cone tends to very small and YOLO is less accurate for detecting small image than ssd.
- **Choice of Mobilenet over other base network like VGG**: Mobilenet is desinged for small compute device and it has the best trade of between accuracy and frame rate on small device like Rasberry Pi and our prototype using Rasberry Pi. Therefore it is the most suitable  base network arichitecture for us.

### Sample Dataset
![cones annotation](https://i.postimg.cc/sfNqhp95/cones.png)

### Annotation format
Dataset is annotated by using [sloth](https://sloth.readthedocs.io/en/latest/), wich is an open source project. It will generate a json label file with every image as a json object. Below is annotations for a sample image, which is a json object.
```
{'annotations': [{'class': 'orange',
                  'height': 35.79507484025583,
                  'type': 'rect',
                  'width': 21.945789931823516,
                  'x': 0.6391977650045684,
                  'y': 8.30957094505939},
                          ...
                          ...
                 {'class': 'orange',
                  'height': 21.093526245150763,
                  'type': 'rect',
                  'width': 15.979944125114201,
                  'x': 57.74086477207935,
                  'y': 11.292493848414042},],
 'class': 'image',
 'filename': 'dataset-02/1_cam-image_array_.jpg'}
```

### Set up the environment
```
$ cd ssd-mobilenet
$ virtualenv env -p python3
$ source env/bin/activate
$ pip install -r requriments.txt
```

### Build dataset
With generated json label file using sloth, we now need to build a dataset by splitting the dataset into *train*, *dev* and *test set*. Below is the step to build the dataset.
```
$ python build_dataset.py --data_dir <path_to_labels_file> --output <path_to_save_dataset>
```

### Training
```
$ python train.py --model_dir <path_to_experiment> --data_dir <path_to_dataset>
```

### Evaluate
```
$ python evaluate.py --model_dir <path_to_experiment> --data_dir <path_to_dataset>
```

### Detect bounding box on a single image
```
$ python detect.py --model_dir <path_to_experiment> --image <path_to_image>
```

### Testing all the components
```
$ python utest.py --model_dir <path_to_experiment>
```

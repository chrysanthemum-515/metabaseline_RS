import os
import pickle
import numpy as np


dataName = 'UCM'
path = './MyProject/UCMerced_LandUse/Images'


def makecatname2label(path):
    img2label = {}
    for (root, dirs, files) in os.walk(path):
        count = 0
        for dir in dirs:
            img2label[dir.lower()] = count
            count = count+1
    catname2label = {}
    for idx, cls in enumerate(sorted(img2label)):
        catname2label[cls] = idx
    return catname2label


def make_data(name2label, path):
    data = [[]for i in range(len(name2label))]
    for (root, dirs, files) in os.walk(path):
        for file in files:
            if(file.endswith('.jpg') or file.endswith('.tif')):
                r = root.split('/')
                if r[-1] in name2label:
                    idx = name2label[r[-1]]
                    data[idx].append(file)
                else:
                    continue
    return data


def dict_slice(dictt, end_idx, val_num):
    keys = dictt.keys()
    dict_slice_train = {}
    dict_slice_val = {}
    dict_slice_test = {}
    for key in list(keys)[:end_idx]:
        dict_slice_train[key] = dictt[key]

    for key in list(keys)[end_idx:end_idx+val_num]:
        dict_slice_val[key] = dictt[key]-end_idx

    for key in list(keys)[end_idx+val_num:]:
        dict_slice_test[key] = dictt[key]-end_idx-val_num
    return dict_slice_train, dict_slice_val, dict_slice_test


def get_label_from_img(datasetName, imgname, img2label):
    """
    给一个图片的文件名,解析出他的label对应的数字
    """
    if datasetName == "AID":
        s = imgname.split('_')
        return img2label[s[0]]
    if datasetName == "NWPU":
        return img2label[imgname[:-8]]
    if datasetName == "UCM":
        return img2label[imgname[:-6]]


def make_data_with_filepath(data, root_path, dataName):
    if dataName == "AID":
        x = [os.path.join(root_path, item.split('_')[0], item)
             for sublist in data for item in sublist]
    if dataName == 'NWPU':
        x = [os.path.join(root_path, item[:-8], item)
             for sublist in data for item in sublist]
    if dataName == 'UCM':
        x = [os.path.join(root_path, item[:-6], item)
             for sublist in data for item in sublist]
    return x


def get_labels(data, n2l, datasetName):
    labels = np.array([get_label_from_img(datasetName, imgname, n2l)
                       for sublist in data for imgname in sublist]).astype(np.int32)
    return labels


def split_train(data):
    train = [[]for _ in range(len(data))]
    val = [[]for _ in range(len(data))]
    test = [[]for _ in range(len(data))]
    for idx in range(len(data)):
        train[idx] = data[idx][:70]
        val[idx] = data[idx][70:85]
        test[idx] = data[idx][-85:]
    return train, val, test


catname2label = makecatname2label(path)
print(catname2label)
train_catname2label = {'agricultural': 0, 'baseballdiamond': 1, 'buildings': 2, 'chaparral': 3, 
                       'denseresidential': 4,'freeway': 5, 'harbor': 6,
                        'mediumresidential': 7, 'overpass': 8, 'parkinglot': 9}                      
val_catname2label = {'airplane': 0, 'forest': 1, 'intersection': 2,
                     'runway': 3, 'storagetanks': 4}
test_catname2label = {'sparseresidential': 0, 'golfcourse': 1, 'mobilehomepark': 2,
                      'tenniscourt': 3,'beach': 4,'river':5}


dtrain = make_data(train_catname2label, path)
dtrain_train, dtrain_val, dtrain_test = split_train(dtrain)
dval = make_data(val_catname2label, path)
dtest = make_data(test_catname2label, path)

data_train_train = make_data_with_filepath(dtrain_train, path, dataName)
labels_train_train = get_labels(dtrain_train, train_catname2label, dataName)
d = {'catname2label': train_catname2label,
     'labels': labels_train_train, 'data': data_train_train}
with open('./meta-baseline/materials/UCM-imagenet/UCMImageNet_category_split_train_phase_train.pickle', 'wb')as f:
    pickle.dump(d, f)

data_train_val = make_data_with_filepath(dtrain_val, path, dataName)
labels_train_val = get_labels(dtrain_val, train_catname2label, dataName)
d = {'catname2label': train_catname2label,
     'labels': labels_train_val, 'data': data_train_val}
with open('./meta-baseline/materials/UCM-imagenet/UCMImageNet_category_split_train_phase_val.pickle', 'wb')as f:
    pickle.dump(d, f)

data_train_test = make_data_with_filepath(dtrain_test, path, dataName)
labels_train_test = get_labels(dtrain_test, train_catname2label, dataName)
d = {'catname2label': train_catname2label,
     'labels': labels_train_test, 'data': data_train_test}
with open('./meta-baseline/materials/UCM-imagenet/UCMImageNet_category_split_train_phase_test.pickle', 'wb')as f:
    pickle.dump(d, f)

data_test = make_data_with_filepath(dtest, path, dataName)
labels_test = get_labels(dtest, test_catname2label, dataName)
d = {'catname2label': test_catname2label,
     'labels': labels_test, 'data': data_test}
with open('./meta-baseline/materials/UCM-imagenet/UCMImageNet_category_split_test.pickle', 'wb')as f:
    pickle.dump(d, f)

data_val = make_data_with_filepath(dval, path, dataName)
labels_val = get_labels(dval, val_catname2label, dataName)
d = {'catname2label': val_catname2label,
     'labels': labels_val, 'data': data_val}
with open('./meta-baseline/materials/UCM-imagenet/UCMImageNet_category_split_val.pickle', 'wb')as f:
    pickle.dump(d, f)

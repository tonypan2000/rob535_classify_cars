from glob import glob
import os


class_to_idx = {'Unknown': 0, 'Compacts': 1, 'Sedans': 2, 'SUVs': 3, 'Coupes': 4,
                'Muscle': 5, 'SportsClassics': 6, 'Sports': 7, 'Super': 8, 'Motorcycles': 9, 'OffRoad': 10,
                'Industrial': 11, 'Utility': 12, 'Vans': 13, 'Cycles': 14, 'Boats': 15,
                'Helicopters': 16, 'Planes': 17, 'Service': 18, 'Emergency': 19, 'Military': 20,
                'Commercial': 21, 'Trains': 22
                }
idx_to_class = {i: c for c, i in class_to_idx.items()}

prediction_files = glob(os.path.join('./', 'exp3', 'labels', '*.txt'))
original_files = glob(os.path.join('./', 'test', '*/*_image.jpg'))
with open(os.path.join('./', 'team20.csv'), 'w') as out:
    out.write('guid/image,label\n')
    index = 0
    prev_label = str(0)
    for prediction_filename in prediction_files:
        img_file = original_files[index]
        index += 1
        img_name = os.path.split(os.path.split(img_file)[0])[1] + '_' + os.path.basename(img_file)
        while os.path.basename(prediction_filename).split('.')[0] != img_name.split('.')[0]:
            guid, image, _ = os.path.basename(img_name).split('_')
            out.write(guid + '/' + image + ',' + prev_label + '\n')
            img_file = original_files[index]
            index += 1
            img_name = os.path.split(os.path.split(img_file)[0])[1] + '_' + os.path.basename(img_file)
        if os.path.basename(prediction_filename).split('.')[0] == img_name.split('.')[0]:
            guid, image, _ = os.path.basename(prediction_filename).split('_')
            with open(prediction_filename, 'r') as input:
                class_id = class_to_idx[input.readline().split()[0]]
                label = str(0)
                if class_id > 0 and class_id < 9:
                    label = str(1)
                elif class_id > 8 and class_id < 15:
                    label = str(2)
                prev_label = label
                out.write(guid + '/' + image + ',' + label + '\n')
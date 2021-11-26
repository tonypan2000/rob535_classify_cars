from glob import glob
import os


prediction_files = glob(os.path.join('./', 'exp4', 'labels', '*.txt'))
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
                class_id = int(input.readline().split()[0])
                label = str(0)
                if class_id > 0 and class_id < 9:
                    label = str(1)
                elif class_id > 8 and class_id < 15:
                    label = str(2)
                prev_label = label
                out.write(guid + '/' + image + ',' + label + '\n')
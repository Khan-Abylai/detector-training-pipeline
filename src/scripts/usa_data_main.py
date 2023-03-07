from pathlib import Path
from glob import glob
import os.path as osp
import numpy as np
import shutil
import pickle
import json
import cv2
import os
import sys

SYNTHETIC_IMAGE_COUNT = 0
SINGLE_BOX_COUNT = 0
SINGLE_PLATE_COUNT = 0
MULTIPLE_PLATE_COUNT = 0

NO_PLATE_COUNT = 0
NO_PROCEEDED_PLATES_COUNT = 0
NO_ANNOTATION_COUNT = 0
ERROR_PLATES = 0


def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation) for
                      im in im_list]
    return cv2.hconcat(im_list_resize)


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def execute_on_file(full_image_path, basename, item, output_image_folder, output_plate_folder, region, debug=False,
                    debug_folder='../debug', prefix=''):
    global SINGLE_PLATE_COUNT, MULTIPLE_PLATE_COUNT, NO_PLATE_COUNT, NO_PROCEEDED_PLATES_COUNT, ERROR_PLATES, SINGLE_BOX_COUNT, SYNTHETIC_IMAGE_COUNT
    image_width = int(item['width'])
    image_height = int(item['height'])
    plate = item['objects']
    if 'issquare' in item:
        is_square = item['issquare']
    else:
        is_square = False
    if len(plate) == 1:
        plate = plate[0]
        if bool(plate):

            plate_label = plate['plate_number']

            if region is None or isinstance(region, type(None)):
                region = 'error'
            else:
                region = region.lower()

            if plate_label is None or isinstance(plate_label, type(None)):
                plate_label = 'error'
            else:
                plate_label = plate_label.lower()
            center_point = np.array(plate['center_point']).astype(np.float32)
            points = order_points(np.array(plate['points']).astype(np.float32))
            lt, rt, rb, lb = points
            plate_width = plate['plate_width']
            plate_height = plate['plate_height']
            lp_coords = np.array([[0, 0], [0, plate_height], [plate_width, 0], [plate_width, plate_height]],
                                 dtype=np.float32)

            image = cv2.imread(full_image_path)
            if image is not None or not isinstance(image, type(None)):

                interim_image_folder = Path(output_image_folder)
                Path(os.path.join(output_image_folder, os.path.basename(os.path.dirname(full_image_path))))
                interim_plate_folder = Path(
                    os.path.join(output_plate_folder, os.path.basename(os.path.dirname(full_image_path))))
                interim_plate_folder = '/home/yeleussinova/data_SSD/usa_images/plates/iteration_11'
                if not interim_image_folder.exists():
                    interim_image_folder.mkdir(parents=True, exist_ok=True)
                # if not interim_plate_folder.exists():
                #     interim_plate_folder.mkdir(parents=True, exist_ok=True)
                #
                file_base_name = basename.split('.')[0]

                if 'error' in region or 'error' in plate_label:
                    print(f'{basename} plate can not be proceed for recognition')
                    ERROR_PLATES += 1
                else:
                    SINGLE_PLATE_COUNT += 1
                    cropped_plate_image = cv2.warpPerspective(image, cv2.getPerspectiveTransform(
                        np.array([lt, lb, rt, rb]).astype(np.float32), lp_coords),
                                                              (int(plate_width), int(plate_height)))

                    if not is_square:
                        cropped_plate_image = cropped_plate_image
                    else:
                        half = int(plate_height // 2)
                        top = cropped_plate_image[:half, :]
                        bottom = cropped_plate_image[half:, :]
                        cropped_plate_image = hconcat_resize_min([top, bottom])

                    plate_destination_path = os.path.join(interim_plate_folder, file_base_name + '.png')

                    cv2.imwrite(plate_destination_path, cropped_plate_image)
                    with open(plate_destination_path.replace('.png', '.txt'), 'w') as f:
                        f.write(plate_label + ',' + region)

                    base_folder = os.path.dirname(full_image_path)

                    synthetic_image_path = glob(os.path.join(base_folder, '*.png'))

                    if len(synthetic_image_path) == 1:
                        synthetic_image_path = synthetic_image_path[0]
                        synthetic_image = cv2.imread(synthetic_image_path)

                        if not is_square:
                            synthetic_image = synthetic_image
                        else:
                            half = synthetic_image.shape[0] // 2
                            top = synthetic_image[:half, :]
                            bottom = synthetic_image[half:, :]
                            synthetic_image = hconcat_resize_min([top, bottom])
                        synthetic_plate_destination_path = plate_destination_path.replace('.jpg', '_synthetic.jpg')
                        cv2.imwrite(synthetic_plate_destination_path, synthetic_image)
                        with open(synthetic_plate_destination_path.replace('.jpg', '.txt'), 'w') as f:
                            f.write(plate_label + ',' + region)
                        SYNTHETIC_IMAGE_COUNT += 1
                    else:
                        print('Synthetic image not exists')

                SINGLE_BOX_COUNT += 1
                box = np.array([center_point, np.array([plate_width, plate_height]), lt, lb, rt, rb]).reshape(-1, 12)
                box[:, ::2] /= image_width
                box[:, 1::2] /= image_height
                image_destination_path = os.path.join(interim_image_folder, file_base_name + '.jpg')
                # shutil.copy(full_image_path, image_destination_path)

                np.savetxt(image_destination_path.replace('.jpg', '.pb'), box)
            else:
                print(f'{basename} image can not be read')
                NO_PROCEEDED_PLATES_COUNT += 1
        else:
            NO_PROCEEDED_PLATES_COUNT += 1
            print(f'{basename} has a problem with points on single plate')
    else:
        print(f'{basename} has multiple plate points')
        MULTIPLE_PLATE_COUNT += 1

if __name__ == '__main__':
    debug = True
    debug_folder = '../debug'
    folder = 'iteration_11'
    file = f'/home/yeleussinova/data_SSD/usa_images/jsons/{folder}.json'
    glob_pattern = f'/home/yeleussinova/data_SSD/usa_images/images/{folder}'
    output_folder = f'/home/yeleussinova/data_SSD/usa_images/plates/{folder}'

    region = os.path.basename(file).replace('.json', '')

    pkl_1 = file.replace('.json', '.pkl')
    pkl_2 = file.replace('.json', '_2.pkl')

    image_output_folder = os.path.join(output_folder)
    plate_output_folder = os.path.join(output_folder, 'plates')

    images = glob(os.path.join(glob_pattern, "*"))
    with open(pkl_1, 'wb') as f:
        pickle.dump(images, f, protocol=pickle.HIGHEST_PROTOCOL)


    file_base_names = np.array([os.path.basename(x) for x in images])
    with open(pkl_2, 'wb') as f:
        pickle.dump(file_base_names, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(file, 'rb') as f:
        annotation_content = json.loads(f.read())

    print(f'Working with:{file}')
    print(f'Full image length is:{len(images)}')
    print(f'Annotation data length:{len(annotation_content)}')
    for idx, (image_basename, image_full_path) in enumerate(zip(file_base_names, images)):
        print(f'Current idx:{idx}')

        alternative_basename = os.path.basename(os.path.dirname(image_full_path))

        if image_basename in annotation_content:
            execute_on_file(image_full_path, image_basename, annotation_content[image_basename],
                            image_output_folder, plate_output_folder, region, debug, debug_folder,
                            prefix='')
        elif alternative_basename in annotation_content:
            execute_on_file(image_full_path, image_basename, annotation_content[alternative_basename],
                            image_output_folder, plate_output_folder, region, debug, debug_folder,
                            prefix='')
        else:
            NO_ANNOTATION_COUNT += 1
            print(f'{image_full_path} not in annotation')

    print(f'Working with:{file}')
    print(f'Full image length is:{len(images)}')
    print(f'Annotation data length:{len(annotation_content)}')
    print(f'No annotation files count:{NO_ANNOTATION_COUNT}')
    print(f'Multiple plate counts:{MULTIPLE_PLATE_COUNT}')
    print(f'No proceed plate counts:{NO_PROCEEDED_PLATES_COUNT}')
    print(f'Error plate counts:{ERROR_PLATES}')
    print(f'Single plate counts:{SINGLE_PLATE_COUNT}')
    print(f'Single box counts:{SINGLE_BOX_COUNT}')
    print(f'Synthetic image count:{SYNTHETIC_IMAGE_COUNT}')

# from scipy.spatial import distance as dist
# def order_points(pts):
#   # sort the points based on their x-coordinates
#   xSorted = pts[np.argsort(pts[:, 0]), :]
#   # grab the left-most and right-most points from the sorted
#   # x-roodinate points
#   leftMost = xSorted[:2, :]
#   rightMost = xSorted[2:, :]
#   # now, sort the left-most coordinates according to their
#   # y-coordinates so we can grab the top-left and bottom-left
#   # points, respectively
#   leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
#   (tl, bl) = leftMost
#   # now that we have the top-left coordinate, use it as an
#   # anchor to calculate the Euclidean distance between the
#   # top-left and right-most points; by the Pythagorean
#   # theorem, the point with the largest distance will be
#   # our bottom-right point
#   D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
#   (br, tr) = rightMost[np.argsort(D)[::-1], :]
#   # return the coordinates in top-left, top-right,
#   # bottom-right, and bottom-left order
#   return np.array([tl, tr, br, bl], dtype="float32")

# platesmaina format
# import json
# import os
# import time
# from glob import glob
#
# import numpy as np
#
# BASE_FOLDER = '/home/yeleussinova/data_SSD/usa_images/'
#
# annotation_path = os.path.join('/home/yeleussinova/data_SSD/usa_images/jsons', 'data_platesmania.json')
#
# with open(annotation_path, 'r') as f:
#     content = json.loads(f.read(), strict=False)
#
# folder = os.path.join(BASE_FOLDER, 'platesmania')
# out_plate_folder = os.path.join(BASE_FOLDER, 'platesmania')
# all_image_folders = glob(os.path.join(folder, "*"))
#
# single_plates = 0
# multi_plates = 0
# no_plates = 0
# classes = {}
# plate_numbers = {}
# errors = 0
# start_time = time.time()
# print("folders", len(all_image_folders))
# minus = 0
# plus = 0
# for idx, folder in enumerate(all_image_folders):
#
#     folder_content = np.array(glob(os.path.join(folder, "*")))
#     keep_mask = [True if os.path.basename(x) in ['plate.txt', 'info.txt'] or ".png" in os.path.basename(x) or (
#             '.jpg' in os.path.basename(x) and '-' not in os.path.basename(x) and '_' not in os.path.basename(
#         x) and '.pb' not in os.path.basename(x)) else False for x in folder_content]
#     remove_mask = [True if not x else False for x in keep_mask]
#
#     keep = folder_content[keep_mask]
#     remove = folder_content[remove_mask]
#
#     print(f"Keep data:{','.join([os.path.basename(x) for x in keep])}")
#     # print(f"Remove data:{','.join([os.path.basename(x) for x in remove])}")
#     # print("_____REMOVING_____")
#     # for item in remove:
#     #     os.remove(item)
#
#     basename = [x for x in keep if ".jpg" in x]
#
#     if len(basename) == 1:
#         image = basename[0]
#         basename = os.path.basename(basename[0])
#     else:
#         basename = None
#         image = None
#
#     if basename in content:
#         obj_ = content[basename]
#         if len(obj_['objects']) != 0:
#             image_width = int(obj_['width'])
#             image_height = int(obj_['height'])
#             print(image_width, image_height)
#             item = obj_['objects']
#             error_plates = []
#             plates = []
#             for sample in item:
#                 stop = 1
#                 if 'class' not in sample or 'plate_number' not in sample or 'points' not in sample:
#                     print("________________ NO INFORMATION _____________")
#                     no_plates += 1
#                     continue
#
#                 class_ = sample['class']
#                 plate_number = sample['plate_number']
#                 if class_ in classes:
#                     classes[class_] += 1
#                 else:
#                     classes[class_] = 1
#
#                 if plate_number in plate_numbers:
#                     plate_numbers[plate_number] += 1
#                 else:
#                     plate_numbers[plate_number] = 1
#
#                 if plate_number is None or isinstance(plate_number, type(None)) or plate_number == 'null':
#                     error_plates.append(None)
#                 else:
#                     points = np.array(sample['points']).astype(np.int32)
#                     points = order_points(points)
#                     lt, rt, rb, lb = points
#                     # lt = points[0] # green
#                     # rt = points[3] # pink
#                     # rb = points[1] # light-blue
#                     # lb = points[2]  # blue
#
#                     # center_point = np.array(sample['center_point']).astype(np.int32)
#                     # center_point = np.array([((rb[0] - lt[0]) / 2) + ((rt[0] - lb[0]) / 2),
#                     #                          ((rb[1] - lt[1]) / 2) + ((lb[1] - rt[1]) / 2)]).astype(np.int32)
#                     center_point = np.array([(lt[0] + ((rt[0] - lt[0]) / 2)),
#                                              (lt[1] + ((lb[1] - lt[1]) / 2))]).astype(np.int32)
#                     stop = 1
#
#                     w = int(((rt[0] - lt[0]) + (rb[0] - lb[0])) / 2)
#
#                     h = int(((lb[1] - lt[1]) + (rb[1] - lt[1])) / 2)
#
#
#                     if(w<0 or h<0 or center_point[0] < 0 or center_point [1]<0):
#                         minus +=1
#                         continue
#                     else:
#                         plates.append([plate_number, center_point, w, h, lt, lb, rt, rb])
#
#             if len(plates) == 1:
#                 plates = plates[0]
#                 box = np.array([plates[1], np.array([plates[2], plates[3]]), plates[4], plates[5], plates[6], plates[7]]).reshape(-1, 12)
#                 box[:, ::2] /= image_width
#                 box[:, 1::2] /= image_height
#                 single_plates += 1
#                 extension = os.path.basename(image).split('.')[-1]
#                 np.savetxt(image.replace(extension, 'pb'), box)
#             elif len(plates) > 1:
#                 boxes = np.array(
#                     [np.array([x[1], np.array([x[2], x[3]]), x[4], x[5], x[6], x[7]]) for x in plates]).reshape(-1, 12)
#                 multi_plates += 1
#                 extension = os.path.basename(image).split('.')[-1]
#                 np.savetxt(image.replace(extension, 'mpb'), boxes)
#             else:
#                 no_plates += 1
#             print(f"Proceeding id: {idx}, f_name:{image}")
#     else:
#         print(f'{basename} was not proceed')
#
# print(single_plates)
# print(multi_plates)
# print(no_plates)
#
# print(classes)
# print(errors)
# print(minus)
# print(plus)
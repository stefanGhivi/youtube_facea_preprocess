import os
import numpy as np
import cv2
from sklearn import cluster
from sklearn.model_selection import StratifiedKFold

original_dataset_path = '/home/ghivi/yt_dataset'
raw_dataset_path = '/home/ghivi/Raw'
processed_dataset_path = '/home/ghivi/Processed'


def create_dirs(save_path, subdirs):
    for subdir in subdirs:
        dir_path = os.path.join(save_path, subdir)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)


def resize_2d_coords(old_width, old_height, new_width, new_height, points):
    x_factor = new_width / old_width
    y_factor = new_height / old_height
    scaling_matrix = np.array([[x_factor, 0, 0], # sx  0   0
                               [0, y_factor, 0], # 0   sy  0
                               [0, 0, 1]])       # 0   0   1

    points_matrix = points.transpose()
    points_matrix = np.vstack((points_matrix, np.ones(len(points)))) # [x_coords,
                                                                     #  y_coords,
                                                                     #  1]

    scaled_points = scaling_matrix @ points_matrix
    scaled_points = np.delete(scaled_points, obj=2, axis=0)
    scaled_points = scaled_points.transpose()

    return scaled_points


def draw(img, bbox, kps):
    cv2.rectangle(img, tuple(bbox[0].astype(int)), tuple(bbox[3].astype(int)), (0,0,255), 1)
    for pt in kps:
        cv2.circle(img, tuple(pt.astype(int)), 1, (255,0,0), 1)


def task_1(original_path, save_path):
    create_dirs(save_path, ['images', 'bounding_boxes', 'landmarks2D'])
    batches = next(os.walk(original_path))[1]
    for batch in batches:
        batch_path = os.path.join(original_path, batch)
        samples = os.listdir(batch_path)
        for sample in samples:
            sample_path = os.path.join(batch_path, sample)
            sample_data = np.load(sample_path)
            imgs = sample_data['colorImages']
            bboxes = sample_data['boundingBox']
            lm2d = sample_data['landmarks2D']
            lm2d = np.moveaxis(lm2d, -1, 0)
            bboxes = np.moveaxis(bboxes, -1, 0)
            imgs = np.moveaxis(imgs, -1, 0)

            assert(len(imgs) == len(bboxes) == len(lm2d))
            for index in range(len(imgs)):
                img = imgs[index]
                img = img[..., ::-1]  #rgb2bgr
                cv2.imwrite(os.path.join(save_path, 'images', sample + str(index) + '.png'), img)

                bbox = bboxes[index]
                np.save(os.path.join(save_path, 'bounding_boxes', sample + str(index) + '.npy'), bbox)

                keypoints2d = lm2d[index]
                np.save(os.path.join(save_path, 'landmarks2D', sample + str(index) + '.npy'), keypoints2d)


def task_2(original_path, save_path, new_width, new_height):

    create_dirs(save_path, ['images', 'bounding_boxes', 'landmarks2D'])
    images_path = os.path.join(original_path, 'images')
    images = os.scandir(images_path)
    bboxes_path = os.path.join(original_path, 'bounding_boxes')
    lms_path = os.path.join(original_path, 'landmarks2D')
    for image in images:
        img = cv2.imread(os.path.join(images_path, image.name))
        old_width = img.shape[1]
        old_height = img.shape[0]
        new_dim = (new_width, new_height)
        resized = cv2.resize(img, new_dim, interpolation=cv2.INTER_AREA)

        old_bbox = np.load(os.path.join(bboxes_path, image.name.replace('.png', '.npy')))
        new_bbox = resize_2d_coords(old_width, old_height, new_width, new_height, old_bbox)

        old_keypoints = np.load(os.path.join(lms_path, image.name.replace('.png', '.npy')))
        new_keypoints = resize_2d_coords(old_width, old_height, new_width, new_height, old_keypoints)

        draw(resized, new_bbox, new_keypoints)
        cv2.imwrite(os.path.join(save_path, 'images', image.name), resized)
        np.save(os.path.join(save_path, 'bounding_boxes', image.name.replace('.png', '.npy')), new_bbox)
        np.save(os.path.join(save_path, 'landmarks2D', image.name.replace('.png', '.npy')), new_keypoints)


def task_3(original_path):
    lm2d_path = os.path.join(original_path, 'landmarks2D')
    landmarks = os.scandir(lm2d_path)
    all_kps = None
    names = []
    for lm in landmarks:
        names.append(lm.name.replace('.npy', ''))
        keypoints = np.load(os.path.join(lm2d_path, lm.name))
        keypoints = np.expand_dims(keypoints, axis=2)
        if all_kps is None:
            all_kps = keypoints
        else:
            all_kps = np.concatenate((all_kps, keypoints), axis=2)

    kps_centered = all_kps - np.tile(all_kps.mean(axis=0), [68, 1, 1])

    kps_normlized = kps_centered / np.tile(
        np.sqrt((kps_centered ** 2).sum(axis=1)).mean(axis=0), [68, 2, 1])

    numClusters = 16
    normalizedShapesTable = np.reshape(kps_normlized, [68 * 2, kps_normlized.shape[2]]).T

    shapesModel = cluster.KMeans(n_clusters=numClusters, n_init=5, random_state=1).fit(normalizedShapesTable[::2, :])
    clusterAssignment = shapesModel.predict(normalizedShapesTable)

    skf = StratifiedKFold(n_splits=10)
    folds = skf.split(names, clusterAssignment)
    return folds

task_1(original_dataset_path, raw_dataset_path)
task_2(raw_dataset_path, processed_dataset_path, 200, 200)
folds = task_3(raw_dataset_path)
import os
import cv2
import numpy as np


def resize(img_array, align_mode):
    _height = len(img_array[0])
    _width = len(img_array[0][0])
    for i in range(1, len(img_array)):
        img = img_array[i]
        height = len(img)
        width = len(img[0])
        if align_mode == "smallest":
            if height < _height:
                _height = height
            if width < _width:
                _width = width
        else:
            if height > _height:
                _height = height
            if width > _width:
                _width = width

    for i in range(0, len(img_array)):
        img1 = cv2.resize(img_array[i], (_width, _height), interpolation=cv2.INTER_CUBIC)
        img_array[i] = img1

    return img_array, (_width, _height)


def images_to_video(data_root):
    filenames = [filename[:-4] for filename in os.listdir(os.path.join(data_root))]
    filenames = np.sort(filenames)

    img_array = []
    for filename in filenames:
        img = cv2.imread(os.path.join(data_root, filename + ".jpg"))
        img_array.append(img)

    img_array, size = resize(img_array, "largest")
    fps = 10
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(f"vis.mp4", fourcc, fps, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


def main():
    data_root = r"vis/counting_1"
    images_to_video(data_root)


if __name__ == "__main__":
    main()

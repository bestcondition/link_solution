from pathlib import Path

import cv2

from link_solution.draw_graph import get_random_sample


def gen_data(
        img_dir_path,
        label_dir_path,
        n: int,
):
    len_n = len(str(n))
    img_dir_path = Path(img_dir_path)
    label_dir_path = Path(label_dir_path)
    img_dir_path.mkdir(parents=True, exist_ok=True)
    label_dir_path.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        name = f"img_{i:0>{len_n}}"
        print(name)
        gen_one_data(
            str(img_dir_path / f"{name}.png"),
            str(label_dir_path / f"{name}.txt"),
        )


def gen_one_data(
        img_file_path,
        label_file_path,
):
    img, txt = get_random_sample()
    cv2.imwrite(img_file_path, img)
    with open(label_file_path, "w") as f:
        f.write(txt)


def main():
    gen_data(
        'data_set/train/images',
        'data_set/train/labels',
        10,
    )
    gen_data(
        'data_set/val/images',
        'data_set/val/labels',
        3,
    )


if __name__ == '__main__':
    main()

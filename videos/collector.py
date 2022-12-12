import natsort
import os, shutil, imageio, cv2


src_folder = "../app/runs/detect/221118_exp11_stable/"
data_list = os.listdir(src_folder)


def make_video(mode: str):
    if opt.mp4:
        make_mp4(mode)
    if opt.gif:
        make_gif(mode)


def make_gif(mode):
    if f"{mode}.gif" in os.listdir("./"):
        os.remove(f"{mode}.gif")

    filenames = os.listdir(f"./{mode}")
    filenames = natsort.natsorted(filenames)

    images = []
    for filename in filenames:
        images.append(imageio.imread(f"./{mode}/" + filename))
    imageio.mimsave(f"{mode}.gif", images)


def make_mp4(mode):
    fps = 10
    video = cv2.VideoWriter(
        f"{mode}.mp4",
        cv2.VideoWriter_fourcc("m", "p", "4", "v"),
        fps,
        (480, 640),
    )

    filenames = os.listdir(f"./{mode}")
    filenames = natsort.natsorted(filenames)
    for filename in filenames:
        video.write(cv2.imread(os.path.join(mode, filename)))

    cv2.destroyAllWindows()
    video.release()


def clear_folder(target: str):
    if os.path.exists(f"./{target}"):
        shutil.rmtree(f"./{target}")
    os.mkdir(f"./{target}")


# 1. d_detect
def collect_d_detect():
    TARGET = "d_detect"
    clear_folder(TARGET)

    for el in data_list:
        if TARGET in el:
            shutil.copy(src_folder + el, f"{TARGET}/" + el[25:])
    make_video(TARGET)


# 2. detect
def collect_detect():
    TARGET = "detect"
    clear_folder(TARGET)

    for el in data_list:
        if TARGET in el and "d_" not in el:
            shutil.copy(src_folder + el, f"{TARGET}/" + el[25:])
    make_video(TARGET)


# 3. depth
def collect_depth():
    TARGET = "depth"
    clear_folder(TARGET)

    for el in data_list:
        if TARGET in el:
            shutil.copy(src_folder + el, f"{TARGET}/" + el[25:])
    make_video(TARGET)


# 4. original image
def collect_jpg():
    TARGET = "jpg"
    clear_folder(TARGET)

    for el in data_list:
        if (
            TARGET in el
            and "detect" not in el
            and "depth" not in el
            and "tracking" not in el
        ):
            shutil.copy(src_folder + el, f"{TARGET}/" + el[25:])
    make_video(TARGET)


# 5. road segmentation
def collect_road():
    TARGET = "segmentation"
    clear_folder(TARGET)

    src_folder = "../../Road-Segmentation/data/"
    segmentation_list = os.listdir(src_folder)
    for el in segmentation_list:
        shutil.copy(src_folder + el, f"{TARGET}/" + el)
    make_video(TARGET)


def main():
    collect_d_detect()
    collect_detect()
    collect_depth()
    collect_jpg()
    collect_road()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mp4", action="store_true")
    parser.add_argument("--gif", action="store_true")
    opt = parser.parse_args()
    main()

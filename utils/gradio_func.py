import os
import cv2


def create_dir(dir_path):
    if os.path.isdir(dir_path):
        os.system(f"rm -r {dir_path}")
    os.makedirs(dir_path)


def get_meta(input_video, input_imgs, fps):
    if input_video is not None:
        video_name = os.path.basename(input_video).split('.')[0]
        dir_path = os.path.join('./temp', video_name)
        create_dir(dir_path)

        print(f"extracting imported video to {dir_path}")
        os.system(f"ffmpeg -i {input_video} -r {str(fps)} {dir_path}/frame_%06d.jpg")

        first_frame_dir = sorted(os.listdir(dir_path))[0]
        first_frame = cv2.imread(os.path.join(dir_path, first_frame_dir))
        first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)

        return first_frame, first_frame, first_frame

    elif input_imgs is not None:
        file_name = input_imgs.name.split('/')[-1].split('.')[0]
        dir_path = os.path.join('./temp', file_name)
        create_dir(dir_path)

        print(f"extracting imported images to {dir_path}")
        os.system(f'unzip {input_imgs.name} -d ./temp ')

        first_frame_dir = sorted(os.listdir(dir_path))[0]
        first_frame = cv2.imread(os.path.join(dir_path, first_frame_dir))
        first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)

        return first_frame, first_frame, first_frame

    else:
        return None, None, None

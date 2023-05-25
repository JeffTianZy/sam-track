import os
import cv2
import copy
import numpy as np
import gradio as gr

from .sam_func import object_track, auto_seg, post_process, init_sam
from .gradio_func import get_meta, create_dir


def clean_temp():
    os.system(f'rm -r ./temp/*')
    return None, None, None, None, None, None, [[], []]


class SamTracker:
    def __init__(self, tracker='csrt'):
        self.tracker = cv2.legacy.MultiTracker_create()
        if tracker == 'csrt':
            self.sub_tracker = cv2.legacy.TrackerCSRT_create()
        else:
            raise NotImplementedError
        self.predictor = init_sam()
        self.bboxs = []


def init_models_CSRT():
    print('Initializing CSRT tracker...')
    sam_tracker = SamTracker(tracker='csrt')
    print('Initialized!')
    return sam_tracker


def click_get_cord(click_stack, first_frame, evt: gr.SelectData):
    print('Click!')

    click_stack[0].append(evt.index[0])
    click_stack[1].append(evt.index[1])

    if len(click_stack[0]) % 2 == 0:
        color = tuple(map(int, np.random.randint(0, 255, size=(3,))))
        first_frame = cv2.rectangle(first_frame, (click_stack[0][-2], click_stack[1][-2]),
                                    (click_stack[0][-1], click_stack[1][-1]), color, 2)

    return click_stack, first_frame


def save_roi(sam_tracker, click_stack, origin_frame, roi_frame):
    print('Save!')
    if sam_tracker is None:
        raise ModuleNotFoundError('User should select and init tracking model first')

    for i in range(len(click_stack[0]) // 2):
        sam_tracker.bboxs.append(
            [click_stack[0][2 * i], click_stack[1][2 * i],
             abs(click_stack[0][2 * i + 1] - click_stack[0][2 * i]), abs(click_stack[1][2 * i + 1] - click_stack[0][2 * i])])

        color = tuple(map(int, np.random.randint(0, 255, size=(3,))))
        roi_frame = cv2.rectangle(roi_frame, (click_stack[0][2 * i], click_stack[1][2 * i]),
                                  (click_stack[0][2 * i + 1], click_stack[1][2 * i + 1]), color, 2)
    click_stack = [[], []]
    first_frame_painted = origin_frame

    return sam_tracker, click_stack, first_frame_painted, origin_frame, roi_frame


def undo_click(click_stack, painted_first_frame, origin_frame):
    print('Undo!')
    new_frame = copy.deepcopy(origin_frame)

    if len(click_stack[0]) % 2 != 0:
        click_stack[0] = click_stack[0][:-1]
        click_stack[1] = click_stack[1][:-1]
        return click_stack, painted_first_frame

    elif len(click_stack[0]) > 0:
        click_stack[0] = click_stack[0][:-2]
        click_stack[1] = click_stack[1][:-2]
        for i in range(len(click_stack[0]) // 2):
            color = tuple(map(int, np.random.randint(0, 255, size=(3,))))
            painted_first_frame = cv2.rectangle(new_frame, (click_stack[0][2 * i], click_stack[1][2 * i]),
                                                (click_stack[0][2 * i + 1], click_stack[1][2 * i + 1]), color, 2)

    return click_stack, painted_first_frame


def clear_click(origin_frame):
    click_stack = [[], []]
    return click_stack, origin_frame


def tracking_objects(sam_tracker, input_video, input_img_seq):
    print('Start tracking!')
    if input_video is not None:
        video_name = os.path.basename(input_video).split('.')[0]
        dir_path = f'./temp/{video_name}'
        file_list = sorted([os.path.join(dir_path, file_dir) for file_dir in os.listdir(dir_path)])
    elif input_img_seq is not None:
        file_name = input_img_seq.name.split('/')[-1].split('.')[0]
        file_path = f'./temp/{file_name}'
        file_list = sorted([os.path.join(file_path, img_name) for img_name in os.listdir(file_path)])
        video_name = file_name
    else:
        return None, None

    tracking_result_dir = f"./saves/tracking_results/{video_name}"
    create_dir(tracking_result_dir)

    io_args = {
        'tracking_result_dir': tracking_result_dir,
        'output_video': f'{tracking_result_dir}/{video_name}_seg.mp4',  # keep same format as input video
        'output_gif': f'{tracking_result_dir}/{video_name}_seg.gif',
    }

    input_tracking(sam_tracker, file_list, io_args, video_name)


def input_tracking(sam_tracker, file_list, io_args, video_name):
    initialized = False

    pred_saves = {i: [sam_tracker.bboxs[i]] for i in range(len(sam_tracker.bboxs))}

    for i, frame_file in enumerate(file_list):
        print(f"tracking on frame {i}, total frame: {len(file_list)}", flush=True)
        frame = cv2.imread(frame_file)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if not initialized:
            for bbox in sam_tracker.bboxs:
                sam_tracker.tracker.add(sam_tracker.sub_tracker, frame, bbox)
            initialized = True
        else:
            success, boxes = sam_tracker.tracker.update(frame)
            for j, newbox in enumerate(boxes):
                pred_saves[j].append(newbox)

    print(pred_saves)

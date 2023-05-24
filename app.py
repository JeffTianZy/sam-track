import gradio as gr
from utils import *

with gr.Blocks() as demo:
    ###################################
    ############ Front-end ############
    ###################################
    with gr.Row():
        with gr.Column():
            tab_video_input = gr.Tab(label='Video Input')
            with tab_video_input:
                input_video = gr.Video(label='Input Video')

            tab_imgs_input = gr.Tab(label='Images Input')
            with tab_imgs_input:
                with gr.Row():
                    input_imgs = gr.File(label='Input Images')
                    with gr.Column(scale=0.25):
                        extract_button = gr.Button(value='extract')
                        fps = gr.Slider(label='useless FPS', minimum=1, maximum=60, value=10, step=1)
            input_first_frame = gr.Image(label='Segment result of first frame', interactive=True, tool='select')
            with gr.Row():
                select_current_roi = gr.Button(value='select current ROI', interactive=True)
                undo = gr.Button(value='Undo', interactive=True)
                save_all_roi = gr.Button(value='Save all ROIs', interactive=True)

            tracking_model = gr.Tab(label='Select T racking Model')
            with tracking_model:
                with gr.Row():
                    track_a = gr.Button(value='Method A', interactive=True)
                    track_b = gr.Button(value='Method B', interactive=True)
                    track_c = gr.Button(value='Method C', interactive=True)

        with gr.Column():
            first_frame_with_roi = gr.Image(label='First frame with selected ROIs', interactive=False)
            start_tracking = gr.Button(value='Start tracking', interactive=True)
            demo_tracking = gr.Tab(label='Tracking Demo')
            with demo_tracking:
                tracking_result = gr.Video(label='Tracking Results', interactive=False)
                with gr.Row():
                    save_vid = gr.Button(value='Save video', interactive=True)
                    save_gif = gr.Button(value='Save GIF', interactive=True)

    with gr.Row():
        with gr.Column(scale=0.5):
            first_frame_seg = gr.Image(label='First frame segmentation result', interactive=False)
        frame_results = gr.Gallery(label='All frame results', show_label=True)

    with gr.Row():
        with gr.Column(scale=0.25):
            post_process_button = gr.Button(value='Do post processing')
            use_erosion = gr.Radio(choices=['True', 'False'], value='True', label='Use erosion', interactive=True)
            kernel_size = gr.Slider(label='Kernel size', minimum=1, maximum=15, value=3, step=1)
            use_smoothing = gr.Radio(choices=['True', 'False'], value='True', label='Use boundary smoothing', interactive=True)
            use_filling = gr.Radio(choices=['True', 'False'], value='True', label='Use Hole filling', interactive=True)
        regenerate = gr.Tab(label='Regenerate Corrupted Frames')
        with regenerate:
            with gr.Row():
                input_regen_frame = gr.Image(label='Regenerate segmentation', interactive=True, tool='select')
                auto_select = gr.Button(value='Auto selection', interactive=True).style(full_width=False)
            with gr.Row():
                select_roi_regen = gr.Button(value='Select current ROI', interactive=True)
                undo_regen = gr.Button(value='Undo', interactive=True)
                save_all_roi_regen = gr.Button(value='Save all ROIs', interactive=True)
    ###################################
    ############ Back-end #############
    ###################################
    input_video.change(fn=get_meta_from_video, inputs=[input_video], outputs=[input_first_frame])


demo.launch()

import gradio as gr
from utils import *

with gr.Blocks() as demo:

    ###################################
    ############ Front-end ############
    ###################################

    click_stack = gr.State([[], []])  # Storage clicks status
    origin_frame = gr.State(None)
    sam_tracker = gr.State(None)

    saving_dirs = gr.State({})

    tracking_model = gr.Tab(label='Select and Init Tracking Model')
    with gr.Row():
        with tracking_model:
            with gr.Row():
                track_a = gr.Button(value='CSRT-tracking', interactive=True)
                track_b = gr.Button(value='Wait Update', interactive=False)
                track_c = gr.Button(value='Wait Update', interactive=False)
                track_d = gr.Button(value='Wait Update', interactive=False)
                track_e = gr.Button(value='Wait Update', interactive=False)
                clean_cache = gr.Button(value='Clean temp', variant="primary")

    with gr.Row():
        with gr.Column():
            tab_video_input = gr.Tab(label='Video Input')
            with tab_video_input:
                input_video = gr.Video(label='Input Video')

            tab_imgs_input = gr.Tab(label='Images Input')
            with tab_imgs_input:
                with gr.Row():
                    input_imgs = gr.File(label='Input Images')

            with gr.Row():
                fps = gr.Slider(label='FPS', minimum=1, maximum=60, value=10, step=1)
                extract_button = gr.Button(value='extract').style(full_width=False)

            input_first_frame = gr.Image(label='Segment result of first frame', interactive=False)
            with gr.Row():
                select_current_roi = gr.Button(value='select current ROI', interactive=True)
                undo = gr.Button(value='Undo', interactive=True)
                clear_all = gr.Button(value='Clear all', interactive=True)

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
            use_smoothing = gr.Radio(choices=['True', 'False'], value='True', label='Use boundary smoothing',
                                     interactive=True)
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

    clean_cache.click(fn=clean_temp,
                      inputs=[],
                      outputs=[input_video,
                               input_imgs,
                               sam_tracker,
                               saving_dirs,
                               input_first_frame,
                               origin_frame,
                               first_frame_with_roi,
                               click_stack
                               ])

    tab_video_input.select(fn=clean_temp,
                           inputs=[],
                           outputs=[input_video,
                                    input_imgs,
                                    sam_tracker,
                                    saving_dirs,
                                    input_first_frame,
                                    origin_frame,
                                    first_frame_with_roi,
                                    click_stack
                                    ])

    tab_imgs_input.select(fn=clean_temp,
                          inputs=[],
                          outputs=[input_video,
                                   input_imgs,
                                   sam_tracker,
                                   saving_dirs,
                                   input_first_frame,
                                   origin_frame,
                                   first_frame_with_roi,
                                   click_stack
                                   ])

    extract_button.click(fn=get_meta,
                         inputs=[input_video, input_imgs, fps],
                         outputs=[input_first_frame, origin_frame, first_frame_with_roi])

    track_a.click(fn=init_models_CSRT,
                  inputs=[],
                  outputs=[sam_tracker])

    input_first_frame.select(fn=click_get_cord,
                             inputs=[click_stack, input_first_frame],
                             outputs=[click_stack, input_first_frame])

    select_current_roi.click(fn=save_roi,
                             inputs=[sam_tracker, click_stack, origin_frame, first_frame_with_roi],
                             outputs=[sam_tracker, click_stack, input_first_frame, origin_frame, first_frame_with_roi])

    undo.click(fn=undo_click,
               inputs=[click_stack, input_first_frame, origin_frame],
               outputs=[click_stack, input_first_frame])

    clear_all.click(fn=clear_click,
                    inputs=[origin_frame],
                    outputs=[click_stack, input_first_frame])

    start_tracking.click(fn=tracking_objects,
                         inputs=[sam_tracker, input_video, input_imgs, fps],
                         outputs=[tracking_result, saving_dirs],
                         queue=True)

    save_vid.click(fn=save_visualization,
                   inputs=[saving_dirs, 'video'],
                   outputs=[])

    save_gif.click(fn=save_visualization,
                   inputs=[saving_dirs, 'gif'],
                   outputs=[])

demo.queue()
demo.launch()

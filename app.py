import gradio as gr
from utils import *

with gr.Blocks() as demo:
    ###################################
    ############ Front-end ############
    ###################################
    with gr.Row():
        with gr.Column(scale=0.5):
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

            tab_everything = gr.Tab(label='Everything')
            with tab_everything:
                with gr.Row():
                    seg_everything_first_frame = gr.Button(value='Segment-Anything for first frame', interactive=True)
    ###################################
    ############ Back-end #############
    ###################################
    input_video.change(fn=get_meta_from_video, inputs=[input_video], outputs=[input_first_frame])


demo.launch()

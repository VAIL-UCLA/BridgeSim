import argparse
import copy
import functools
import os
import pathlib
import pickle

import gradio as gr
import numpy as np
import torch
import uuid
from omegaconf import OmegaConf

from bmt.dataset.preprocess_action_label import TurnAction
from bmt.dataset.preprocessor import preprocess_scenario_description_for_motionlm
from bmt.gradio_ui.plot import plot_gt, plot_pred, create_animation_from_pred, create_animation_from_gt
from bmt.utils import REPO_ROOT
from bmt.utils import utils

os.environ['GRADIO_TEMP_DIR'] = str(REPO_ROOT / "gradio_tmp")

dpi = 100

parser = argparse.ArgumentParser()
parser.add_argument("--share", action="store_true", help="Enable sharing")
parser.add_argument("--default_ckpt", "--default_model", "--ckpt", type=str, default="../ckpt/last.ckpt")
parser.add_argument("--default_data", "--data", type=str, default="data/20scenarios")
parser.add_argument("--port", type=int, default=7860)
parser.add_argument("--title", type=str, default="Motion Generation")
parser.add_argument("--default_config_path", type=str, default="motion_default.yaml")

parser.add_argument(
    "--display_video", "--video", action="store_true", help="by default display static images, can display mp4 video"
)
args = parser.parse_args()

default_config_path = args.default_config_path
default_config = OmegaConf.load(REPO_ROOT / "cfgs" / default_config_path)

OmegaConf.set_struct(default_config, False)
default_config.PREPROCESSING.keep_all_data = True
default_config.ROOT_DIR = str(REPO_ROOT.resolve())
OmegaConf.set_struct(default_config, True)

DEFAULT_DATA_PATH = args.default_data
DEFAULT_MODEL = args.default_ckpt or None

NUM_OF_MODES = 6
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LENGTH = 1000


class State:
    model = None
    model_path = None
    config: dict = default_config
    dataset_path: pathlib.Path = REPO_ROOT / DEFAULT_DATA_PATH

    scenario = None

    raw_data_files = None
    data_files = None

    raw_data_dict = None
    data_dict = None

    default_config: dict = default_config

    # click support
    sel_point = None
    xlim, ylim = None, None
    fig_width, fig_height = None, None  # in inches
    fig_dpi = None
    bbox_x0, bbox_y0, bbox_w, bbox_h = None, None, None, None
    original_img = None
    modified_img = None


state = State()


def ckpt_callback(ckpt_path):
    from bmt.models.motionlm_lightning import MotionLMLightning

    msg = "Failed!"
    temperature = 1.0
    safe_agents = ""
    turn_agents = ""
    main_vis = None
    sampling_method = "topp"

    if ckpt_path.lower() == "debug":
        try:
            config = copy.deepcopy(state.default_config)
            OmegaConf.set_struct(config, False)
            config.MODEL.D_MODEL = 32
            config.MODEL.NUM_DECODER_LAYERS = 1
            config.MODEL.NUM_ATTN_LAYERS = 1
            # config.ACTION_LABEL.USE_SAFETY_LABEL = True
            # config.ACTION_LABEL.USE_ACTION_LABEL = True
            OmegaConf.set_struct(config, True)
            model = MotionLMLightning(config)
            model = model.to(device)
            msg = "DEBUG MODEL LOADED!"
            config = model.config
            temperature = config.SAMPLING.TEMPERATURE
            state.model = model
            state.config = config
            sampling_method = config.SAMPLING.SAMPLING_METHOD
        except Exception as e:
            print("Error: ", e)
            msg = "Failed to load DEBUG model!"

        return [msg, sampling_method, temperature, main_vis] + [""] * 7 + [0.0] * 5

    ckpt_path = ckpt_path.replace("\\", "")
    path = pathlib.Path(ckpt_path)
    path = REPO_ROOT / path

    if path.is_dir():
        path = path / "last.ckpt"

    print("Loading model from: ", path.absolute())
    if not path.exists():
        msg = "{} does not exist!".format(path)
        return [msg, sampling_method, temperature, main_vis] + [""] * 7 + [0.0] * 5

    try:
        # model = utils.get_model(
        #     config=None, checkpoint_path=path, device=device, default_config=default_config_path
        # ).eval()
        model = utils.get_model(checkpoint_path=path).eval()
        msg = "Model loaded successfully!"
        config = model.config
        temperature = config.SAMPLING.TEMPERATURE
        state.model = model
        state.config = config
        sampling_method = config.SAMPLING.SAMPLING_METHOD

        infer_heading_params = (
            config.TOKENIZATION.MIN_SPEED,  #or 0.0,
            config.TOKENIZATION.MAX_HEADING_DIFF,
            config.TOKENIZATION.MIN_DISPLACEMENT_INIT,  # or 0.0,
            config.TOKENIZATION.MIN_DISPLACEMENT,  #or 0.0,
            config.TOKENIZATION.SMOOTH_FACTOR,  #or 1.0,
        )
        
        config.eval_backward_model = False

    except Exception as e:
        print("Error: ", e)
        raise e
        msg = "Failed to load model!"
        infer_heading_params = [0.0] * 5

    return [msg, sampling_method, temperature, main_vis] + [""] * 7 + list(infer_heading_params)


def on_dataset_path_submit(path):

    print(state, type(state))

    FAILED_MSG = "Failed!"

    path = pathlib.Path(path)
    path = REPO_ROOT / path

    if not path.exists():
        return FAILED_MSG

    if not path.is_dir():
        return FAILED_MSG

    state.dataset_path = path
    print(state.dataset_path)

    files = os.listdir(path)
    files = [f for f in files if f.endswith(".pkl")]
    print("Files: ", files)

    if not hasattr(state, "count"):
        state.count = 0
    state.count += 1

    return [
        "Dataset with {} Scenarios Listed!".format(len(files)),
        gr.FileExplorer(
            file_count="single",
            root_dir=state.dataset_path,
            # root_dir=path,
            glob="**/*.pkl",
            scale=1
            # label="UPDATED={}".format(state.count),
            # interactive=True
        )
    ]


def on_data_file_name_search(search):
    return gr.FileExplorer(file_count="single", root_dir=state.dataset_path, glob=f"**/*{search}*.pkl", scale=1)


def on_data_file_select(file_path):
    if not file_path:
        return (None, ) + ("", ) * 2

    file_path = pathlib.Path(file_path)
    assert state.dataset_path is not None
    file_path = state.dataset_path / file_path

    with open(file_path, "rb") as f:
        data = pickle.load(f)

    state.scenario = data
    scenario_data_dict = preprocess_scenario_description_for_motionlm(
        scenario=data,
        config=state.config,
        in_evaluation=True,
        keep_all_data=True,
        # cache=None,
        tokenizer=state.model.model.tokenizer,
        backward_prediction=False
    )

    state.raw_data_dict = scenario_data_dict

    if args.display_video:
        gt_gif_path = create_animation_from_gt(
            scenario_data_dict,
            save_path=str(REPO_ROOT / "gradio_tmp" / "gt_animation_{}.mp4".format(uuid.uuid4())),
            dpi=dpi,
            # draw_non_ooi=False
        )
        return (
            gr.Video(value=gt_gif_path, label="Ground Truth Trajectories",
                     autoplay=True),  # Use gr.Video instead of gr.Image
            "",
            "",
        )
    else:
        img, info_dict = plot_gt(scenario_data_dict, get_info=True)
        state.original_img = img
        state.xlim = info_dict["xlim"]
        state.ylim = info_dict["ylim"]
        state.fig_width, state.fig_height = info_dict["fig_size"]
        state.fig_dpi = info_dict["fig_dpi"]
        state.bbox_x0 = info_dict["bbox_x0"]
        state.bbox_y0 = info_dict["bbox_y0"]
        state.bbox_w = info_dict["bbox_w"]
        state.bbox_h = info_dict["bbox_h"]
        return (gr.Image(value=img, label=data["id"]), ) + ("", ) * 2


def on_generate_button_click(
    # MIN_SPEED, MAX_HEADING_DIFF, MIN_DISPLACEMENT_INIT, MIN_DISPLACEMENT, smooth_factor,
    sampling_method,
    temperature,
    seed,
    agents_safe_0,
    agents_safe_1,
    # agents_turn_stop, agents_turn_straight, agents_turn_left, agents_turn_right,
    # agents_turn_uturn,
    only_draw_gt,
    only_draw_detokenized,
    draw_backward_prediction=False
):

    # TODO: Seed is not respect!
    # TODO: Seed is not respect!
    # TODO: Seed is not respect!

    assert sampling_method in ["softmax", "topp"], "Invalid sampling method! {}".format(sampling_method)

    if state.scenario is None:
        return (
            None,
            "Data is not loaded!",
        ) + (None, ) * 2

    if (not state.config.get("BACKWARD_PREDICTION", False)) and draw_backward_prediction:
        print("BACKWARD_PREDICTION is not enabled in the config!")
        return (
            None,
            "BACKWARD_PREDICTION is not enabled in the config!",
        ) + (None, ) * 2

    model = state.model
    if model is None and not only_draw_gt:
        return (
            None,
            "Model is not loaded!",
        ) + (None, ) * 2

    # if smooth_factor:
    config = copy.deepcopy(state.config)
    OmegaConf.set_struct(config, False)
    # state.config.TOKENIZATION.SMOOTH_FACTOR = None
    # state.config.TOKENIZATION.MIN_DISPLACEMENT = MIN_DISPLACEMENT
    # state.config.TOKENIZATION.MIN_DISPLACEMENT_INIT = MIN_DISPLACEMENT_INIT
    # state.config.TOKENIZATION.MAX_HEADING_DIFF = MAX_HEADING_DIFF
    # state.config.TOKENIZATION.MIN_SPEED = MIN_SPEED
    OmegaConf.set_struct(config, True)



    print("draw_backward_prediction", draw_backward_prediction)

    data_dict = preprocess_scenario_description_for_motionlm(
        scenario=copy.deepcopy(state.scenario),
        config=config,
        in_evaluation=True,
        keep_all_data=True,
        # cache=None,
        backward_prediction=draw_backward_prediction,
        tokenizer=model.model.tokenizer
    )

    # # ===== Overwrite the labels =====
    # def _parse_agents(agents_str):
    #     is_raw = False
    #     if agents_str:
    #         if agents_str.startswith("[RAW]"):
    #             agents_str = agents_str[5:]
    #             is_raw = True
    #             if not agents_str:
    #                 return [], is_raw
    #         return [int(agent.strip()) for agent in agents_str.split(",")], is_raw
    #     else:
    #         is_raw = True
    #         return [], is_raw

    # def _fill_label(data_dict, label_name, agents, label_value):
    #     data_dict[label_name] = -1
    #     if agents:
    #         label = data_dict[label_name]
    #         for aid in agents:
    #             assert 0 <= aid < label.shape[0], (aid, label.shape)
    #             data_dict[label_name][aid] = label_value

    # if 'decoder/label_safety' in data_dict:
    #     agents_safe_0, agents_safe_0_is_raw = _parse_agents(agents_safe_0)
    #     _fill_label(data_dict, 'decoder/label_safety', agents_safe_0, 0)
    #     agents_safe_1, agents_safe_1_is_raw = _parse_agents(agents_safe_1)
    #     _fill_label(data_dict, 'decoder/label_safety', agents_safe_1, 1)
    #     print(f"the input safety label for inference:", data_dict['decoder/label_safety'])
    # else:
    #     print("decoder/label_safety is not in the data_dict!")

    if only_draw_gt:
        output_dict = data_dict

        if args.display_video:
            video_path = str(REPO_ROOT / "gradio_tmp" / "gt_animation_{}.mp4".format(uuid.uuid4()))
            video_path = create_animation_from_gt(output_dict, save_path=video_path, dpi=dpi)
            print("gt_mp4_path", video_path)
        else:
            img = plot_gt(output_dict)

    else:
        input_data_dict = utils.numpy_to_torch(data_dict, device)

        # Extend the batch dim:
        input_data_dict = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v for k, v in input_data_dict.items()}
        input_data_dict["in_evaluation"] = torch.tensor([1], dtype=bool).to(device)
        input_data_dict["in_backward_prediction"] = torch.tensor([draw_backward_prediction], dtype=bool).to(device)

        if not only_draw_detokenized:
            print("NO only_draw_detokenized!")
            with torch.no_grad():
                output_dict = model.model.autoregressive_rollout(
                    input_data_dict,
                    # num_decode_steps=None,
                    sampling_method=sampling_method,
                    temperature=temperature,
                    backward_prediction=draw_backward_prediction
                )

            output_dict = model.model.tokenizer.detokenize(
                output_dict, detokenizing_gt=False, backward_prediction=draw_backward_prediction, flip_wrong_heading=config.TOKENIZATION.FLIP_WRONG_HEADING,
            )
        else:
            # ===== DEBUG =====
            output_dict = input_data_dict
            output_dict["decoder/output_action"] = output_dict["decoder/target_action"]
            fill_zero = ~output_dict["decoder/target_action_valid_mask"]
            output_dict["decoder/input_action_valid_mask"][fill_zero] = False

            output_dict = model.model.tokenizer.detokenize(output_dict, detokenizing_gt=True)

        output_dict = {
            k: (v.squeeze(0).cpu().numpy() if isinstance(v, torch.Tensor) else v)
            for k, v in output_dict.items()
        }

        if args.display_video:
            video_path = str(REPO_ROOT / "gradio_tmp" / "pred_animation_{}.mp4".format(uuid.uuid4()))
            video_path = create_animation_from_pred(output_dict, save_path=video_path, dpi=dpi)
            print("predict gif path:", video_path)

        else:
            img = plot_pred(output_dict)

    # Postprocess
    if "decoder/label_safety" in output_dict:
        safety_label = output_dict["decoder/label_safety"]
        if agents_safe_0_is_raw:
            aid = [str(v) for v in (safety_label == 0).nonzero()[0]]
            agents_safe_0 = "[RAW]" + ",".join(aid)
        if agents_safe_1_is_raw:
            aid = [str(v) for v in (safety_label == 1).nonzero()[0]]
            agents_safe_1 = "[RAW]" + ",".join(aid)

    if args.display_video:
        return (
            # gr.Image(value=img, label=state.scenario["id"]),
            gr.Video(
                value=video_path,
                label="Generated Trajectory Prediction",
                show_download_button=True,
                width=LENGTH,
                height=LENGTH,
                interactive=False,
                format="mp4",
                autoplay=True,
            ),  # Use gr.Video instead of gr.Image for GIF
            "Scenario Generated!",
            ", ".join([str(v) for v in agents_safe_0]) if isinstance(agents_safe_0, list) else agents_safe_0,
            ", ".join([str(v) for v in agents_safe_1]) if isinstance(agents_safe_1, list) else agents_safe_1,
        )
    else:
        return (
            gr.Image(value=img, label=state.scenario["id"]),
            "Scenario Generated!",
            ", ".join([str(v) for v in agents_safe_0]) if isinstance(agents_safe_0, list) else agents_safe_0,
            ", ".join([str(v) for v in agents_safe_1]) if isinstance(agents_safe_1, list) else agents_safe_1,
        )


import cv2
from PIL import Image


def on_scenario_vis_click(event: gr.SelectData):
    # Get axes relative coordinates (0 to 1 within the axes)
    width, height = state.fig_width * state.fig_dpi, state.fig_height * state.fig_dpi
    if state.fig_dpi is not None:
        # Convert to relative figure coordinates (0 to 1 within figure)
        x_rel_fig = (event.index[0] - state.bbox_x0 * width) / (state.bbox_w * width)
        y_rel_fig = (event.index[1] - state.bbox_y0 * height) / (state.bbox_h * height)  # note that y-axis is inverted
        print(
            f"Data: ({event.index[0]:.2f}, {event.index[1]:.2f}), Relative to Figure: ({x_rel_fig:.2f}, {y_rel_fig:.2f})"
        )
        print(f"xlim: {state.xlim}, ylim: {state.ylim}")
        state.sel_point = (
            x_rel_fig * (state.xlim[1] - state.xlim[0]) + state.xlim[0],
            y_rel_fig * (state.ylim[0] - state.ylim[1]) + state.ylim[1]
        )
        print(state.sel_point)
        cv2_original = cv2.cvtColor(np.array(state.original_img), cv2.COLOR_BGR2RGB)
        cv2.circle(cv2_original, event.index, 10, (0, 0, 255), -1)
        state.modified_img = Image.fromarray(cv2.cvtColor(cv2_original, cv2.COLOR_BGR2RGB))
        return state.modified_img, state.sel_point
    else:
        print("Select a scenario first")
        return gr.update()


def on_clear_click():
    raise ValueError()
    state.sel_point = None
    state.modified_img = None
    return state.original_img, state.sel_point


# ============================================================
# ======================== GRADIO UI =========================
# ============================================================
with gr.Blocks(theme=gr.themes.Soft(text_size="lg"), title=args.title) as demo:
    with gr.Group():
        gr.Markdown("  ## Data")
        with gr.Row():
            with gr.Column(scale=3):
                inp = gr.Textbox(label="Path to Dataset Folder", value=DEFAULT_DATA_PATH)

            with gr.Column(scale=1):
                out = gr.Textbox(label="Status", placeholder="Enter to submit...")

        # gr.Markdown("## Visualization")
        with gr.Row(equal_height=True):  # Future release fix: https://github.com/gradio-app/gradio/pull/9577
            with gr.Column(scale=1):
                with gr.Group():
                    file_name_input = gr.Textbox(label="Search Scenario ID", max_lines=1)
                    file_explorer = gr.FileExplorer(
                        root_dir=state.dataset_path,
                        glob="**/*.pkl",
                        file_count="single",
                        interactive=True,
                        container=True,
                        max_height=900
                    )
            with gr.Column(scale=2):
                with gr.Row():
                    sel_point = gr.Textbox(
                        label="Selected Point",
                        interactive=False,
                        placeholder="Click on scenario map to select a point"
                    )
                    clear = gr.Button(value="Clear Selection", interactive=True, visible=True)

                if args.display_video:
                    gt_vis = gr.Video(
                        label="Ground Truth Trajectories",
                        show_download_button=True,
                        width=LENGTH,
                        height=LENGTH,
                        interactive=False,
                        format="mp4"
                    )
                else:
                    gt_vis = gr.Image(
                        label="Original Scenario",
                        show_download_button=True,
                        width=LENGTH,
                        height=LENGTH,
                        interactive=False
                    )

    with gr.Group():
        gr.Markdown("## Model")
        with gr.Row():
            with gr.Column(scale=3):
                if DEFAULT_MODEL:
                    ckpt_input = gr.Textbox(
                        label="Path to model checkpoint", value=DEFAULT_MODEL, placeholder="/home/.../last.ckpt"
                    )
                else:
                    ckpt_input = gr.Textbox(
                        label="Path to model checkpoint",
                        placeholder="/home/.../last.ckpt (Type 'debug' for debug model!)"
                    )
            with gr.Column(scale=1):
                ckpt_output = gr.Textbox(label="Status", placeholder="Enter to load...")

        gr.Markdown("## Visualization")
        with gr.Row():
            with gr.Column(scale=1):
                sampling_method = gr.Radio(label="Sampling Method", choices=["softmax", "topp"], value="topp")
                temperature = gr.Slider(
                    label="Sampling Temperature", minimum=0.0, maximum=2.0, step=0.1, value=1.0, interactive=True
                )
                seed = gr.Number(label="Seed (TODO: not used)", value=42, precision=0)

                gr.Markdown(
                    "### Agents ID to assign labels:\n1. Split by comma ','\n2. If empty, original labels are used and printed as `[RAW]`"
                )
                agents_safe_0 = gr.Textbox(label="label_safety = 0 (NO COLL)", interactive=True, placeholder="0, 1")
                agents_safe_1 = gr.Textbox(label="label_safety = 1 (W/ COLL)", interactive=True, placeholder="0, 1")

                generate_button = gr.Button(value="Generate")
                generate_backward_prediction_button = gr.Button(value="Generate Backward Prediction")
                draw_gt_button = gr.Button(value="Draw Original Scenario")
                draw_detok_button = gr.Button(value="Draw Raw Detokenized Scenario")

            with gr.Column(scale=2):
                main_vis_text = gr.Textbox(label="Status", placeholder="", interactive=False)
                if args.display_video:
                    main_vis = gr.Video(
                        label="Generated Scenario",
                        show_download_button=True,
                        width=LENGTH,
                        height=LENGTH,
                        format="mp4"
                    )
                else:
                    main_vis = gr.Image(
                        label="Generated Scenario", show_download_button=True, width=LENGTH, height=LENGTH
                    )

    inp.submit(on_dataset_path_submit, inputs=inp, outputs=[out, file_explorer])
    file_name_input.change(on_data_file_name_search, inputs=file_name_input, outputs=file_explorer)
    file_explorer.change(
        on_data_file_select,
        inputs=file_explorer,
        outputs=[
            gt_vis,
            agents_safe_0,
            agents_safe_1,
        ],
    )

    ckpt_input.submit(
        ckpt_callback,
        inputs=ckpt_input,
        outputs=[
            ckpt_output,
            sampling_method,
            temperature,
            main_vis,
            agents_safe_0,
            agents_safe_1,
        ],
    )

    # TODO: Interesting feature to allow user click the scenario map to select a point
    # gt_vis.select(on_scenario_vis_click, outputs=[gt_vis, sel_point])
    # clear.click(on_clear_click, outputs=[gt_vis, sel_point])

    generate_button.click(
        functools.partial(on_generate_button_click, only_draw_gt=False, only_draw_detokenized=False),
        inputs=[
            sampling_method,
            temperature,
            seed,
            agents_safe_0,
            agents_safe_1,
        ],
        outputs=[
            main_vis,
            main_vis_text,
            agents_safe_0,
            agents_safe_1,
        ],
    )
    generate_backward_prediction_button.click(
        functools.partial(
            on_generate_button_click, only_draw_gt=False, only_draw_detokenized=False, draw_backward_prediction=True
        ),
        inputs=[
            sampling_method,
            temperature,
            seed,
            agents_safe_0,
            agents_safe_1,
        ],
        outputs=[
            main_vis,
            main_vis_text,
            agents_safe_0,
            agents_safe_1,
        ],
    )
    draw_gt_button.click(
        functools.partial(on_generate_button_click, only_draw_gt=True, only_draw_detokenized=False),
        inputs=[
            sampling_method,
            temperature,
            seed,
            agents_safe_0,
            agents_safe_1,
        ],
        outputs=[
            main_vis,
            main_vis_text,
            agents_safe_0,
            agents_safe_1,
        ],
    )
    draw_detok_button.click(
        functools.partial(on_generate_button_click, only_draw_gt=False, only_draw_detokenized=True),
        inputs=[
            sampling_method,
            temperature,
            seed,
            agents_safe_0,
            agents_safe_1,
        ],
        outputs=[
            main_vis,
            main_vis_text,
            agents_safe_0,
            agents_safe_1,
        ],
    )

if DEFAULT_MODEL:
    print("Loading default model from: ", DEFAULT_MODEL)
    ckpt_callback(DEFAULT_MODEL)

demo.queue().launch(server_port=args.port, share=args.share)

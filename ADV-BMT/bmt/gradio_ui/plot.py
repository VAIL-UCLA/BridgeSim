import PIL
import hydra
import matplotlib.pyplot as plt
import numpy as np
import omegaconf
import seaborn as sns
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Polygon, Circle, Rectangle

from bmt.dataset.dataset import InfgenDataset
from bmt.utils import REPO_ROOT
import torch
import pathlib

BOUNDARY = 10

EGO_FONT_SIZE = 15
MODELED_FONT_SIZE = 15
NON_EGO_FONT_SIZE = 12



def get_limit(agent_pos, map_pos):
    assert agent_pos.shape[-1] == 2
    assert map_pos.shape[-1] == 2
    agent_pos = agent_pos.reshape(-1, 2)
    map_pos = map_pos.reshape(-1, 2)
    axmin, aymin = tuple(agent_pos.min(0))
    axmax, aymax = tuple(agent_pos.max(0))
    mxmin, mymin = tuple(map_pos.min(0))
    mxmax, mymax = tuple(map_pos.max(0))
    xmin = max(axmin, mxmin) - BOUNDARY
    ymin = max(aymin, mymin) - BOUNDARY
    xmax = min(axmax, mxmax) + BOUNDARY
    ymax = min(aymax, mymax) + BOUNDARY
    return {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax}


def cal_polygon_contour(x, y, theta, width, length):
    left_front_x = x + 0.5 * length * np.cos(theta) - 0.5 * width * np.sin(theta)
    left_front_y = y + 0.5 * length * np.sin(theta) + 0.5 * width * np.cos(theta)
    left_front = np.column_stack((left_front_x, left_front_y))

    right_front_x = x + 0.5 * length * np.cos(theta) + 0.5 * width * np.sin(theta)
    right_front_y = y + 0.5 * length * np.sin(theta) - 0.5 * width * np.cos(theta)
    right_front = np.column_stack((right_front_x, right_front_y))

    right_back_x = x - 0.5 * length * np.cos(theta) + 0.5 * width * np.sin(theta)
    right_back_y = y - 0.5 * length * np.sin(theta) - 0.5 * width * np.cos(theta)
    right_back = np.column_stack((right_back_x, right_back_y))

    left_back_x = x - 0.5 * length * np.cos(theta) - 0.5 * width * np.sin(theta)
    left_back_y = y - 0.5 * length * np.sin(theta) + 0.5 * width * np.cos(theta)
    left_back = np.column_stack((left_back_x, left_back_y))

    polygon_contour = np.concatenate(
        (left_front[:, None, :], right_front[:, None, :], right_back[:, None, :], left_back[:, None, :]), axis=1
    )

    return polygon_contour


def draw_2d(pos, mask=None, **kwargs):  # for trajectory
    # pos: (-1, 2)
    # mask: (-1,
    if mask is None:
        return plt.plot(pos[..., 0], pos[..., 1], **kwargs)
    return plt.plot(pos[..., 0][mask], pos[..., 1][mask], **kwargs)


def draw_position(
    ax, pos, heading, width, length, fill_color, text=None, fontsize=20, position_kwargs=None, contour_kwargs=None
):
    position_kwargs = position_kwargs or {}
    contour_kwargs = contour_kwargs or {}

    position_kwargs["color"] = fill_color

    contour = cal_polygon_contour(
        x=np.array([pos[0]]), y=np.array([pos[1]]), theta=np.array([heading]), width=width, length=length
    )

    ax.fill(contour[0][:, 0], contour[0][:, 1], **position_kwargs)

    contour_closed = np.concatenate([contour[0], contour[0][:1]], axis=0)

    ax.plot(contour_closed[:, 0], contour_closed[:, 1], color='black', linewidth=1, **contour_kwargs)

    ax.plot(contour[0][:, 0], contour[0][:, 1], color='black', linewidth=1, **contour_kwargs)

    if text is not None:
        ax.text(pos[0], pos[1], text, color=fill_color, fontsize=fontsize)


def draw_trajectory(
    *,
    ax,
    pos,
    heading,
    width,
    length,
    fill_color,
    mask=None,
    text=None,
    fontsize=20,
    traj_kwargs=None,
    contour_kwargs=None,
    draw_line=False,
    draw_text=True
):
    traj_kwargs = traj_kwargs or {}
    contour_kwargs = contour_kwargs or {}

    traj_kwargs["color"] = fill_color

    assert heading.shape == pos.shape[:-1]
    assert isinstance(width, (float, np.floating)) or heading.shape == width.shape
    assert isinstance(length, (float, np.floating)) or heading.shape == length.shape

    non_zero_mask = (width != 0) & (length != 0)
    indices = np.where(non_zero_mask)[0]
    selected_width = width[non_zero_mask]
    selected_length = length[non_zero_mask]

    if indices.size > 0:
        selected_index = indices[0]
        selected_width = width[selected_index]
        selected_length = length[selected_index]

    else:
        print("no shape to draw")
        return
    
    contour = cal_polygon_contour(x=pos[..., 0], y=pos[..., 1], theta=heading, width=selected_width, length=selected_length)
    selected_contour = contour[::5]

    if mask is not None:
        contour = contour[mask]
        selected_contour = selected_contour[mask[::5]]

    assert (contour != 0.0).all()
    assert (selected_contour != 0.0).all()

    if len(contour) == 0:
        # Nothing to draw
        print("Agent {} has no valid data to draw".format(text))
        return

    if mask is not None and not mask[0]:
        selected_contour = np.concatenate([contour[0][None], selected_contour])
    if mask is not None and not mask[-1]:
        selected_contour = np.concatenate([selected_contour, contour[-1][None]])

    reverse = contour[::-1]
    if draw_line:
        ax.plot(contour.mean(1)[:, 0], contour.mean(1)[:, 1], **traj_kwargs)

    else:
        for i, ct in enumerate(reverse):
            # Calculate alpha based on the position in the sequence
            # for i=0, it's the last element, we want its alpha = 1
            alpha = 1.0 - min(0.9, i / len(contour))

            # Plot the contour with the calculated alpha
            ax.fill(ct[:, 0], ct[:, 1], alpha=alpha, **traj_kwargs)

    # Plot the original contours
    reverse = selected_contour[::-1]
    for poly in reverse:
        ax.fill(poly[:, 0], poly[:, 1], **contour_kwargs)

    if text is not None and draw_text:
        c = selected_contour[0]
        c_final = selected_contour[-1]
        c = (c - c_final) * 0.05 + c
        rand = np.random.randint(4)
        col = [min(v * 0.8, 1.0) for v in fill_color]
        ax.text(c[rand][0], c[rand][1], text, color=col, fontsize=fontsize)


def draw_crosswalk(ax, polygon, mask, alpha=0.1):
    polygon = polygon[mask]
    polygon = Polygon(polygon, closed=True, edgecolor='red', facecolor='yellow', alpha=alpha, linewidth=0.5)
    ax.add_patch(polygon)


def draw_stop_sign(ax, x, y, r=3):
    print("Draw stop sign triggered")
    # Number of sides for the stop sign (octagon)
    sides = 8

    # Angle between each vertex in radians
    angle = 2 * np.pi / sides

    # Calculate the vertices of the octagon
    vertices = [(x + r * np.cos((i + 0.5) * angle), y + r * np.sin((i + 0.5) * angle)) for i in range(sides)]

    # Create a polygon patch for the octagon (stop sign)
    octagon = Polygon(vertices, closed=True, edgecolor='black', facecolor='red')

    # Add the polygon patch to the axis
    ax.add_patch(octagon)


def draw_traffic_light(ax, center, fill, radius=1.5, alpha=0.4):
    circle = Circle(center, radius, edgecolor='black', facecolor=fill, alpha=alpha)
    ax.add_patch(circle)
    return circle


def _plot_map(data_dict, ax, dont_draw_lane=False):
    map_pos = data_dict["vis/map_feature"][:, :, :2]  # (num map, num vec, 2)
    map_mask = data_dict["encoder/map_feature_valid_mask"]  # (num map. num vec)
    map_feat = data_dict["vis/map_feature"]
    is_road_boundary = map_feat[..., 0, 15]
    is_lane = map_feat[..., 0, 13]
    is_crosswalk = map_feat[..., 0, 22]
    is_stop_sign = map_feat[..., 0, 24]
    for map_feat_ind in range(map_pos.shape[0]):
        if is_road_boundary[map_feat_ind]:
            draw_2d(map_pos[map_feat_ind], map_mask[map_feat_ind], c=(1.0, 0, 0, 1), linewidth=0.5)
        elif is_crosswalk[map_feat_ind]:
            draw_crosswalk(ax, polygon=map_pos[map_feat_ind], mask=map_mask[map_feat_ind])
        elif is_stop_sign[map_feat_ind]:
            draw_stop_sign(ax, map_pos[map_feat_ind][0][0], map_pos[map_feat_ind][0][1])
        elif is_lane[map_feat_ind] and (not dont_draw_lane):
            draw_2d(map_pos[map_feat_ind], map_mask[map_feat_ind], c=(0.5, 0.5, 0.5, 0.2), linewidth=0.5)
        else:
            draw_2d(map_pos[map_feat_ind], map_mask[map_feat_ind], c=(0.5, 0.5, 0.5, 0.5), label="map")


def _plot_traffic_light(data_dict, ax, step=None):
    tl_state = data_dict["encoder/traffic_light_feature"]  # T, NT, 7
    tl_pos = data_dict["encoder/traffic_light_position"][:, :2]  # NT, 3
    tl_mask = data_dict["encoder/traffic_light_valid_mask"]  # T, NT
    if tl_mask.ndim == 1:
        tl_mask = tl_mask[None]
        step = 0
    if tl_state.ndim == 2:
        tl_state = tl_state[None]
        step = 0
    if step is None:
        step = 0
    patches = []
    for tl_ind in range(tl_state.shape[1]):
        if not tl_mask[0, tl_ind]:
            continue
        tl_pos_t = tl_pos[tl_ind]
        tl_state_t = tl_state[step, tl_ind]
        if tl_state_t[3] == 1:
            color = 'green'
        elif tl_state_t[4] == 1:
            color = 'yellow'
        elif tl_state_t[5] == 1:
            color = 'red'
        elif tl_state_t[6] == 1:
            color = 'gray'
        else:
            continue
        patches.append(draw_traffic_light(ax, tl_pos_t, color))
    return patches


def _plot_gt(data_dict, ax, draw_line=False, draw_text=True, ooi=None, draw_map=True, draw_map_only=False):
    agent_pos = data_dict["decoder/agent_position"][:91, :, :2]  # (91, N, 2)
    agent_heading = data_dict["decoder/agent_heading"]  # (91, N, 2)
    agent_velocity = data_dict["decoder/agent_velocity"]  # (91, N, 2)
    agent_shape = data_dict["decoder/agent_shape"]  # (91, N, 2)
    agent_mask = data_dict["decoder/agent_valid_mask"]
    ego_agent_id = data_dict['decoder/sdc_index']

    if draw_map:
        _plot_map(data_dict, ax)

    T, N, _ = agent_pos.shape

    if not draw_map_only:

        modeled_agents_indicies = np.concatenate([data_dict["decoder/object_of_interest_id"], np.atleast_1d(ego_agent_id)])

        cmap = sns.color_palette("colorblind", n_colors=N)
        plotted_count = 0

        draw_trajectory(
            ax=ax,
            pos=agent_pos[:, ego_agent_id],
            heading=agent_heading[:, ego_agent_id],
            width=agent_shape[:, ego_agent_id, 1],
            length=agent_shape[:, ego_agent_id, 0],
            mask=agent_mask[:, ego_agent_id],
            fill_color=cmap[0],
            traj_kwargs=dict(),
            contour_kwargs=dict(
                edgecolor="k",
                linewidth=0.1,
                fill=False,
            ),
            text="{}-SDC".format(str(ego_agent_id)),
            fontsize=EGO_FONT_SIZE,
            draw_line=draw_line,
            draw_text=draw_text,
        )
        plotted_count += 1

        if not ooi:
            ooi = np.arange(N)

        for agent_ind in ooi:
            if agent_ind == ego_agent_id:
                continue
            if agent_ind in modeled_agents_indicies:
                text = "{}-OOI".format(str(agent_ind))
                fontsize = MODELED_FONT_SIZE
            else:
                text = str(agent_ind)
                fontsize = NON_EGO_FONT_SIZE
            draw_trajectory(
                ax=ax,
                pos=agent_pos[:, agent_ind],
                heading=agent_heading[:, agent_ind],
                width=agent_shape[:, agent_ind, 1],
                length=agent_shape[:, agent_ind, 0],
                mask=agent_mask[:, agent_ind],
                fill_color=cmap[plotted_count],
                traj_kwargs=dict(),
                contour_kwargs=dict(
                    edgecolor="k",
                    linewidth=0.1,
                    fill=False,
                ),
                text=text,
                fontsize=fontsize,
                draw_line=draw_line,
                draw_text=draw_text
            )
            plotted_count += 1

    if draw_map:
        _plot_traffic_light(data_dict, ax)

    return get_limit(
        agent_pos=agent_pos[agent_mask],
        map_pos=data_dict["vis/map_feature"][:, :, :2][data_dict["encoder/map_feature_valid_mask"]]
    )


def plot_gt(data_dict, get_info=False, save_path=None, ooi=None, draw_map=True):
    fig = plt.figure(figsize=(10, 10), dpi=300)
    ax = fig.add_subplot(111)
    ax.set_aspect(1)

    ret = _plot_gt(data_dict, ax, ooi=ooi, draw_map=draw_map)

    xmin, xmax, ymin, ymax = ret["xmin"], ret["xmax"], ret["ymin"], ret["ymax"]

    ax.set_xlim(xmin - BOUNDARY, xmax + BOUNDARY)
    ax.set_ylim(ymin - BOUNDARY, ymax + BOUNDARY)
    ax.set_aspect(1)
    fig.tight_layout(pad=0.05)
    fig.canvas.draw()

    if get_info:
        bbox_x0, bbox_y0, bbox_w, bbox_h = ax.get_position().bounds  # relative to figure size
        info_dict = {
            "xlim": ax.get_xlim(),
            "ylim": ax.get_ylim(),
            "fig_size": tuple(fig.get_size_inches().tolist()),
            "fig_dpi": fig.dpi,
            "bbox_x0": bbox_x0,
            "bbox_y0": bbox_y0,
            "bbox_w": bbox_w,
            "bbox_h": bbox_h,
        }
    # plt.show()

    if save_path:
        if not pathlib.Path(save_path).parent.exists():
            pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        # save figure
        fig.savefig(save_path)

    # Convert canvas to PIL Image (matplotlib 3.8+ compatible)
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    ret = PIL.Image.frombuffer('RGBA', (w, h), buf, 'raw', 'RGBA', 0, 1).convert('RGB')
    plt.close(fig)
    if get_info:
        return ret, info_dict
    return ret


def plot_pred(data_dict, show=False, save_path=None, ooi=None):

    fig = plt.figure(figsize=(10, 10), dpi=300)
    ax = fig.add_subplot(111)
    ax.set_aspect(1)

    _plot_map(data_dict, ax)

    agent_pos = data_dict["decoder/reconstructed_position"][:, :, :2]  # (91, N, 2)
    agent_heading = data_dict["decoder/reconstructed_heading"]  # (91, N, 2)
    # agent_velocity = data_dict["decoder/agent_velocity"]  # (91, N, 2)
    # agent_shape = data_dict["decoder/agent_shape"][10]  # TODO hardcoded

    agent_mask = data_dict["decoder/reconstructed_valid_mask"]
    if 'decoder/sdc_index' in data_dict:
        ego_agent_id = data_dict['decoder/sdc_index']
    else:
        ego_agent_id = 0

    T, N, _ = agent_pos.shape

    agent_shape = data_dict["decoder/current_agent_shape"]
    agent_shape = np.tile(agent_shape[None], (T, 1, 1))

    if "decoder/object_of_interest_id" in data_dict:
        modeled_agents_indicies = np.concatenate([data_dict["decoder/object_of_interest_id"], np.atleast_1d(ego_agent_id)])
    else:
        modeled_agents_indicies = []

    cmap = sns.color_palette("colorblind", n_colors=N)
    plotted_count = 0
    draw_trajectory(
        ax=ax,
        pos=agent_pos[:, ego_agent_id],
        heading=agent_heading[:, ego_agent_id],
        width=agent_shape[:, ego_agent_id, 1],
        length=agent_shape[:, ego_agent_id, 0],
        mask=agent_mask[:, ego_agent_id],
        fill_color=cmap[0],
        traj_kwargs=dict(),
        contour_kwargs=dict(
            edgecolor="k",
            linewidth=0.1,
            fill=False,
        ),
        text="{}-SDC".format(str(ego_agent_id)),
        fontsize=EGO_FONT_SIZE,
    )
    plotted_count += 1


    if not ooi:
        ooi = np.arange(N)

    for agent_ind in ooi:
        if agent_ind == ego_agent_id:
            continue
        if agent_ind in modeled_agents_indicies:
            text = "{}-OOI".format(str(agent_ind))
            fontsize = MODELED_FONT_SIZE
        else:
            text = str(agent_ind)
            fontsize = NON_EGO_FONT_SIZE
        draw_trajectory(
            ax=ax,
            pos=agent_pos[:, agent_ind],
            heading=agent_heading[:, agent_ind],
            width=agent_shape[:, agent_ind, 1],
            length=agent_shape[:, agent_ind, 0],
            mask=agent_mask[:, agent_ind],
            fill_color=cmap[plotted_count],
            traj_kwargs=dict(),
            contour_kwargs=dict(
                edgecolor="k",
                linewidth=0.1,
                fill=False,
            ),
            text=text,
            fontsize=fontsize,
        )
        plotted_count += 1

    _plot_traffic_light(data_dict, ax)

    p = agent_pos[agent_mask]
    xmax, ymax = p.max(0)
    xmin, ymin = p.min(0)
    ax.set_xlim(xmin - BOUNDARY, xmax + BOUNDARY)
    ax.set_ylim(ymin - BOUNDARY, ymax + BOUNDARY)
    ax.set_aspect(1)
    fig.tight_layout(pad=0.05)
    fig.canvas.draw()

    if show:
        plt.show()

    # Convert canvas to PIL Image (matplotlib 3.8+ compatible)
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    ret = PIL.Image.frombuffer('RGBA', (w, h), buf, 'raw', 'RGBA', 0, 1).convert('RGB')

    if not show:
        plt.close(fig)

    if save_path:
        # save figure
        if not pathlib.Path(save_path).parent.exists():
            pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
    return ret


def _animate(
    save_path, agent_pos, agent_heading, agent_mask, agent_shape, data_dict, fps=10, dpi=300, draw_traffic=True
):
    # TODO: Agent mask is not considered.

    # all_agent_pos = data_dict["decoder/agent_position"][:91, :, :2]
    # all_agent_heading = data_dict["decoder/agent_heading"]
    # all_agent_shape = data_dict["decoder/agent_shape"][10]
    if "decoder/labeled_agent_id" in data_dict:
        ooi = data_dict["decoder/labeled_agent_id"]
    else:
        ooi = []

    if 'decoder/sdc_index' in data_dict:
        ego_agent_id = int(data_dict['decoder/sdc_index'])
    else:
        ego_agent_id = 0

    assert agent_pos.ndim == 3
    T = agent_pos.shape[0]  # Number of timesteps
    N = agent_pos.shape[1]  # Number of agents

    cmap = sns.color_palette("colorblind", n_colors=N)  # Color for each agent

    all_agent_positions = agent_pos[:, :, ...].reshape(-1, 2)
    xmin, ymin = all_agent_positions.min(axis=0)
    xmax, ymax = all_agent_positions.max(axis=0)
    xlim, ylim = (xmin - 10, xmax + 10), (ymin - 10, ymax + 10)  # Adjust `BOUNDARY` as needed

    writer = FFMpegWriter(fps=fps, codec='libx264', extra_args=['-preset', 'ultrafast', '-crf', '23', '-threads', '4'])
    fig, ax = plt.subplots(figsize=(10, 10), dpi=dpi)
    ax.set_aspect(1)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    _plot_map(data_dict, ax, dont_draw_lane=True)
    _plot_traffic_light(data_dict, ax)

    agent_patches = []
    agent_texts = []
    for agent_ind in range(N):
        if not draw_traffic and agent_ind not in ooi:
            continue
        face_color = cmap[0] if agent_ind == ego_agent_id else cmap[agent_ind]
        label = "{}-SDC".format(ego_agent_id) if agent_ind == ego_agent_id else \
            "{}-OOI".format(agent_ind) if agent_ind in ooi else str(agent_ind)

        # Create a rectangular patch for each agent with black edge
        rect = Rectangle(
            (0, 0),
            agent_shape[agent_ind, 0],
            agent_shape[agent_ind, 1],
            facecolor=face_color,
            edgecolor='black',
            linewidth=0.6,
            zorder=10
        )
        agent_patches.append(rect)
        ax.add_patch(rect)

        text = ax.text(0, 0, label, color=face_color, fontsize=11, ha='center', va='center', zorder=15)
        agent_texts.append(text)

    with writer.saving(fig, save_path, dpi=dpi):
        for t in range(T):
            pos = agent_pos[t]  # update agent positions and labels for each frame
            heading = agent_heading[t]

            for agent_ind, (rect, text) in enumerate(zip(agent_patches, agent_texts)):
                x, y = pos[agent_ind]

                if not agent_mask[t, agent_ind]:
                    rect.set_visible(False)
                    text.set_visible(False)
                    x = -10000
                    y = -10000
                else:
                    rect.set_visible(True)
                    text.set_visible(True)

                rect.set_xy((x - agent_shape[agent_ind, 0] / 2, y - agent_shape[agent_ind, 1] / 2))
                rect.angle = np.degrees(heading[agent_ind])

                rect.set_edgecolor('black')
                rect.set_linewidth(0.8)

                text.set_position((x, y))
                text.set_text(text.get_text())  # forces the text to render

            writer.grab_frame()


def create_animation_from_gt(data_dict, save_path='gt_animation.mp4', fps=10, dpi=300, draw_traffic=True):
    _animate(
        save_path=save_path,
        agent_pos=data_dict["decoder/agent_position"][:91, :, :2],
        agent_mask=data_dict["decoder/agent_valid_mask"],
        agent_heading=data_dict["decoder/agent_heading"],
        agent_shape=data_dict["decoder/current_agent_shape"],
        data_dict=data_dict,
        dpi=dpi,
        draw_traffic=draw_traffic,
        fps=fps,
    )
    print(f"MP4 video saved at {save_path}")
    plt.close()
    return save_path


def create_animation_from_pred(data_dict, save_path='pred_animation.mp4', fps=10, dpi=300, draw_traffic=True):
    all_agent_shape = data_dict["decoder/current_agent_shape"]
    _animate(
        save_path=save_path,
        agent_pos=data_dict["decoder/reconstructed_position"],
        agent_mask=data_dict["decoder/reconstructed_valid_mask"],
        agent_heading=data_dict["decoder/reconstructed_heading"],
        agent_shape=all_agent_shape,
        data_dict=data_dict,
        dpi=dpi,
        draw_traffic=draw_traffic,
        fps=fps,
    )
    print(f"MP4 video saved at {save_path}")
    plt.close()
    return save_path


@hydra.main(version_base=None, config_path=str(REPO_ROOT / "cfgs"), config_name="motion_default.yaml")
def debug(config):
    omegaconf.OmegaConf.set_struct(config, False)
    config.PREPROCESSING.keep_all_data = True
    omegaconf.OmegaConf.set_struct(config, True)
    test_dataset = InfgenDataset(config, "test")
    ddd = iter(test_dataset)
    while True:
        try:
            raw_data = data = next(ddd)

            from bmt.tokenization import get_tokenizer
            tokenizer = get_tokenizer(config)
            data, _ = tokenizer.tokenize_numpy_array(data)
            data["decoder/output_action"] = data["decoder/target_action"]
            fill_zero = ~data["decoder/target_action_valid_mask"]
            data["decoder/input_action_valid_mask"][fill_zero] = False

            data = tokenizer.detokenize_numpy_array(data, detokenizing_gt=True)
            raw_data.update(data)
            # plot_pred(raw_data)
            plot_pred(raw_data, show=True)

            # break
        except StopIteration:
            break
    print("End")


@hydra.main(version_base=None, config_path=str(REPO_ROOT / "cfgs"), config_name="motion_default.yaml")
def debug_backward_prediction(config):
    omegaconf.OmegaConf.set_struct(config, False)
    config.PREPROCESSING.keep_all_data = True
    omegaconf.OmegaConf.set_struct(config, True)
    test_dataset = InfgenDataset(config, "test")
    ddd = iter(test_dataset)
    while True:
        try:
            raw_data = data = next(ddd)

            from bmt.tokenization import get_tokenizer
            tokenizer = get_tokenizer(config)
            data, _ = tokenizer.tokenize_numpy_array(data, backward_prediction=True)
            data["decoder/output_action"] = data["decoder/target_action"]
            fill_zero = ~data["decoder/target_action_valid_mask"]
            data["decoder/input_action_valid_mask"][fill_zero] = False

            data = tokenizer.detokenize_numpy_array(data, detokenizing_gt=True, backward_prediction=True)
            raw_data.update(data)
            # plot_pred(raw_data)
            plot_pred(raw_data, show=True)

            # break
        except StopIteration:
            break
    print("End")


def run_backward_prediction_with_teacher_forcing(
    model, config, backward_input_dict, tokenizer, not_teacher_forcing_ids
):
    device = backward_input_dict["decoder/agent_position"].device

    # Force to run backward prediction first to make sure the data is tokenized correctly.
    tok_data_dict, _ = tokenizer.tokenize(backward_input_dict, backward_prediction=True)
    backward_input_dict.update(tok_data_dict)

    backward_input_dict["in_evaluation"] = torch.tensor([1], dtype=bool).to(device)
    backward_input_dict["in_backward_prediction"] = torch.tensor([1], dtype=bool).to(device)
    with torch.no_grad():
        ar_func = model.model.autoregressive_rollout_backward_prediction_with_replay
        # ar_func = model.model.autoregressive_rollout_backward_prediction
        backward_output_dict = ar_func(
            backward_input_dict,
            num_decode_steps=None,
            sampling_method=config.SAMPLING.SAMPLING_METHOD,
            temperature=config.SAMPLING.TEMPERATURE,
            not_teacher_forcing_ids=not_teacher_forcing_ids,
        )
    backward_output_dict = tokenizer.detokenize(
        backward_output_dict,
        detokenizing_gt=False,
        backward_prediction=True,
        flip_wrong_heading=True,
    )
    return backward_output_dict


def run_forward_prediction_with_teacher_forcing(model, config, forward_input_dict, tokenizer, teacher_forcing_ids):
    device = forward_input_dict["decoder/agent_position"].device

    # Force to run backward prediction first to make sure the data is tokenized correctly.
    f_tok_data_dict, _ = tokenizer.tokenize(forward_input_dict, backward_prediction=False)
    forward_input_dict.update(f_tok_data_dict)

    forward_input_dict["in_evaluation"] = torch.tensor([1], dtype=bool).to(device)
    forward_input_dict["in_backward_prediction"] = torch.tensor([0], dtype=bool).to(device)
    with torch.no_grad():
        forward_output_dict = model.model.autoregressive_rollout_with_replay(
            forward_input_dict,
            sampling_method=config.SAMPLING.SAMPLING_METHOD,
            temperature=config.SAMPLING.TEMPERATURE,
            backward_prediction=False,
            teacher_forcing_ids=teacher_forcing_ids,
        )
    forward_output_dict = tokenizer.detokenize(
        forward_output_dict, detokenizing_gt=False, backward_prediction=False, flip_wrong_heading=True
    )
    return forward_output_dict


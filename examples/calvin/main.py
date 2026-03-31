import sys
from pathlib import Path

# Ensure repo root is on sys.path so "examples.calvin" works when running this file directly
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import collections
import dataclasses
import logging
import json
import pathlib
from pathlib import Path
from omegaconf import OmegaConf
import imageio
import calvin_env
import hydra
import re
import os
from examples.calvin.multistep_sequences import get_sequences, get_sequences_saved
from examples.calvin.evaluate_utils import (
    get_env_state_for_initial_condition,
    print_and_save
)
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
from tqdm import tqdm
import tyro
from typing import Optional
# import multiprocessing as multiprocessing

EP_LEN = 360

@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 9000
    resize_size: int = 224
    replan_steps: int = 10
    num_sequences: int = 5
    # action_type: str = "joint"
    # action_type: str = "eef"
    action_type: str = "eef_rel"
    video_out_path: Optional[str] = None
    eval_dir: Optional[str] = None
    eval_key: str = "A_fulltune_TEST"
    model_step: str = "last"
    seed: int = 7  # Random Seed (for reproducibility)

    model: str = "debug"
    # model: str = "evaluation"
    def __post_init__(self) -> None:
        if self.video_out_path is None:
            self.video_out_path = f"finetune_data/calvin/{self.eval_key}/{self.model_step}/action_{self.replan_steps}_seq_{self.num_sequences}/videos/"
        if self.eval_dir is None:
            self.eval_dir = f"finetune_data/calvin/{self.eval_key}/{self.model_step}/action_{self.replan_steps}_seq_{self.num_sequences}/metric/"


def get_env(dataset_path, obs_space=None, show_gui=True, **kwargs):
    from pathlib import Path

    from omegaconf import OmegaConf

    render_conf = OmegaConf.load(Path(dataset_path) / ".hydra" / "merged_config.yaml")
    if os.path.exists(Path(dataset_path) / ".hydra" / "merged_config.yaml"):
        print("**********************render_conf exists")
    if obs_space is not None:
        exclude_keys = set(render_conf.cameras.keys()) - {
            re.split("_", key)[1] for key in obs_space["rgb_obs"] + obs_space["depth_obs"]
        }
        for k in exclude_keys:
            del render_conf.cameras[k]
    if "scene" in kwargs:
        print("**********************scene exists")
        scene_cfg = OmegaConf.load(Path(calvin_env.__file__).parents[1] / "conf/scene" / f"{kwargs['scene']}.yaml")
        if os.path.exists(Path(calvin_env.__file__).parents[1] / "conf/scene" / f"{kwargs['scene']}.yaml"):
            print("**********************scene_cfg exists")
        OmegaConf.update(render_conf, "scene", scene_cfg)
    if not hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        hydra.initialize(".")
    env = hydra.utils.instantiate(render_conf.env, show_gui=show_gui, use_vr=False, use_scene_info=True)
    return env


def make_env(dataset_path, show_gui=True, split="validation", scene=None):
    val_folder = Path(dataset_path) / f"{split}"
    print("val_folder:", val_folder)
    if scene is not None:
        env = get_env(val_folder, show_gui=show_gui, scene=scene)
    else:
        env = get_env(val_folder, show_gui=show_gui)

    return env

from collections import Counter
def count_success(results):
    count = Counter(results)
    step_success = []
    for i in range(1, 6):
        n_success = sum(count[j] for j in reversed(range(i, 6)))
        sr = n_success / len(results)
        step_success.append(sr)
    return step_success

def eval_calvin(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Start evaluation
    conf_dir = Path(f"./calvin_models") / "conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)

    os.makedirs(args.eval_dir, exist_ok=True)
    eval_sr_path = os.path.join(args.eval_dir, "success_rate.txt")
    eval_result_path = os.path.join(args.eval_dir, "results.txt")

    args.calvin_dataset_path = "/media/jushen/gongda-wu/research/datasets/Calvin/task_D_D"
    env = make_env(args.calvin_dataset_path, show_gui=False)
    eval_sequences = get_sequences_saved(args.num_sequences, num_workers=1, filename=f"finetune_data/calvin/sequences/eval_episode_{args.num_sequences}.json")
    # eval_sequences = get_sequences(args.num_sequences, num_workers=1)

    val_annotations = OmegaConf.load(
        conf_dir / "annotations/new_playtable_validation.yaml"
    )
    # print(val_annotations)
    results = []
    eval_sequences = tqdm(eval_sequences, position=0, leave=True)

    for seq_ind, (initial_state, eval_sequence) in enumerate(eval_sequences):

        robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
        env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
        success_counter, video_aggregators = 0, []
        count = 0
        for subtask in eval_sequence:
            task_flag = False
            lang_annotation = val_annotations[subtask][0]
            action_plan = collections.deque()
            replay_images = []
            obs = env.get_obs()
            start_info = env.get_info()
            # print('------------------------------')
            # print(f'task: {lang_annotation}')
            # pbar = tqdm(range(EP_LEN))
            # for step in pbar:
            last_gripper_cmd = 1 
            for step in range(EP_LEN):
                img = obs["rgb_obs"]["rgb_static"]
                wrist_img = obs["rgb_obs"]["rgb_gripper"]
                img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                )
                wrist_img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                )
                replay_images.append(obs["rgb_obs"]["rgb_static"])

                if not action_plan:
                    # Finished executing previous action chunk -- compute new chunk
                    # Prepare observations dict
                    # if args.model == "debug":
                    #     print("model inference")

                    if "eef" in args.action_type:
                        obs_state = np.concatenate([obs['robot_obs'][:6], obs['robot_obs'][-1:]]).astype(np.float32)
                    elif "joint" in args.action_type:
                        obs_state = obs['robot_obs'][-8:]
                    else:
                        raise NotImplementedError(f"Unknown action type: {args.action_type}")

                    element = {
                        "observation/image": img,
                        "observation/wrist_image": wrist_img,
                        "observation/state": obs_state,
                        "prompt": str(lang_annotation),
                    }
                    action_chunk = client.infer(element)["actions"]
                    assert (
                        len(action_chunk) >= args.replan_steps
                    ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                    action_plan.extend(action_chunk[: args.replan_steps])

                
                action = action_plan.popleft()
                action = action.copy()
                # if action[-1] < -0.2:
                #     action[-1] = -1
                # elif action[-1] > 0.2:
                #     action[-1] = 1
                # else:
                #     action[-1] = last_gripper_cmd
                # last_gripper_cmd = action[-1]
                if action[-1] < 0:
                    action[-1] = -1
                else:
                    action[-1] = 1
                if args.action_type == "eef":
                    assert action.shape[0] == 7
                    curr_action = [action[:3], action[3:6], action[[6]]]
                    action_dict = {
                        "type": "cartesian_abs",
                        "action": curr_action  # 原来的 action 值
                    }
                elif args.action_type == "joint":
                    assert action.shape[0] == 8
                    action_dict = {
                        "type": "joint_abs",
                        "action": action  # 原来的 action 值
                    }
                elif args.action_type == "eef_rel":
                    assert action.shape[0] == 7
                    action_dict = {
                        "type": "cartesian_rel",
                        "action": action 
                    }
                    # action_dict = action
                else:
                    raise NotImplementedError(f"Unknown action type: {args.action_type}")
                
                obs, _, _, current_info = env.step(action_dict)
                # if args.model == "debug":
                #     import cv2
                #     cv2.imwrite(f"visualization/obs_{step}.png", obs["rgb_obs"]["rgb_static"])

                # check if current step solves a task
                current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
                if len(current_task_info) > 0:
                    # print(f"task {subtask} solved at step {step}")
                    task_flag = True
                    break
            # only save video for failed tasks
            if not task_flag:
                if args.model=="debug":
                    task_segment = lang_annotation.replace(" ", "_")
                    imageio.mimwrite(
                        pathlib.Path(args.video_out_path) / f"{seq_ind}_{count}_rollout_{task_segment}_{task_flag}.mp4",
                        [np.asarray(x) for x in replay_images],
                        fps=30,
                        macro_block_size=1
                    )
                    count += 1
                else:
                    pass
            if task_flag:
                if args.model=="debug":
                    # save video for successful tasks
                    task_segment = lang_annotation.replace(" ", "_")
                    imageio.mimwrite(
                        pathlib.Path(args.video_out_path) / f"{seq_ind}_{count}_rollout_{task_segment}_{task_flag}.mp4",
                        [np.asarray(x) for x in replay_images],
                        fps=30,
                        macro_block_size=1
                    )
                    count += 1
                success_counter += 1
            else:
                # print(f"task {subtask} not solved")
                break


        results.append(success_counter)
        success_list = count_success(results)

        with open(eval_sr_path, 'a') as f:
            line =f"{seq_ind}/{args.num_sequences}: "
            for sr in success_list:
                line += f"{sr:.3f} | "
            line += "\n"
            f.write(line)
        eval_sequences.set_description(
            " ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(success_list)]) + "|"
        )

    print_and_save(results, eval_sequences, eval_result_path, None)

if __name__ == "__main__":
    import multiprocessing
    if multiprocessing.get_start_method(allow_none=True) != "spawn":  
        multiprocessing.set_start_method("spawn", force=True)
    logging.basicConfig(level=logging.INFO)
    # args = 
    eval_calvin(tyro.cli(Args))

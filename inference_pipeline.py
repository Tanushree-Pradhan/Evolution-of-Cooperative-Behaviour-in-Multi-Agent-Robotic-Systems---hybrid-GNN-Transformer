"""
Inference pipeline for SwarmGNN-Former.
Self contained: config, logging, model load, and execution-only simulation.
"""
import os
import math
import random
from typing import List, Tuple
from collections import deque

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
from datetime import datetime

import pygame
from pygame import Color
from Box2D import (
    b2World, b2BodyDef, b2FixtureDef, b2CircleShape, b2PolygonShape,
    b2ContactListener, b2_staticBody, b2_dynamicBody
)

# ---------------------------
# Logging (self-contained)
# ---------------------------
import logging
import sys

def setup_logger(name: str, level: int = logging.INFO, logfile: str = "results/inference.log") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if logger.handlers:
        return logger
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(ch)
    try:
        fh = logging.FileHandler(logfile, mode='a')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(fh)
    except Exception as e:
        logger.warning(f"Could not attach file handler: {e}")
    return logger

logger = setup_logger(__name__)

# ---------------------------
# Config (inference copy)
# ---------------------------
WIDTH = 800
HEIGHT = 600
PPM = 100.0
TIME_STEP = 1.0 / 60.0
VEL_ITERS = 10
POS_ITERS = 10

def px_to_m(px):
    return px / PPM

def m_to_px(m):
    return m * PPM

MAX_DIMENSION_M = max(px_to_m(WIDTH), px_to_m(HEIGHT))
MAX_DISTANCE_M = (px_to_m(WIDTH) ** 2 + px_to_m(HEIGHT) ** 2) ** 0.5

class RobotConfig:
    def __init__(self):
        self.number = 5
        self.size_px = 15
        self.density = 3.0
        self.max_speed = 3.5
        self.linear_damping = 1.5
        self.angular_damping = 3.0
        self.detect_radius_m = 3.0
        self.comms_range_m = 5.0
        self.avoidance_radius_m = 0.4
        self.avoidance_weight = 0.2

class ObjectConfig:
    def __init__(self):
        self.size_px = 50
        self.density = 150.0

class PushConfig:
    def __init__(self):
        self.target_offset = 0.1
        self.distance_threshold = 0.2
        self.speed = 3.5
        self.arc_degrees = 80

class WallConfig:
    def __init__(self):
        self.avoid_distance_m = 0.5
        self.avoid_force = 1.0

class NNConfig:
    def __init__(self):
        self.input_size = 6
        self.hidden_size = 12
        self.output_size = 2
        self.history_len = 8
        self.k_neighbors = 4

class VizConfig:
    def __init__(self):
        self.goal_size_px = 30
        self.goal_x_px = 650
        self.goal_y_px = 300
        self.path_color = (128, 128, 128, 200)

class InferenceConfig:
    def __init__(self):
        self.num_trials = 5  # Number of inference runs to perform
        self.max_steps = 1000  # Maximum steps per trial
        self.visualize_trials = True  # Show visualization for each trial

robot_cfg = RobotConfig()
inference_cfg = InferenceConfig()
object_cfg = ObjectConfig()
push_cfg = PushConfig()
wall_cfg = WallConfig()
nn_cfg = NNConfig()
viz_cfg = VizConfig()

MODE = 'EXECUTE'
RESULTS_DIR = "results"
MODEL_PT = os.path.join(RESULTS_DIR, "trained_model.pt")

NN_INPUT_SIZE = nn_cfg.input_size
NN_HIDDEN = nn_cfg.hidden_size
NN_OUTPUT_SIZE = nn_cfg.output_size
SEQ_LEN = nn_cfg.history_len

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

K_NEIGH = nn_cfg.k_neighbors
NEIGH_FEAT_DIM = 3

ROBOT_NUMBER = robot_cfg.number
ROBOT_SIZE = robot_cfg.size_px
ROBOT_DENSITY = robot_cfg.density
ROBOT_MAX_SPEED = robot_cfg.max_speed
ROBOT_LINEAR_DAMPING = robot_cfg.linear_damping
ROBOT_ANGULAR_DAMPING = robot_cfg.angular_damping
ROBOT_DETECT_RADIUS = robot_cfg.detect_radius_m
ROBOT_COMMS_RANGE = robot_cfg.comms_range_m

AVOIDANCE_RADIUS = robot_cfg.avoidance_radius_m
AVOIDANCE_WEIGHT = robot_cfg.avoidance_weight
TARGET_PUSH_OFFSET = push_cfg.target_offset
PUSH_DIST_THRESHOLD = push_cfg.distance_threshold
PUSH_SPEED = push_cfg.speed
PUSH_ARC_DEGREES = push_cfg.arc_degrees

OBJECT_SIZE = object_cfg.size_px
OBJECT_DENSITY = object_cfg.density
GOAL_SIZE = viz_cfg.goal_size_px

WALL_AVOID_DIST_M = wall_cfg.avoid_distance_m
WALL_AVOID_FORCE = wall_cfg.avoid_force

FIXED_GOAL_X_PX = viz_cfg.goal_x_px
FIXED_GOAL_Y_PX = viz_cfg.goal_y_px
PATH_COLOR = Color(*viz_cfg.path_color)

# ---------------------------
# Converters
# ---------------------------
def vec_m_to_px(pos) -> Tuple[int, int]:
    return int(m_to_px(pos[0])), int(m_to_px(pos[1]))

# ---------------------------
# Physics Entities
# ---------------------------
class GoalContactListener(b2ContactListener):
    def __init__(self):
        super().__init__()
        self.hit = False
    def BeginContact(self, contact):
        a = contact.fixtureA.body.userData
        b = contact.fixtureB.body.userData
        if (isinstance(a, Goal) and isinstance(b, PushableObj)) or (isinstance(b, Goal) and isinstance(a, PushableObj)):
            self.hit = True

class Wall:
    def __init__(self, world, x_px, y_px, w_px, h_px):
        self.w_px, self.h_px = w_px, h_px
        bd = b2BodyDef(position=(px_to_m(x_px), px_to_m(y_px)), type=b2_staticBody)
        body = world.CreateBody(bd)
        box = b2PolygonShape(box=(px_to_m(w_px / 2.0), px_to_m(h_px / 2.0)))
        body.CreateFixture(shape=box, density=1.0)
        body.userData = self
        self.body = body
    def draw(self, surf):
        rect = pygame.Rect(int(m_to_px(self.body.position[0]) - self.w_px / 2), int(m_to_px(self.body.position[1]) - self.h_px / 2), int(self.w_px), int(self.h_px))
        pygame.draw.rect(surf, Color("black"), rect)

class Goal:
    def __init__(self, world, x_px, y_px, size_px):
        self.size_px = size_px
        bd = b2BodyDef(position=(px_to_m(FIXED_GOAL_X_PX), px_to_m(FIXED_GOAL_Y_PX)), type=b2_staticBody)
        body = world.CreateBody(bd)
        shape = b2CircleShape(radius=px_to_m(size_px))
        body.CreateFixture(shape=shape, friction=0.01)
        body.userData = self
        self.body = body
    def draw(self, surf):
        p = vec_m_to_px(self.body.worldCenter)
        pygame.draw.circle(surf, (119, 136, 153), p, int(self.size_px), width=0)
        pygame.draw.circle(surf, Color("black"), p, int(self.size_px), width=1)

class PushableObj:
    def __init__(self, world, x_px, y_px, size_px):
        self.size_px = size_px
        bd = b2BodyDef(position=(px_to_m(x_px), px_to_m(y_px)), type=b2_dynamicBody, linearDamping=0.2, angularDamping=0.5)
        body = world.CreateBody(bd)
        shape = b2PolygonShape(box=(px_to_m(size_px), px_to_m(size_px)))
        fd = b2FixtureDef(shape=shape, density=OBJECT_DENSITY, friction=0.5, restitution=0.05)
        body.CreateFixture(fd)
        body.userData = self
        self.body = body
        initial_pos = (body.position[0], body.position[1])
        self.path_points: List[Tuple[float, float]] = [initial_pos]
        self.path_length_m: float = 0.0
        self.last_pos_m: Tuple[float, float] = initial_pos
    def update_path(self):
        current_pos_m = (self.body.position[0], self.body.position[1])
        dx = current_pos_m[0] - self.last_pos_m[0]
        dy = current_pos_m[1] - self.last_pos_m[1]
        distance_moved = math.hypot(dx, dy)
        if distance_moved > 0.01:
            self.path_points.append(current_pos_m)
            self.path_length_m += distance_moved
            self.last_pos_m = current_pos_m
    def reset_path(self):
        current_pos = (self.body.position[0], self.body.position[1])
        self.path_points = [current_pos]
        self.path_length_m = 0.0
        self.last_pos_m = current_pos
    def draw(self, surf, robots_pushing=None):
        px_points = [vec_m_to_px(p) for p in self.path_points]
        if len(px_points) > 1:
            pygame.draw.aalines(surf, PATH_COLOR, False, px_points, 1)
        if robots_pushing:
            crate_center_px = vec_m_to_px(self.body.worldCenter)
            for robot in robots_pushing:
                robot_center_px = vec_m_to_px(robot.body.worldCenter)
                pygame.draw.line(surf, Color(255, 165, 0), crate_center_px, robot_center_px, 2)
        p_m = self.body.worldCenter
        angle = self.body.angle
        half_width_m = px_to_m(self.size_px)
        unrotated_corners = [
            pygame.Vector2(-half_width_m, -half_width_m),
            pygame.Vector2( half_width_m, -half_width_m),
            pygame.Vector2( half_width_m,  half_width_m),
            pygame.Vector2(-half_width_m,  half_width_m)
        ]
        pygame_points = []
        rotation_degrees = math.degrees(angle)
        for corner_vec in unrotated_corners:
            rotated_corner = corner_vec.rotate(rotation_degrees)
            px_x = m_to_px(p_m[0] + rotated_corner.x)
            px_y = m_to_px(p_m[1] + rotated_corner.y)
            pygame_points.append((int(px_x), int(px_y)))
        pygame.draw.polygon(surf, Color("brown"), pygame_points, 0)
        pygame.draw.polygon(surf, Color("black"), pygame_points, 1)

# ---------------------------
# Model
# ---------------------------
class SwarmGNNFormer(nn.Module):
    def __init__(self, input_dim=NN_INPUT_SIZE, embed_dim=NN_HIDDEN, num_heads=4, ff_dim=None, num_layers=2, k_neighbors=K_NEIGH, max_seq=64):
        super().__init__()
        ff_dim = ff_dim or (embed_dim * 4)
        self.k = k_neighbors
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim, batch_first=True, norm_first=True, activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.neigh_mlp = nn.Sequential(
            nn.Linear(NEIGH_FEAT_DIM, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim),
            nn.GELU()
        )
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, NN_OUTPUT_SIZE),
            nn.Tanh()
        )

    def forward(self, seq_x, neigh_x):
        B = seq_x.size(0)
        seq_len = seq_x.size(1)
        x = self.input_proj(seq_x)
        x = x + self.pos_embed[:, :seq_len, :]
        x = self.transformer(x)
        last = x[:, -1, :]
        B, K, D = neigh_x.shape
        neigh_flat = neigh_x.view(B * K, D)
        neigh_emb = self.neigh_mlp(neigh_flat)
        neigh_emb = neigh_emb.view(B, K, -1)
        neigh_agg = neigh_emb.mean(dim=1)
        fused = torch.cat([last, neigh_agg], dim=1)
        out = self.fusion(fused)
        return out

def load_model_from_pt(filename_pt=MODEL_PT, device=DEVICE):
    if not os.path.exists(filename_pt):
        logger.warning(f"Model file {filename_pt} not found.")
        return None
    ckpt = torch.load(filename_pt, map_location=device)
    model = SwarmGNNFormer().to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    logger.info(f"Loaded model from {filename_pt}")
    return model

# ---------------------------
# Helpers
# ---------------------------
def pad_or_cut_neighbors(neigh_list, k=K_NEIGH):
    arr = np.zeros((k, NEIGH_FEAT_DIM), dtype=np.float32)
    for i, v in enumerate(neigh_list[:k]):
        arr[i, :] = np.array(v, dtype=np.float32)
    return arr

# ---------------------------
# SwarmRobot (execution)
# ---------------------------
class SwarmRobot:
    def __init__(self, world: b2World, x_px: float, y_px: float, robot_id: int):
        self.size_px = ROBOT_SIZE
        self.robot_id = robot_id
        bd = b2BodyDef(position=(px_to_m(x_px), px_to_m(y_px)), type=b2_dynamicBody, linearDamping=ROBOT_LINEAR_DAMPING, angularDamping=ROBOT_ANGULAR_DAMPING)
        body = world.CreateBody(bd)
        shape = b2CircleShape(radius=px_to_m(self.size_px))
        body.CreateFixture(shape=shape, density=ROBOT_DENSITY, friction=0.6, restitution=0.05)
        body.userData = self
        self.body = body
        self.state = 'search'
        self.random_dir = pygame.Vector2(random.uniform(-1,1), random.uniform(-1,1)).normalize()
        self.target_position = None
        self.history = deque(maxlen=SEQ_LEN)
        for _ in range(SEQ_LEN):
            self.history.append([0.0] * NN_INPUT_SIZE)
        self.model = None

    def set_model(self, model: nn.Module):
        self.model = model

    def random_walk(self):
        pos = self.body.position
        if random.random() < 0.01:
            self.random_dir = pygame.Vector2(random.uniform(-1,1), random.uniform(-1,1)).normalize()
        if pos[0] < WALL_AVOID_DIST_M or pos[0] > px_to_m(WIDTH) - WALL_AVOID_DIST_M:
            self.random_dir.x *= -1
        if pos[1] < WALL_AVOID_DIST_M or pos[1] > px_to_m(HEIGHT) - WALL_AVOID_DIST_M:
            self.random_dir.y *= -1
        self.body.linearVelocity = (self.random_dir.x * ROBOT_MAX_SPEED, self.random_dir.y * ROBOT_MAX_SPEED)

    def rule_based_control(self, desired_dir: pygame.Vector2, all_robots: List['SwarmRobot']) -> pygame.Vector2:
        avoid_vec = pygame.Vector2(0,0)
        robot_pos = self.body.position
        for other in all_robots:
            if other is self:
                continue
            other_pos = other.body.position
            dist = math.hypot(robot_pos[0] - other_pos[0], robot_pos[1] - other_pos[1])
            if dist < AVOIDANCE_RADIUS and dist > 0:
                away_vec = pygame.Vector2(robot_pos[0] - other_pos[0], robot_pos[1] - other_pos[1])
                avoid_vec += away_vec.normalize() / (dist**2)
        if avoid_vec.length() > 0:
            avoid_vec.normalize_ip()
            final_dir = desired_dir.normalize() * (1 - AVOIDANCE_WEIGHT) + avoid_vec * AVOIDANCE_WEIGHT
            return final_dir.normalize() if final_dir.length() > 0 else desired_dir.normalize()
        else:
            return desired_dir.normalize() if desired_dir.length() > 0 else desired_dir

    def signal_crate_location(self, all_robots: List['SwarmRobot'], crate_pos: Tuple[float,float]):
        robot_pos = self.body.position
        for other in all_robots:
            if other is self:
                continue
            other_pos = other.body.position
            dist = math.hypot(robot_pos[0] - other_pos[0], robot_pos[1] - other_pos[1])
            if dist < ROBOT_COMMS_RANGE and other.state == 'search':
                other.target_position = crate_pos
                other.state = 'go_to_crate'
            elif dist < ROBOT_COMMS_RANGE and other.state == 'go_to_crate':
                other.target_position = crate_pos

    def apply_wall_repulsion(self, current_dir: pygame.Vector2) -> pygame.Vector2:
        pos = self.body.position
        repulsion = pygame.Vector2(0, 0)
        if pos[0] < WALL_AVOID_DIST_M:
            repulsion.x = WALL_AVOID_FORCE * (1.0 - pos[0] / WALL_AVOID_DIST_M)
        elif pos[0] > px_to_m(WIDTH) - WALL_AVOID_DIST_M:
            repulsion.x = -WALL_AVOID_FORCE * (1.0 - (px_to_m(WIDTH) - pos[0]) / WALL_AVOID_DIST_M)
        if pos[1] < WALL_AVOID_DIST_M:
            repulsion.y = WALL_AVOID_FORCE * (1.0 - pos[1] / WALL_AVOID_DIST_M)
        elif pos[1] > px_to_m(HEIGHT) - WALL_AVOID_DIST_M:
            repulsion.y = -WALL_AVOID_FORCE * (1.0 - (px_to_m(HEIGHT) - pos[1]) / WALL_AVOID_DIST_M)
        if repulsion.length() > 0:
            repulsion.normalize_ip()
            final_dir = current_dir * (1.0 - 0.2) + repulsion * 0.2
            return final_dir.normalize() if final_dir.length() > 0 else current_dir.normalize()
        else:
            return current_dir

    def compute_neighbors(self, all_robots: List['SwarmRobot'], K=K_NEIGH):
        me = self.body.position
        neigh = []
        for other in all_robots:
            if other is self:
                continue
            op = other.body.position
            dx = op[0] - me[0]
            dy = op[1] - me[1]
            dist = math.hypot(dx, dy)
            rel_speed = math.hypot(other.body.linearVelocity[0], other.body.linearVelocity[1])
            neigh.append((dist, dx, dy, rel_speed))
        neigh.sort(key=lambda x: x[0])
        feat_list = []
        for item in neigh[:K]:
            _, dx, dy, rs = item
            feat_list.append([dx, dy, rs])
        return pad_or_cut_neighbors(feat_list, k=K)

    def update_behavior(self, crate, goal, all_robots):
        crate_pos = crate.body.position
        robot_pos = self.body.position
        goal_pos = goal.body.position
        dist_to_crate = math.hypot(crate_pos[0] - robot_pos[0], crate_pos[1] - robot_pos[1])
        final_vx, final_vy = 0.0, 0.0

        if self.state == 'search':
            self.random_walk()
            if dist_to_crate < ROBOT_DETECT_RADIUS:
                self.target_position = crate_pos
                self.state = 'push'
                self.signal_crate_location(all_robots, crate_pos)
        elif self.state in ['go_to_crate', 'push']:
            if self.state == 'go_to_crate':
                current_target_pos = self.target_position
                if dist_to_crate < ROBOT_DETECT_RADIUS:
                    self.state = 'push'
                    return
            else:
                goal_vector = pygame.Vector2(goal_pos[0] - crate_pos[0], goal_pos[1] - crate_pos[1])
                if goal_vector.length() > 0:
                    goal_vector.normalize_ip()
                angle_separation = PUSH_ARC_DEGREES / (ROBOT_NUMBER - 1 if ROBOT_NUMBER > 1 else 1)
                angle_offset = (angle_separation * self.robot_id) - (PUSH_ARC_DEGREES / 2)
                push_angle_rad = math.atan2(goal_vector.y, goal_vector.x) + math.pi + math.radians(angle_offset)
                pushing_distance_m = px_to_m(crate.size_px) + px_to_m(self.size_px) + TARGET_PUSH_OFFSET
                push_spot_x = crate_pos[0] + math.cos(push_angle_rad) * pushing_distance_m
                push_spot_y = crate_pos[1] + math.sin(push_angle_rad) * pushing_distance_m
                current_target_pos = (push_spot_x, push_spot_y)

            dist_to_target = math.hypot(current_target_pos[0] - robot_pos[0], current_target_pos[1] - robot_pos[1])
            is_actively_pushing = (self.state == 'push' and dist_to_target <= PUSH_DIST_THRESHOLD)
            if is_actively_pushing:
                input_vec = pygame.Vector2(goal_pos[0] - crate_pos[0], goal_pos[1] - crate_pos[1])
                target_dir = input_vec.normalize() if input_vec.length() > 0 else pygame.Vector2(0,0)
            else:
                input_vec = pygame.Vector2(current_target_pos[0] - robot_pos[0], current_target_pos[1] - robot_pos[1])
                target_dir = input_vec.normalize() if input_vec.length() > 0 else pygame.Vector2(0,0)

            current_speed = math.hypot(self.body.linearVelocity[0], self.body.linearVelocity[1])
            norm_target_x = current_target_pos[0] / MAX_DIMENSION_M
            norm_target_y = current_target_pos[1] / MAX_DIMENSION_M
            feature = [target_dir.x, target_dir.y, dist_to_target / MAX_DISTANCE_M, current_speed / ROBOT_MAX_SPEED, norm_target_x, norm_target_y]
            self.history.append(feature)

            if self.model is None:
                final_dir = self.rule_based_control(target_dir, all_robots)
                final_dir = self.apply_wall_repulsion(final_dir)
                if is_actively_pushing:
                    final_vx = final_dir.x * PUSH_SPEED * 1.5
                    final_vy = final_dir.y * PUSH_SPEED * 1.5
                else:
                    final_vx = final_dir.x * ROBOT_MAX_SPEED
                    final_vy = final_dir.y * ROBOT_MAX_SPEED
                self.body.linearVelocity = (final_vx, final_vy)
                return

            hist_list = list(self.history)
            if len(hist_list) < SEQ_LEN:
                pad = [[0.0]*NN_INPUT_SIZE] * (SEQ_LEN - len(hist_list))
                seq = pad + hist_list
            else:
                seq = hist_list[-SEQ_LEN:]
            neigh = self.compute_neighbors(all_robots, K=K_NEIGH)

            x = torch.tensor([seq], dtype=torch.float32, device=DEVICE)
            n = torch.tensor([neigh], dtype=torch.float32, device=DEVICE)
            with torch.no_grad():
                out = self.model(x, n).cpu().numpy()[0]
            nn_dir = pygame.Vector2(float(out[0]), float(out[1]))
            final_dir = self.rule_based_control(nn_dir.normalize() if nn_dir.length() > 0 else nn_dir, all_robots)
            final_dir = self.apply_wall_repulsion(final_dir)
            if is_actively_pushing:
                final_dir = self.rule_based_control(target_dir, all_robots)
                final_dir = self.apply_wall_repulsion(final_dir)
                final_vx = final_dir.x * PUSH_SPEED * 1.5
                final_vy = final_dir.y * PUSH_SPEED * 1.5
            else:
                speed_mult = min(1.0, nn_dir.length() + 0.5)
                final_vx = final_dir.x * ROBOT_MAX_SPEED * speed_mult
                final_vy = final_dir.y * ROBOT_MAX_SPEED * speed_mult
            self.body.linearVelocity = (final_vx, final_vy)
        else:
            self.random_walk()

    def draw(self, surf):
        p = vec_m_to_px(self.body.worldCenter)
        color = Color("red") if self.state == 'push' else Color("blue") if self.state == 'go_to_crate' else Color("green")
        pygame.draw.circle(surf, color, p, int(self.size_px), width=0)
        pygame.draw.circle(surf, Color("black"), p, int(self.size_px), width=1)

# ---------------------------
# Environment helpers
# ---------------------------
def reset_positions(robots: List[SwarmRobot], crate: PushableObj, goal: Goal) -> bool:
    safe_margin_m = px_to_m(OBJECT_SIZE * 2)
    new_crate_x = random.uniform(safe_margin_m * 2, px_to_m(WIDTH) / 2 - safe_margin_m)
    new_crate_y = random.uniform(safe_margin_m, px_to_m(HEIGHT) - safe_margin_m)
    crate.body.position = (new_crate_x, new_crate_y)
    crate.body.linearVelocity = (0, 0)
    crate.body.angularVelocity = 0
    crate.body.angle = random.uniform(0, math.pi * 2)
    crate.body.awake = True
    crate.reset_path()

    robot_margin_m = px_to_m(ROBOT_SIZE * 3)
    start_x_m = new_crate_x - px_to_m(100)
    for i, robot in enumerate(robots):
        rand_offset_x = random.uniform(-robot_margin_m, robot_margin_m)
        rand_offset_y = random.uniform(-robot_margin_m, robot_margin_m)
        new_x = start_x_m + rand_offset_x
        new_y = new_crate_y + rand_offset_y + (i * px_to_m(1))
        new_x = max(px_to_m(ROBOT_SIZE), min(new_x, px_to_m(WIDTH) - px_to_m(ROBOT_SIZE)))
        robot.body.position = (new_x, new_y)
        robot.body.linearVelocity = (0, 0)
        robot.body.angularVelocity = 0
        robot.state = 'search'
        robot.target_position = None
        robot.random_dir = pygame.Vector2(random.uniform(-1,1), random.uniform(-1,1)).normalize()
        robot.history = deque([[0.0]*NN_INPUT_SIZE for _ in range(SEQ_LEN)], maxlen=SEQ_LEN)
    return False

# ---------------------------
# Metrics Tracker for Inference
# ---------------------------
class InferenceMetrics:
    def __init__(self, trial_num=1):
        self.trial_num = trial_num
        self.step_history = []
        self.collaboration_history = []  # (searching, going, pushing) per step
        self.robot_states_history = []  # Track state changes over time
        self.path_positions = []  # Crate positions over time
        self.total_steps = 0
        self.final_path_length = 0.0
        self.start_time = None
        self.end_time = None
        self.success = False
        
    def record_step(self, step, robots, crate_pos):
        """Record metrics for current step"""
        self.step_history.append(step)
        
        # Count robot states
        searching = sum(1 for r in robots if r.state == 'search')
        going = sum(1 for r in robots if r.state == 'go_to_crate')
        pushing = sum(1 for r in robots if r.state == 'push')
        self.collaboration_history.append((searching, going, pushing))
        
        # Record crate position
        self.path_positions.append((crate_pos[0], crate_pos[1]))
    
    def finalize(self, total_steps, path_length, success=True):
        """Finalize metrics after completion"""
        self.total_steps = total_steps
        self.final_path_length = path_length
        self.success = success
        self.end_time = datetime.now()

# ---------------------------
# Visualization Functions
# ---------------------------
def plot_inference_metrics(metrics, save_dir=RESULTS_DIR):
    """Generate execution performance plots"""
    logger.info("Generating inference performance plots...")
    
    if len(metrics.step_history) == 0:
        logger.warning("No metrics to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Inference Execution Metrics', fontsize=16, fontweight='bold')
    
    # Plot 1: Robot state distribution over time
    ax = axes[0, 0]
    steps = metrics.step_history
    searching = [c[0] for c in metrics.collaboration_history]
    going = [c[1] for c in metrics.collaboration_history]
    pushing = [c[2] for c in metrics.collaboration_history]
    
    ax.plot(steps, searching, label='Searching', color='#06A77D', linewidth=2, alpha=0.8)
    ax.plot(steps, going, label='Going to Crate', color='#2E86AB', linewidth=2, alpha=0.8)
    ax.plot(steps, pushing, label='Pushing', color='#D62828', linewidth=2, alpha=0.8)
    ax.set_xlabel('Simulation Step', fontsize=12)
    ax.set_ylabel('Number of Robots', fontsize=12)
    ax.set_title('Robot State Distribution Over Time', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Collaboration efficiency
    ax = axes[0, 1]
    total_robots = ROBOT_NUMBER
    efficiency = [(g + p) / total_robots * 100 for s, g, p in metrics.collaboration_history]
    ax.plot(steps, efficiency, linewidth=2, color='#F18F01')
    ax.axhline(np.mean(efficiency), color='red', linestyle='--', 
               label=f'Mean: {np.mean(efficiency):.1f}%', linewidth=2)
    ax.set_xlabel('Simulation Step', fontsize=12)
    ax.set_ylabel('Collaboration Efficiency (%)', fontsize=12)
    ax.set_title('Team Collaboration Efficiency', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])
    
    # Plot 3: Crate trajectory
    ax = axes[1, 0]
    if len(metrics.path_positions) > 0:
        x_pos = [p[0] for p in metrics.path_positions]
        y_pos = [p[1] for p in metrics.path_positions]
        
        # Plot trajectory with color gradient
        points = np.array([x_pos, y_pos]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        from matplotlib.collections import LineCollection
        lc = LineCollection(segments, cmap='viridis', linewidths=2)
        lc.set_array(np.linspace(0, 1, len(segments)))
        ax.add_collection(lc)
        
        # Mark start and end
        ax.plot(x_pos[0], y_pos[0], 'go', markersize=12, label='Start', markeredgecolor='black', markeredgewidth=2)
        ax.plot(x_pos[-1], y_pos[-1], 'r*', markersize=15, label='Goal', markeredgecolor='black', markeredgewidth=2)
        
        ax.set_xlabel('X Position (m)', fontsize=12)
        ax.set_ylabel('Y Position (m)', fontsize=12)
        ax.set_title('Crate Trajectory', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    # Plot 4: Performance summary
    ax = axes[1, 1]
    duration = (metrics.end_time - metrics.start_time).total_seconds() if metrics.end_time else 0
    
    summary_text = f"Total Steps: {metrics.total_steps:,}\\n"
    summary_text += f"Final Path Length: {metrics.final_path_length:.2f} m\\n"
    summary_text += f"Execution Time: {duration:.1f} seconds\\n\\n"
    
    if metrics.collaboration_history:
        avg_searching = np.mean([c[0] for c in metrics.collaboration_history])
        avg_going = np.mean([c[1] for c in metrics.collaboration_history])
        avg_pushing = np.mean([c[2] for c in metrics.collaboration_history])
        avg_efficiency = np.mean(efficiency)
        
        summary_text += f"Avg. Collaboration:\\n"
        summary_text += f"  Searching: {avg_searching:.1f}\\n"
        summary_text += f"  Going: {avg_going:.1f}\\n"
        summary_text += f"  Pushing: {avg_pushing:.1f}\\n"
        summary_text += f"  Efficiency: {avg_efficiency:.1f}%"
    
    ax.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.6))
    ax.axis('off')
    ax.set_title('Execution Summary', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'inference_execution_metrics.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved inference metrics to {save_path}")

def save_inference_summary(metrics, save_dir=RESULTS_DIR):
    """Save inference execution summary"""
    summary_path = os.path.join(save_dir, 'inference_summary.txt')
    
    duration = (metrics.end_time - metrics.start_time).total_seconds() if metrics.end_time else 0
    
    with open(summary_path, 'w') as f:
        f.write("="*70 + "\\n")
        f.write("SwarmGNN-Former Inference Execution Summary\\n")
        f.write("="*70 + "\\n\\n")
        f.write(f"Execution Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n\\n")
        
        f.write("EXECUTION PERFORMANCE\\n")
        f.write("-"*70 + "\\n")
        f.write(f"Total Steps: {metrics.total_steps:,}\\n")
        f.write(f"Final Path Length: {metrics.final_path_length:.2f} meters\\n")
        f.write(f"Execution Time: {duration:.2f} seconds\\n")
        f.write(f"Steps per Second: {metrics.total_steps/duration:.1f}\\n\\n")
        
        if metrics.collaboration_history:
            searching_vals = [c[0] for c in metrics.collaboration_history]
            going_vals = [c[1] for c in metrics.collaboration_history]
            pushing_vals = [c[2] for c in metrics.collaboration_history]
            efficiency_vals = [(g + p) / ROBOT_NUMBER * 100 for s, g, p in metrics.collaboration_history]
            
            f.write("COLLABORATION STATISTICS\\n")
            f.write("-"*70 + "\\n")
            f.write(f"Average Robots Searching: {np.mean(searching_vals):.2f} ± {np.std(searching_vals):.2f}\\n")
            f.write(f"Average Robots Going to Crate: {np.mean(going_vals):.2f} ± {np.std(going_vals):.2f}\\n")
            f.write(f"Average Robots Pushing: {np.mean(pushing_vals):.2f} ± {np.std(pushing_vals):.2f}\\n")
            f.write(f"Average Team Efficiency: {np.mean(efficiency_vals):.1f}%\\n")
            f.write(f"Peak Team Efficiency: {np.max(efficiency_vals):.1f}%\\n\\n")
        
        f.write("MODEL CONFIGURATION\\n")
        f.write("-"*70 + "\\n")
        f.write(f"Number of Robots: {ROBOT_NUMBER}\\n")
        f.write(f"Model Input Dim: {NN_INPUT_SIZE}\\n")
        f.write(f"Model Hidden Dim: {NN_HIDDEN}\\n")
        f.write(f"Sequence Length: {SEQ_LEN}\\n")
        f.write(f"K Neighbors: {K_NEIGH}\\n")
        f.write(f"Device: {DEVICE}\\n")
    
    logger.info(f"Saved inference summary to {summary_path}")

def plot_multi_trial_comparison(all_metrics, save_dir=RESULTS_DIR):
    """Generate comparative plots across multiple trials"""
    logger.info("Generating multi-trial comparison plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Multi-Trial Inference Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: Steps comparison
    ax1 = axes[0, 0]
    trial_nums = [m.trial_num for m in all_metrics]
    steps = [m.total_steps for m in all_metrics]
    colors = ['green' if m.success else 'red' for m in all_metrics]
    ax1.bar(trial_nums, steps, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Trial Number')
    ax1.set_ylabel('Total Steps')
    ax1.set_title('Steps per Trial')
    ax1.axhline(np.mean(steps), color='blue', linestyle='--', linewidth=2, label=f'Mean: {np.mean(steps):.1f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Path length comparison
    ax2 = axes[0, 1]
    path_lengths = [m.final_path_length for m in all_metrics]
    ax2.bar(trial_nums, path_lengths, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Trial Number')
    ax2.set_ylabel('Path Length (meters)')
    ax2.set_title('Path Length per Trial')
    ax2.axhline(np.mean(path_lengths), color='blue', linestyle='--', linewidth=2, label=f'Mean: {np.mean(path_lengths):.2f}m')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Execution time comparison
    ax3 = axes[0, 2]
    exec_times = [(m.end_time - m.start_time).total_seconds() for m in all_metrics if m.end_time and m.start_time]
    ax3.bar(trial_nums[:len(exec_times)], exec_times, color=colors[:len(exec_times)], alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Trial Number')
    ax3.set_ylabel('Time (seconds)')
    ax3.set_title('Execution Time per Trial')
    if exec_times:
        ax3.axhline(np.mean(exec_times), color='blue', linestyle='--', linewidth=2, label=f'Mean: {np.mean(exec_times):.2f}s')
        ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: All path trajectories overlaid
    ax4 = axes[1, 0]
    for i, metrics in enumerate(all_metrics):
        if metrics.path_positions:
            path_array = np.array(metrics.path_positions)
            color = 'green' if metrics.success else 'red'
            ax4.plot(path_array[:, 0], path_array[:, 1], linewidth=2, alpha=0.6, label=f'Trial {metrics.trial_num}', color=plt.cm.viridis(i/len(all_metrics)))
    ax4.set_xlabel('X Position (meters)')
    ax4.set_ylabel('Y Position (meters)')
    ax4.set_title('All Path Trajectories')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.set_aspect('equal')
    
    # Plot 5: Success rate and statistics
    ax5 = axes[1, 1]
    ax5.axis('off')
    
    success_count = sum(1 for m in all_metrics if m.success)
    success_rate = success_count / len(all_metrics) * 100 if all_metrics else 0
    
    stats_text = f"""Overall Statistics:
    
Total Trials: {len(all_metrics)}
Successful: {success_count}
Success Rate: {success_rate:.1f}%

Steps:
  Mean: {np.mean(steps):.1f} ± {np.std(steps):.1f}
  Min: {np.min(steps)}
  Max: {np.max(steps)}

Path Length:
  Mean: {np.mean(path_lengths):.2f} ± {np.std(path_lengths):.2f}m
  Min: {np.min(path_lengths):.2f}m
  Max: {np.max(path_lengths):.2f}m"""
    
    if exec_times:
        stats_text += f"""

Execution Time:
  Mean: {np.mean(exec_times):.2f} ± {np.std(exec_times):.2f}s
  Min: {np.min(exec_times):.2f}s
  Max: {np.max(exec_times):.2f}s"""
    
    ax5.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
             family='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # Plot 6: Distribution box plots
    ax6 = axes[1, 2]
    data_to_plot = [steps, path_lengths]
    labels = ['Steps', 'Path Length\n(meters)']
    
    bp = ax6.boxplot(data_to_plot, labels=labels, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightgreen')
        patch.set_alpha(0.7)
    ax6.set_title('Distribution Comparison')
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'multi_trial_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved multi-trial comparison to {save_path}")

def save_multi_trial_summary(all_metrics, save_dir=RESULTS_DIR):
    """Save comprehensive summary of all trials"""
    summary_path = os.path.join(save_dir, 'multi_trial_summary.txt')
    
    with open(summary_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("SwarmGNN-Former Multi-Trial Inference Summary\n")
        f.write("="*70 + "\n\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Overall statistics
        success_count = sum(1 for m in all_metrics if m.success)
        success_rate = success_count / len(all_metrics) * 100 if all_metrics else 0
        
        steps = [m.total_steps for m in all_metrics]
        path_lengths = [m.final_path_length for m in all_metrics]
        exec_times = [(m.end_time - m.start_time).total_seconds() for m in all_metrics if m.end_time and m.start_time]
        
        f.write("OVERALL PERFORMANCE\n")
        f.write("-"*70 + "\n")
        f.write(f"Total Trials: {len(all_metrics)}\n")
        f.write(f"Successful Trials: {success_count}\n")
        f.write(f"Success Rate: {success_rate:.1f}%\n\n")
        
        f.write("STEPS STATISTICS\n")
        f.write("-"*70 + "\n")
        f.write(f"Mean: {np.mean(steps):.1f} ± {np.std(steps):.1f}\n")
        f.write(f"Min: {np.min(steps)}\n")
        f.write(f"Max: {np.max(steps)}\n")
        f.write(f"Median: {np.median(steps):.1f}\n\n")
        
        f.write("PATH LENGTH STATISTICS (meters)\n")
        f.write("-"*70 + "\n")
        f.write(f"Mean: {np.mean(path_lengths):.2f} ± {np.std(path_lengths):.2f}\n")
        f.write(f"Min: {np.min(path_lengths):.2f}\n")
        f.write(f"Max: {np.max(path_lengths):.2f}\n")
        f.write(f"Median: {np.median(path_lengths):.2f}\n\n")
        
        if exec_times:
            f.write("EXECUTION TIME STATISTICS (seconds)\n")
            f.write("-"*70 + "\n")
            f.write(f"Mean: {np.mean(exec_times):.2f} ± {np.std(exec_times):.2f}\n")
            f.write(f"Min: {np.min(exec_times):.2f}\n")
            f.write(f"Max: {np.max(exec_times):.2f}\n")
            f.write(f"Median: {np.median(exec_times):.2f}\n\n")
        
        # Individual trial details
        f.write("\n" + "="*70 + "\n")
        f.write("INDIVIDUAL TRIAL RESULTS\n")
        f.write("="*70 + "\n\n")
        
        for metrics in all_metrics:
            exec_time = (metrics.end_time - metrics.start_time).total_seconds() if metrics.end_time and metrics.start_time else 0
            status = "SUCCESS" if metrics.success else "INCOMPLETE"
            
            f.write(f"Trial {metrics.trial_num}: {status}\n")
            f.write(f"  Steps: {metrics.total_steps}\n")
            f.write(f"  Path Length: {metrics.final_path_length:.2f}m\n")
            f.write(f"  Execution Time: {exec_time:.2f}s\n")
            f.write("\n")
    
    logger.info(f"Saved multi-trial summary to {summary_path}")

# ---------------------------
# Main loop (EXECUTE only)
# ---------------------------
def run_single_trial(trial_num, learned_model, visualize=True):
    """Run a single inference trial"""
    logger.info(f"\n{'='*70}")
    logger.info(f"Starting Trial {trial_num}")
    logger.info(f"{'='*70}")
    
    if visualize:
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption(f"SwarmGNN-Former - Trial {trial_num}")
        clock = pygame.time.Clock()
        font = pygame.font.Font(None, 24)
    
    world = b2World(gravity=(0, 0))
    contact_listener = GoalContactListener()
    world.contactListener = contact_listener

    walls = [Wall(world, WIDTH / 2, -5, WIDTH, 10), Wall(world, WIDTH / 2, HEIGHT + 5, WIDTH, 10), Wall(world, -5, HEIGHT / 2, 10, HEIGHT), Wall(world, WIDTH + 5, HEIGHT / 2, 10, HEIGHT)]

    goal = Goal(world, 0, 0, GOAL_SIZE)
    crate = PushableObj(world, WIDTH / 4, HEIGHT / 2, OBJECT_SIZE)
    robots = [SwarmRobot(world, 0, 0, i) for i in range(ROBOT_NUMBER)]

    for r in robots:
        r.set_model(learned_model)

    contact_listener.hit = reset_positions(robots, crate, goal)
    
    # Initialize metrics tracker
    metrics = InferenceMetrics(trial_num=trial_num)
    metrics.start_time = datetime.now()

    steps = 0
    running = True
    max_steps = inference_cfg.max_steps
    
    while running and steps < max_steps:
        if visualize:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break

            screen.fill((255,255,255))
        
        robots_pushing, robots_going, robots_search = [], [], []

        for robot in robots:
            robot.update_behavior(crate, goal, robots)
            if robot.state == 'push':
                robots_pushing.append(robot)
            elif robot.state == 'go_to_crate':
                robots_going.append(robot)
            elif robot.state == 'search':
                robots_search.append(robot)

        crate_pos = crate.body.position
        
        # Record metrics every 10 steps to reduce overhead
        if steps % 10 == 0:
            metrics.record_step(steps, robots, crate_pos)
        
        if robots_pushing:
            for r in robots_pushing:
                r.signal_crate_location(robots, crate_pos)
        if robots_going and steps % 10 == 0:
            for r in robots_going[:3]:
                r.signal_crate_location(robots, crate_pos)
        if steps % 30 == 0 and (robots_pushing or robots_going) and robots_search:
            if robots_pushing:
                robots_pushing[0].signal_crate_location(robots, crate_pos)
            elif robots_going:
                robots_going[0].signal_crate_location(robots, crate_pos)

        world.Step(TIME_STEP, VEL_ITERS, POS_ITERS)
        crate.update_path()

        if visualize:
            for wall in walls: wall.draw(screen)
            goal.draw(screen)
            crate.draw(screen, robots_pushing)
            for r in robots: r.draw(screen)

            path_text = font.render(f"Path Len: {crate.path_length_m:.2f} m", True, Color("black"))
            screen.blit(path_text, (10,10))
            status_text = font.render(f"Trial: {trial_num} | Steps: {steps}/{max_steps}", True, Color("black"))
            screen.blit(status_text, (10, 35))
            collab_text = font.render(f"Search: {len(robots_search)} | Going: {len(robots_going)} | Pushing: {len(robots_pushing)}", True, Color("black"))
            screen.blit(collab_text, (10, 60))
            pygame.display.flip()
            clock.tick(60)
        
        steps += 1

        if contact_listener.hit:
            logger.info(f"Trial {trial_num}: Crate delivered successfully!")
            logger.info(f"  Total Steps: {steps}")
            logger.info(f"  Final Path Length: {crate.path_length_m:.2f} m")
            
            # Finalize metrics
            metrics.finalize(steps, crate.path_length_m, success=True)
            running = False
            break
    
    # If we hit max steps without success
    if steps >= max_steps and not contact_listener.hit:
        logger.warning(f"Trial {trial_num}: Max steps ({max_steps}) reached without completion")
        metrics.finalize(steps, crate.path_length_m, success=False)
    
    return metrics

def main():
    logger.info("="*70)
    logger.info("SwarmGNN-Former Multi-Trial Inference Execution")
    logger.info("="*70)
    
    pygame.init()

    learned_model = load_model_from_pt(MODEL_PT)
    if learned_model is None:
        logger.error("No model available for inference. Run training_pipeline.py first.")
        return
    
    logger.info(f"Running {inference_cfg.num_trials} inference trials")
    logger.info(f"Max steps per trial: {inference_cfg.max_steps}")
    logger.info(f"Visualization: {'Enabled' if inference_cfg.visualize_trials else 'Disabled'}")
    
    # Run all trials
    all_metrics = []
    for trial_num in range(1, inference_cfg.num_trials + 1):
        metrics = run_single_trial(trial_num, learned_model, visualize=inference_cfg.visualize_trials)
        all_metrics.append(metrics)
        
        # Save individual trial results
        plot_inference_metrics(metrics)
        save_inference_summary(metrics)
    
    pygame.quit()
    
    # Generate multi-trial analysis
    logger.info("\n" + "="*70)
    logger.info("GENERATING MULTI-TRIAL ANALYSIS")
    logger.info("="*70)
    
    plot_multi_trial_comparison(all_metrics)
    save_multi_trial_summary(all_metrics)
    
    # Print summary statistics
    success_count = sum(1 for m in all_metrics if m.success)
    logger.info("\n" + "="*70)
    logger.info("MULTI-TRIAL INFERENCE COMPLETE!")
    logger.info("="*70)
    logger.info(f"Total Trials: {len(all_metrics)}")
    logger.info(f"Successful: {success_count}/{len(all_metrics)} ({success_count/len(all_metrics)*100:.1f}%)")
    logger.info(f"Results saved to: {os.path.abspath(RESULTS_DIR)}")
    logger.info("="*70)

if __name__ == "__main__":
    main()

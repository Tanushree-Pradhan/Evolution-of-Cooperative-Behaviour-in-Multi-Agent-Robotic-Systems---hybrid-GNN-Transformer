"""
Training pipeline for SwarmGNN-Former.
Self contained: config, logging, simulation for data collection, and model training.
"""
import os
import math
import random
import json
from typing import List, Tuple
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
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

def setup_logger(name: str, level: int = logging.INFO, logfile: str = "training.log") -> logging.Logger:
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
# Config (training copy)
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
        self.epochs = 20
        self.learning_rate = 0.001
        self.history_len = 8
        self.k_neighbors = 4

class TrainConfig:
    def __init__(self):
        self.results_dir = "results"
        self.data_filename = "training_data.csv"
        self.model_filename = "trained_model.json"
        self.target_deliveries = 20
        self.collection_steps = 20000
        self.collection_reset_steps = 5000
        self.validation_split = 0.2
        self.early_stopping_patience = 5
        self.batch_size = 64
        self.save_every = 10
        self.collection_save_interval = 5000

class VizConfig:
    def __init__(self):
        self.goal_size_px = 30
        self.goal_x_px = 650
        self.goal_y_px = 300
        self.path_color = (128, 128, 128, 200)

robot_cfg = RobotConfig()
object_cfg = ObjectConfig()
push_cfg = PushConfig()
wall_cfg = WallConfig()
nn_cfg = NNConfig()
train_cfg = TrainConfig()
viz_cfg = VizConfig()

# Derived constants
MODE = 'COLLECT_DATA'
RESULTS_DIR = train_cfg.results_dir
DATA_FILENAME = os.path.join(RESULTS_DIR, train_cfg.data_filename)
MODEL_PT = os.path.join(RESULTS_DIR, train_cfg.model_filename.replace('.json', '.pt') if train_cfg.model_filename.endswith('.json') else train_cfg.model_filename + '.pt')
MODEL_JSON = os.path.join(RESULTS_DIR, train_cfg.model_filename if train_cfg.model_filename.endswith('.json') else train_cfg.model_filename + '.json')

# Create results directory
os.makedirs(RESULTS_DIR, exist_ok=True)

NN_INPUT_SIZE = nn_cfg.input_size
NN_HIDDEN = nn_cfg.hidden_size
NN_OUTPUT_SIZE = nn_cfg.output_size
SEQ_LEN = nn_cfg.history_len

BATCH_SIZE = train_cfg.batch_size
EPOCHS = nn_cfg.epochs
LR = nn_cfg.learning_rate
VAL_SPLIT = train_cfg.validation_split
EARLY_STOPPING = train_cfg.early_stopping_patience
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

# ---------------------------
# Dataset and helpers
# ---------------------------
class SequenceGraphDataset(Dataset):
    def __init__(self, X_seq: np.ndarray, Nbrs: np.ndarray, Y: np.ndarray):
        self.X = torch.tensor(X_seq, dtype=torch.float32)
        self.N = torch.tensor(Nbrs, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.N[idx], self.Y[idx]

def pad_or_cut_neighbors(neigh_list, k=K_NEIGH):
    arr = np.zeros((k, NEIGH_FEAT_DIM), dtype=np.float32)
    for i, v in enumerate(neigh_list[:k]):
        arr[i, :] = np.array(v, dtype=np.float32)
    return arr

# ---------------------------
# Save/load helpers
# ---------------------------
def save_model_and_json(model: nn.Module, optimizer_state, filename_pt=MODEL_PT, filename_json=MODEL_JSON):
    torch.save({'model_state': model.state_dict(), 'optimizer_state': optimizer_state}, filename_pt)
    export = {k: v.cpu().numpy().tolist() for k, v in model.state_dict().items()}
    with open(filename_json, 'w') as f:
        json.dump(export, f, indent=2)
    logger.info(f"Saved model to {filename_pt} and JSON export to {filename_json}")

# ---------------------------
# SwarmRobot (data collection)
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

    def update_behavior(self, crate, goal, all_robots, data_recorder=None):
        crate_pos = crate.body.position
        robot_pos = self.body.position
        goal_pos = goal.body.position
        dist_to_crate = math.hypot(crate_pos[0] - robot_pos[0], crate_pos[1] - robot_pos[1])
        final_vx, final_vy = 0.0, 0.0

        if self.state == 'search':
            self.random_walk()
            final_vx, final_vy = self.body.linearVelocity
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

            final_dir = self.rule_based_control(target_dir, all_robots)
            final_dir = self.apply_wall_repulsion(final_dir)
            if is_actively_pushing:
                final_vx = final_dir.x * PUSH_SPEED * 1.5
                final_vy = final_dir.y * PUSH_SPEED * 1.5
            else:
                final_vx = final_dir.x * ROBOT_MAX_SPEED
                final_vy = final_dir.y * ROBOT_MAX_SPEED

            hist_list = list(self.history)
            if len(hist_list) < SEQ_LEN:
                pad = [[0.0]*NN_INPUT_SIZE] * (SEQ_LEN - len(hist_list))
                seq = pad + hist_list
            else:
                seq = hist_list[-SEQ_LEN:]
            neigh = self.compute_neighbors(all_robots, K=K_NEIGH)
            if data_recorder:
                data_recorder(seq, neigh.tolist(), [final_dir.x, final_dir.y])
            self.body.linearVelocity = (final_vx, final_vy)
        else:
            self.random_walk()
            self.body.linearVelocity = (final_vx, final_vy)

    def draw(self, surf):
        p = vec_m_to_px(self.body.worldCenter)
        color = Color("red") if self.state == 'push' else Color("blue") if self.state == 'go_to_crate' else Color("green")
        pygame.draw.circle(surf, color, p, int(self.size_px), width=0)
        pygame.draw.circle(surf, Color("black"), p, int(self.size_px), width=1)

# ---------------------------
# Data recorder with metrics
# ---------------------------
class DataRecorder:
    def __init__(self):
        self.seqs = []
        self.neighs = []
        self.actions = []
        self.delivery_metrics = []  # Track delivery stats
        self.step_counts = []  # Steps per delivery
        self.path_lengths = []  # Path length per delivery
        self.timestamps = []  # Time per delivery
    def append(self, seq, neigh, action):
        self.seqs.append(seq)
        self.neighs.append(neigh)
        self.actions.append(action)
    def record_delivery(self, delivery_num, steps, path_length, timestamp):
        """Record metrics for a successful delivery"""
        self.delivery_metrics.append({
            'delivery': delivery_num,
            'steps': steps,
            'path_length': path_length,
            'timestamp': timestamp
        })
        self.step_counts.append(steps)
        self.path_lengths.append(path_length)
        self.timestamps.append(timestamp)
    
    def save_as_csv(self, filename=DATA_FILENAME):
        with open(filename, 'w') as f:
            for seq, neigh, act in zip(self.seqs, self.neighs, self.actions):
                flat_seq = [str(x) for row in seq for x in row]
                flat_neigh = [str(x) for row in neigh for x in row]
                line = ",".join(flat_seq + flat_neigh + [str(act[0]), str(act[1])])
                f.write(line + "\n")
        logger.info(f"Saved {len(self.seqs)} samples to {filename}")
    def build_numpy(self):
        if len(self.seqs) == 0:
            return (np.zeros((0, SEQ_LEN, NN_INPUT_SIZE), dtype=np.float32), np.zeros((0, K_NEIGH, NEIGH_FEAT_DIM), dtype=np.float32), np.zeros((0, NN_OUTPUT_SIZE), dtype=np.float32))
        X = np.array(self.seqs, dtype=np.float32)
        N = np.array(self.neighs, dtype=np.float32)
        Y = np.array(self.actions, dtype=np.float32)
        return X, N, Y

# ---------------------------
# Training
# ---------------------------
def train_model(X, Nbrs, Y, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, val_split=VAL_SPLIT, early_stop=EARLY_STOPPING):
    if X.shape[0] == 0:
        logger.error("No training samples!")
        return None
    dataset = SequenceGraphDataset(X, Nbrs, Y)
    N_total = len(dataset)
    val_n = int(N_total * val_split)
    train_n = N_total - val_n
    train_set, val_set = random_split(dataset, [train_n, val_n] if val_n>0 else [N_total, 0])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False) if val_n>0 else None

    model = SwarmGNNFormer().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val = float('inf')
    best_state = None
    patience = 0
    train_losses, val_losses = [], []

    for epoch in range(1, epochs+1):
        model.train()
        running, total = 0.0, 0
        for xb, nb, yb in train_loader:
            xb = xb.to(DEVICE); nb = nb.to(DEVICE); yb = yb.to(DEVICE)
            pred = model(xb, nb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)
            total += xb.size(0)
        train_loss = running / total
        train_losses.append(train_loss)

        val_loss = None
        if val_loader:
            model.eval()
            r, t = 0.0, 0
            with torch.no_grad():
                for xb, nb, yb in val_loader:
                    xb = xb.to(DEVICE); nb = nb.to(DEVICE); yb = yb.to(DEVICE)
                    pred = model(xb, nb)
                    loss = criterion(pred, yb)
                    r += loss.item() * xb.size(0)
                    t += xb.size(0)
            val_loss = r / t
            val_losses.append(val_loss)
            logger.info(f"Epoch {epoch}/{epochs} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
        else:
            logger.info(f"Epoch {epoch}/{epochs} train_loss={train_loss:.6f}")

        if val_loss is not None:
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_state = model.state_dict()
                patience = 0
            else:
                patience += 1
            if patience >= early_stop:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        else:
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)

    save_model_and_json(model, optimizer.state_dict(), filename_pt=MODEL_PT, filename_json=MODEL_JSON)

    # Save learning curves data
    return model, train_losses, val_losses

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
# Main loop for data collection
# ---------------------------
def run_data_collection() -> DataRecorder:
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f"SwarmGNN-Former - Data Collection")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)

    world = b2World(gravity=(0, 0))
    contact_listener = GoalContactListener()
    world.contactListener = contact_listener

    walls = [Wall(world, WIDTH / 2, -5, WIDTH, 10), Wall(world, WIDTH / 2, HEIGHT + 5, WIDTH, 10), Wall(world, -5, HEIGHT / 2, 10, HEIGHT), Wall(world, WIDTH + 5, HEIGHT / 2, 10, HEIGHT)]

    goal = Goal(world, 0, 0, GOAL_SIZE)
    crate = PushableObj(world, WIDTH / 4, HEIGHT / 2, OBJECT_SIZE)
    robots = [SwarmRobot(world, 0, 0, i) for i in range(ROBOT_NUMBER)]

    contact_listener.hit = reset_positions(robots, crate, goal)
    logger.info("Initial environment reset complete for data collection.")

    data_rec = DataRecorder()
    steps = 0
    deliveries_count = 0
    delivery_start_step = 0  # Track steps for each delivery
    start_time = datetime.now()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((255,255,255))
        robots_pushing, robots_going, robots_search = [], [], []

        for robot in robots:
            robot.update_behavior(crate, goal, robots, data_rec.append)
            if robot.state == 'push':
                robots_pushing.append(robot)
            elif robot.state == 'go_to_crate':
                robots_going.append(robot)
            elif robot.state == 'search':
                robots_search.append(robot)

        crate_pos = crate.body.position
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

        for wall in walls: wall.draw(screen)
        goal.draw(screen)
        crate.draw(screen, robots_pushing)
        for r in robots: r.draw(screen)

        path_text = font.render(f"Path Len: {crate.path_length_m:.2f} m", True, Color("black"))
        screen.blit(path_text, (10,10))
        status_text = font.render(f"Deliveries: {deliveries_count}", True, Color("black"))
        screen.blit(status_text, (10, 35))
        collab_text = font.render(f"Search: {len(robots_search)} | Going: {len(robots_going)} | Pushing: {len(robots_pushing)}", True, Color("black"))
        screen.blit(collab_text, (10, 60))
        pygame.display.flip()
        clock.tick(60)
        steps += 1

        if contact_listener.hit:
            deliveries_count += 1
            delivery_steps = steps - delivery_start_step
            timestamp = (datetime.now() - start_time).total_seconds()
            data_rec.record_delivery(deliveries_count, delivery_steps, crate.path_length_m, timestamp)
            logger.info(f"Goal hit! Delivery #{deliveries_count}. Steps: {delivery_steps}. Path Len: {crate.path_length_m:.2f} m.")
            contact_listener.hit = reset_positions(robots, crate, goal)
            delivery_start_step = steps
            if deliveries_count % max(1, int(train_cfg.save_every)) == 0:
                data_rec.save_as_csv(DATA_FILENAME)
        elif steps % train_cfg.collection_save_interval == 0 and steps > 0:
            logger.info(f"Auto-save dataset at step {steps}")
            data_rec.save_as_csv(DATA_FILENAME)

        if steps % train_cfg.collection_reset_steps == 0 and steps < train_cfg.collection_steps:
            logger.info("Periodic reset during data collection.")
            contact_listener.hit = reset_positions(robots, crate, goal)

        if steps >= train_cfg.collection_steps:
            logger.info("Collection safety limit reached. Ending data collection.")
            running = False

        if deliveries_count >= train_cfg.target_deliveries:
            logger.info("Collected target deliveries. Ending collection.")
            running = False

    data_rec.save_as_csv(DATA_FILENAME)
    logger.info("Data collection finished.")
    pygame.quit()
    return data_rec

# ---------------------------
# Analysis and Visualization Functions
# ---------------------------
def print_model_architecture(model):
    """Print detailed model architecture"""
    logger.info("="*70)
    logger.info("MODEL ARCHITECTURE: SwarmGNN-Former")
    logger.info("="*70)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"\nTotal Parameters: {total_params:,}")
    logger.info(f"Trainable Parameters: {trainable_params:,}")
    logger.info(f"\nArchitecture Components:")
    logger.info(f"  - Input Dimension: {NN_INPUT_SIZE}")
    logger.info(f"  - Embedding Dimension: {NN_HIDDEN}")
    logger.info(f"  - Sequence Length: {SEQ_LEN}")
    logger.info(f"  - Number of Neighbors: {K_NEIGH}")
    logger.info(f"  - Neighbor Feature Dim: {NEIGH_FEAT_DIM}")
    logger.info(f"  - Output Dimension: {NN_OUTPUT_SIZE}")
    logger.info(f"\nLayer-by-Layer Breakdown:")
    
    for name, module in model.named_children():
        num_params = sum(p.numel() for p in module.parameters())
        logger.info(f"  {name}: {num_params:,} parameters")
    
    # Save architecture diagram to text file
    arch_file = os.path.join(RESULTS_DIR, 'model_architecture.txt')
    with open(arch_file, 'w') as f:
        f.write("="*70 + "\\n")
        f.write("SwarmGNN-Former Model Architecture\\n")
        f.write("="*70 + "\\n\\n")
        f.write(f"Total Parameters: {total_params:,}\\n")
        f.write(f"Trainable Parameters: {trainable_params:,}\\n\\n")
        f.write("Model Structure:\\n")
        f.write(str(model) + "\\n\\n")
        f.write("Layer-wise Parameter Count:\\n")
        for name, module in model.named_children():
            num_params = sum(p.numel() for p in module.parameters())
            f.write(f"  {name}: {num_params:,} parameters\\n")
    
    logger.info(f"\\nArchitecture saved to {arch_file}")
    logger.info("="*70)

def plot_data_collection_metrics(data_rec):
    """Generate publication-quality plots for data collection phase"""
    logger.info("Generating data collection analysis plots...")
    
    if len(data_rec.delivery_metrics) == 0:
        logger.warning("No delivery metrics to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Data Collection Metrics', fontsize=16, fontweight='bold')
    
    # Plot 1: Steps per delivery over time
    ax = axes[0, 0]
    deliveries = [m['delivery'] for m in data_rec.delivery_metrics]
    steps = [m['steps'] for m in data_rec.delivery_metrics]
    ax.plot(deliveries, steps, 'o-', linewidth=2, markersize=6, color='#2E86AB')
    ax.axhline(np.mean(steps), color='red', linestyle='--', label=f'Mean: {np.mean(steps):.1f}')
    ax.set_xlabel('Delivery Number', fontsize=12)
    ax.set_ylabel('Steps to Complete', fontsize=12)
    ax.set_title('Steps per Delivery', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Path length per delivery
    ax = axes[0, 1]
    path_lengths = [m['path_length'] for m in data_rec.delivery_metrics]
    ax.plot(deliveries, path_lengths, 's-', linewidth=2, markersize=6, color='#A23B72')
    ax.axhline(np.mean(path_lengths), color='green', linestyle='--', label=f'Mean: {np.mean(path_lengths):.2f}m')
    ax.set_xlabel('Delivery Number', fontsize=12)
    ax.set_ylabel('Path Length (meters)', fontsize=12)
    ax.set_title('Path Length per Delivery', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Distribution of actions
    ax = axes[1, 0]
    actions = np.array(data_rec.actions)
    vx = actions[:, 0]
    vy = actions[:, 1]
    ax.scatter(vx, vy, alpha=0.3, s=10, c='#F18F01')
    ax.set_xlabel('vx (normalized)', fontsize=12)
    ax.set_ylabel('vy (normalized)', fontsize=12)
    ax.set_title(f'Action Distribution (n={len(actions):,})', fontsize=13, fontweight='bold')
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axvline(0, color='k', linewidth=0.5)
    circle = plt.Circle((0, 0), 1, fill=False, color='red', linestyle='--', linewidth=1.5)
    ax.add_patch(circle)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Cumulative data samples
    ax = axes[1, 1]
    cumulative_samples = np.cumsum([m['steps'] for m in data_rec.delivery_metrics])
    ax.plot(deliveries, cumulative_samples, '^-', linewidth=2, markersize=6, color='#06A77D')
    ax.set_xlabel('Delivery Number', fontsize=12)
    ax.set_ylabel('Cumulative Samples', fontsize=12)
    ax.set_title('Data Collection Progress', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, 'data_collection_metrics.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved data collection metrics to {save_path}")

def plot_training_results(train_losses, val_losses, model, X, Nbrs, Y):
    """Generate comprehensive training analysis plots"""
    logger.info("Generating training analysis plots...")
    
    # Create multi-panel figure
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Plot 1: Learning curves
    ax1 = fig.add_subplot(gs[0, :2])
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'o-', linewidth=2, markersize=4, label='Training Loss', color='#D62828')
    if len(val_losses) > 0:
        ax1.plot(epochs, val_losses, 's-', linewidth=2, markersize=4, label='Validation Loss', color='#003049')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('MSE Loss', fontsize=12)
    ax1.set_title('Learning Curves', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Plot 2: Loss statistics
    ax2 = fig.add_subplot(gs[0, 2])
    final_train = train_losses[-1] if train_losses else 0
    final_val = val_losses[-1] if val_losses else 0
    best_val = min(val_losses) if val_losses else 0
    stats_text = f"Final Train: {final_train:.6f}\\n"
    if val_losses:
        stats_text += f"Final Val: {final_val:.6f}\\n"
        stats_text += f"Best Val: {best_val:.6f}\\n"
        stats_text += f"Epochs: {len(train_losses)}"
    ax2.text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax2.axis('off')
    ax2.set_title('Training Statistics', fontsize=13, fontweight='bold')
    
    # Plot 3-4: Data distribution
    ax3 = fig.add_subplot(gs[1, 0])
    input_magnitudes = np.linalg.norm(Y, axis=1)
    ax3.hist(input_magnitudes, bins=50, edgecolor='black', alpha=0.7, color='#F77F00')
    ax3.set_xlabel('Action Magnitude', fontsize=11)
    ax3.set_ylabel('Frequency', fontsize=11)
    ax3.set_title('Action Magnitude Distribution', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 5: Prediction scatter (sample validation)
    ax4 = fig.add_subplot(gs[1, 1])
    model.eval()
    with torch.no_grad():
        sample_idx = np.random.choice(len(X), min(500, len(X)), replace=False)
        X_sample = torch.tensor(X[sample_idx], dtype=torch.float32, device=DEVICE)
        N_sample = torch.tensor(Nbrs[sample_idx], dtype=torch.float32, device=DEVICE)
        Y_sample = Y[sample_idx]
        pred_sample = model(X_sample, N_sample).cpu().numpy()
    
    ax4.scatter(Y_sample[:, 0], pred_sample[:, 0], alpha=0.5, s=10, label='vx', color='#2A9D8F')
    ax4.scatter(Y_sample[:, 1], pred_sample[:, 1], alpha=0.5, s=10, label='vy', color='#E76F51')
    ax4.plot([-1, 1], [-1, 1], 'k--', linewidth=1, label='Perfect')
    ax4.set_xlabel('True Action', fontsize=11)
    ax4.set_ylabel('Predicted Action', fontsize=11)
    ax4.set_title('Prediction vs Ground Truth', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_aspect('equal')
    
    # Plot 6: Residual analysis
    ax5 = fig.add_subplot(gs[1, 2])
    residuals = Y_sample - pred_sample
    residual_magnitudes = np.linalg.norm(residuals, axis=1)
    ax5.hist(residual_magnitudes, bins=30, edgecolor='black', alpha=0.7, color='#8338EC')
    ax5.set_xlabel('Prediction Error', fontsize=11)
    ax5.set_ylabel('Frequency', fontsize=11)
    ax5.set_title('Residual Distribution', fontsize=12, fontweight='bold')
    ax5.axvline(np.mean(residual_magnitudes), color='red', linestyle='--', 
                label=f'Mean: {np.mean(residual_magnitudes):.4f}')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Plot 7-8-9: Feature analysis
    ax6 = fig.add_subplot(gs[2, 0])
    last_seq = X[:, -1, :]  # Last timestep features
    for i in range(min(4, last_seq.shape[1])):
        ax6.hist(last_seq[:, i], bins=30, alpha=0.5, label=f'Feature {i}')
    ax6.set_xlabel('Feature Value', fontsize=11)
    ax6.set_ylabel('Frequency', fontsize=11)
    ax6.set_title('Input Feature Distributions', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3, axis='y')
    
    # Neighbor feature analysis
    ax7 = fig.add_subplot(gs[2, 1])
    neigh_flat = Nbrs.reshape(-1, NEIGH_FEAT_DIM)
    for i in range(NEIGH_FEAT_DIM):
        ax7.hist(neigh_flat[:, i], bins=30, alpha=0.5, label=f'Neigh {i}')
    ax7.set_xlabel('Neighbor Feature Value', fontsize=11)
    ax7.set_ylabel('Frequency', fontsize=11)
    ax7.set_title('Neighbor Feature Distributions', fontsize=12, fontweight='bold')
    ax7.legend(fontsize=8)
    ax7.grid(True, alpha=0.3, axis='y')
    
    # Dataset info
    ax8 = fig.add_subplot(gs[2, 2])
    dataset_info = f"Dataset Size: {len(X):,}\\n"
    dataset_info += f"Sequence Length: {SEQ_LEN}\\n"
    dataset_info += f"Input Dim: {NN_INPUT_SIZE}\\n"
    dataset_info += f"Neighbors: {K_NEIGH}\\n"
    dataset_info += f"Batch Size: {BATCH_SIZE}\\n"
    dataset_info += f"Device: {DEVICE}\\n"
    dataset_info += f"Train Samples: {int(len(X)*(1-VAL_SPLIT)):,}\\n"
    dataset_info += f"Val Samples: {int(len(X)*VAL_SPLIT):,}"
    ax8.text(0.1, 0.5, dataset_info, fontsize=10, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax8.axis('off')
    ax8.set_title('Dataset Information', fontsize=13, fontweight='bold')
    
    fig.suptitle('Training Analysis & Model Performance', fontsize=16, fontweight='bold', y=0.995)
    
    save_path = os.path.join(RESULTS_DIR, 'training_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved training analysis to {save_path}")
    
    # Also save simple learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'o-', linewidth=2, label='Training Loss', color='#D62828')
    if len(val_losses) > 0:
        plt.plot(epochs, val_losses, 's-', linewidth=2, label='Validation Loss', color='#003049')
    plt.xlabel('Epoch', fontsize=13)
    plt.ylabel('MSE Loss', fontsize=13)
    plt.title('Learning Curves', fontsize=15, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, 'learning_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved learning curves to {save_path}")

def save_training_summary(data_rec, train_losses, val_losses, X, Y):
    """Save comprehensive training summary report"""
    summary_path = os.path.join(RESULTS_DIR, 'training_summary.txt')
    
    with open(summary_path, 'w') as f:
        f.write("="*70 + "\\n")
        f.write("SwarmGNN-Former Training Summary\\n")
        f.write("="*70 + "\\n\\n")
        f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n\\n")
        
        f.write("DATA COLLECTION PHASE\\n")
        f.write("-"*70 + "\\n")
        f.write(f"Total Deliveries: {len(data_rec.delivery_metrics)}\\n")
        f.write(f"Total Samples Collected: {len(data_rec.seqs):,}\\n")
        if data_rec.step_counts:
            f.write(f"Avg Steps per Delivery: {np.mean(data_rec.step_counts):.1f} ± {np.std(data_rec.step_counts):.1f}\\n")
            f.write(f"Avg Path Length: {np.mean(data_rec.path_lengths):.2f}m ± {np.std(data_rec.path_lengths):.2f}m\\n")
        f.write("\\n")
        
        f.write("TRAINING PHASE\\n")
        f.write("-"*70 + "\\n")
        f.write(f"Dataset Size: {len(X):,} sequences\\n")
        f.write(f"Training Samples: {int(len(X)*(1-VAL_SPLIT)):,}\\n")
        f.write(f"Validation Samples: {int(len(X)*VAL_SPLIT):,}\\n")
        f.write(f"Epochs Completed: {len(train_losses)}\\n")
        f.write(f"Final Training Loss: {train_losses[-1]:.6f}\\n")
        if val_losses:
            f.write(f"Final Validation Loss: {val_losses[-1]:.6f}\\n")
            f.write(f"Best Validation Loss: {min(val_losses):.6f}\\n")
        f.write("\\n")
        
        f.write("MODEL CONFIGURATION\\n")
        f.write("-"*70 + "\\n")
        f.write(f"Input Dimension: {NN_INPUT_SIZE}\\n")
        f.write(f"Hidden Dimension: {NN_HIDDEN}\\n")
        f.write(f"Output Dimension: {NN_OUTPUT_SIZE}\\n")
        f.write(f"Sequence Length: {SEQ_LEN}\\n")
        f.write(f"K Neighbors: {K_NEIGH}\\n")
        f.write(f"Batch Size: {BATCH_SIZE}\\n")
        f.write(f"Learning Rate: {LR}\\n")
        f.write(f"Device: {DEVICE}\\n")
        f.write("\\n")
        
        f.write("DATA STATISTICS\\n")
        f.write("-"*70 + "\\n")
        action_magnitudes = np.linalg.norm(Y, axis=1)
        f.write(f"Action Magnitude: {np.mean(action_magnitudes):.4f} ± {np.std(action_magnitudes):.4f}\\n")
        f.write(f"Action vx range: [{Y[:,0].min():.4f}, {Y[:,0].max():.4f}]\\n")
        f.write(f"Action vy range: [{Y[:,1].min():.4f}, {Y[:,1].max():.4f}]\\n")
    
    logger.info(f"Saved training summary to {summary_path}")

# ---------------------------
# Orchestration
# ---------------------------
def run_training_cycle():
    logger.info("="*70)
    logger.info("START SwarmGNN-Former Training Cycle (separated)")
    logger.info("="*70)

    # Clean up old results in results directory
    old_files = [DATA_FILENAME, MODEL_PT, MODEL_JSON]
    for path in old_files:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass

    # Phase 1: Data Collection
    logger.info("\\n" + "="*70)
    logger.info("PHASE 1: DATA COLLECTION")
    logger.info("="*70)
    data_rec = run_data_collection()
    
    # Generate data collection plots
    plot_data_collection_metrics(data_rec)

    # Phase 2: Parse and prepare dataset
    logger.info("\\n" + "="*70)
    logger.info("PHASE 2: DATASET PREPARATION")
    logger.info("="*70)
    raw_X, raw_N, raw_Y = [], [], []
    with open(DATA_FILENAME, 'r') as f:
        for line in f:
            vals = [float(x) for x in line.strip().split(',')]
            expected = SEQ_LEN * NN_INPUT_SIZE + K_NEIGH * NEIGH_FEAT_DIM + 2
            if len(vals) != expected:
                logger.warning("Skipping line with incorrect length.")
                continue
            offset = 0
            seq = []
            for _ in range(SEQ_LEN):
                seq.append(vals[offset: offset + NN_INPUT_SIZE]); offset += NN_INPUT_SIZE
            neigh = []
            for _ in range(K_NEIGH):
                neigh.append(vals[offset: offset + NEIGH_FEAT_DIM]); offset += NEIGH_FEAT_DIM
            act = vals[offset: offset + 2]
            raw_X.append(seq); raw_N.append(neigh); raw_Y.append(act)

    if len(raw_X) == 0:
        logger.error("No valid sequences after parsing. Aborting.")
        return

    X = np.array(raw_X, dtype=np.float32)
    Nbrs = np.array(raw_N, dtype=np.float32)
    Y = np.array(raw_Y, dtype=np.float32)
    logger.info(f"Dataset shapes: X={X.shape}, Nbrs={Nbrs.shape}, Y={Y.shape}")

    # Phase 3: Model Training
    logger.info("\\n" + "="*70)
    logger.info("PHASE 3: MODEL TRAINING")
    logger.info("="*70)
    result = train_model(X, Nbrs, Y)
    if result is None:
        logger.error("Training failed. Aborting.")
        return
    
    model, train_losses, val_losses = result

    # Phase 4: Analysis and Visualization
    logger.info("\\n" + "="*70)
    logger.info("PHASE 4: ANALYSIS & VISUALIZATION")
    logger.info("="*70)
    
    # Print and save model architecture
    print_model_architecture(model)
    
    # Generate training analysis plots
    plot_training_results(train_losses, val_losses, model, X, Nbrs, Y)
    
    # Save comprehensive summary
    save_training_summary(data_rec, train_losses, val_losses, X, Y)

    logger.info("\\n" + "="*70)
    logger.info("TRAINING COMPLETE!")
    logger.info("="*70)
    logger.info(f"All results saved to: {os.path.abspath(RESULTS_DIR)}")
    logger.info("Run inference_pipeline.py for execution mode.")
    logger.info("="*70)

if __name__ == "__main__":
    run_training_cycle()

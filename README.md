# SwarmGNN-Former: Separated Training and Inference Pipelines

## Overview
This directory contains fully self-contained, separated training and inference pipelines for the SwarmGNN-Former model, designed for swarm robotics imitation learning with comprehensive research publication features.

## Files

### Main Scripts
- **`training_pipeline.py`** - Complete training pipeline with data collection, model training, and analysis
- **`inference_pipeline.py`** - Execution pipeline using trained models with performance metrics

### Configuration
Both files are self-contained with embedded configurations:
- Robot parameters (size, speed, behavior)
- Object and environment settings
- Neural network architecture
- Training hyperparameters
- Visualization settings

All configurations are duplicated separately in each file for complete independence.

## Results Directory Structure

All outputs are saved to the `../results/` directory:

### Training Outputs
- `trained_model.pt` - PyTorch model checkpoint
- `trained_model.json` - Model weights in JSON format
- `training_data.csv` - Collected expert demonstration data
- `model_architecture.txt` - Detailed model architecture and parameter counts
- `training_summary.txt` - Comprehensive training statistics and metrics

### Training Visualizations
- `data_collection_metrics.png` - 4-panel analysis of data collection phase
  - Steps per delivery over time
  - Path length per delivery
  - Action distribution scatter plot
  - Cumulative sample collection progress
  
- `training_analysis.png` - 9-panel comprehensive training analysis
  - Learning curves (train/val loss)
  - Training statistics summary
  - Action magnitude distribution
  - Prediction vs ground truth scatter
  - Residual distribution
  - Input feature distributions
  - Neighbor feature distributions
  - Dataset information

- `learning_curves.png` - Clean publication-ready learning curves

### Inference Outputs
- `inference_summary.txt` - Individual trial execution metrics and statistics
- `multi_trial_summary.txt` - Aggregated statistics across all trials
- `inference_execution_metrics.png` - 4-panel execution analysis per trial
  - Step count progression over time
  - Team collaboration distribution (searching/going/pushing)
  - Crate path trajectory with start/end markers
  - Execution performance summary with efficiency metrics

### Multi-Trial Analysis
- `multi_trial_comparison.png` - 6-panel comparative analysis across trials
  - Steps per trial with mean baseline
  - Path length per trial with mean baseline
  - Execution time comparison
  - All path trajectories overlaid with color coding
  - Overall statistics (success rate, means, std deviations)
  - Distribution box plots for steps and path lengths

### Log Files
- `training.log` - Training phase log
- `inference.log` - Inference phase log

## Usage

### 1. Training Phase
```bash
python training_pipeline.py
```

**Process:**
1. Data Collection: Robots use expert rule-based behavior to collect demonstrations
2. Dataset Preparation: Parse and validate collected sequences
3. Model Training: Train SwarmGNN-Former with transformer + GNN architecture
4. Analysis: Generate all plots, summaries, and architecture diagrams

**Output:** All training artifacts saved to `../results/`

### 2. Inference Phase
```bash
python inference_pipeline.py
```

**Process:**
1. Load trained model from `../results/trained_model.pt`
2. Run multiple independent trials (default: 5 trials)
3. Track performance metrics for each trial
4. Generate individual trial analysis
5. Create comparative multi-trial analysis

**Output:** Individual trial metrics + aggregated multi-trial comparison saved to `../results/`

**Configuration Options:**
Edit `InferenceConfig` class in `inference_pipeline.py`:
```python
class InferenceConfig:
    def __init__(self):
        self.num_trials = 5  # Number of inference runs (default: 5)
        self.max_steps = 1000  # Maximum steps per trial (default: 1000)
        self.visualize_trials = True  # Show pygame window (default: True)
```

## Model Architecture

**SwarmGNN-Former** combines:
- **Temporal Pathway**: Transformer encoder over historical robot state sequences
- **Neighbor Pathway**: MLP processing K-nearest neighbor features
- **Fusion Network**: Combines temporal and spatial information for action prediction

### Key Features
- Input: 6D state features (direction, distance, speed, position)
- Sequence Length: 8 timesteps
- K-Neighbors: 4 nearest robots
- Total Parameters: ~5,120 (fully trainable)

## Research Publication Features

### Data Collection Metrics
- Delivery statistics (steps, path lengths, timestamps)
- Sample collection progress tracking
- Action distribution analysis
- Temporal performance trends

### Training Metrics
- Learning curves with train/validation splits
- Prediction accuracy visualizations
- Residual analysis
- Feature distribution analysis
- Dataset statistics

### Inference Metrics
- Robot state transitions over time
- Team collaboration efficiency
- Trajectory visualization
- Execution performance summary
- Real-time collaboration statistics

## Key Improvements

1. **Complete Separation**: Training and inference are fully independent
2. **Self-Contained Config**: No external config files needed
3. **Organized Outputs**: All results in dedicated directory
4. **Publication-Ready Plots**: High-DPI, well-labeled visualizations
5. **Comprehensive Logging**: Detailed progress tracking and summaries
6. **Model Architecture Documentation**: Complete parameter breakdown
7. **Performance Analysis**: Quantitative metrics for research evaluation

## Dependencies

- Python 3.7+
- PyTorch
- NumPy
- Matplotlib
- Seaborn
- Pygame
- Box2D (pybox2d)

## Configuration Customization

Edit the configuration classes at the top of each file:
- `RobotConfig` - Robot behavior parameters
- `ObjectConfig` - Pushable object properties
- `PushConfig` - Pushing behavior settings
- `WallConfig` - Wall avoidance parameters
- `NNConfig` - Neural network architecture
- `TrainConfig` - Training hyperparameters (training_pipeline.py only)
- `VizConfig` - Visualization settings

## Notes

- Results directory is automatically created if it doesn't exist
- Training collects 20 successful deliveries by default
- GPU acceleration used if available (CUDA)
- All plots saved at 300 DPI for publication quality
- Metrics tracked every 10 steps during inference to minimize overhead

# Learning Transferable Cooperative Behavior in Multi-Agent Teams

[Akshat Agarwal](https://agakshat.github.io)\*, [Sumit Kumar](https://sumitsk.github.io)\*, [Katia Sycara](http://www.cs.cmu.edu/~sycara/)

Robotics Institute, Carnegie Mellon University

Official repository of the 'Learning Transferable Cooperative Behavior in Multi-Agent Teams' paper: https://arxiv.org/abs/1906.01202

Presented at the [Learning and Reasoning with Graph-Structured Representations](https://graphreason.github.io/) workshop ([paper](https://graphreason.github.io/papers/29.pdf)) at ICML 2019, Long Beach, USA.

---

## Changes from Original

This fork includes several upgrades:

- **VMAS GPU acceleration** — `simple_spread` training runs on GPU via [VectorizedMultiAgentSimulator](https://github.com/proroklab/VectorizedMultiAgentSimulator) (v1.5.2), eliminating CPU-GPU roundtrips in the training hot path
- **MPE2 environment adapter** — `simple_formation` and `simple_line` use PettingZoo MPE2 with a gym-compatible wrapper (`mpe2_adapter.py`)
- **Success bonus** — optional `+5` reward when all agents simultaneously cover their landmarks (Hungarian matching)
- **Python 3.9 / PyTorch 1.12+** — tested with conda environment `rl`

## Installation

```bash
pip install -r requirements.txt
```

For VMAS support, clone the simulator:

```bash
cd marl_transfer
git clone https://github.com/proroklab/VectorizedMultiAgentSimulator.git
```

Required packages: `torch`, `numpy`, `scipy`, `gym`, `gymnasium`, `pygame`, `tensorboardX`, `pettingzoo`.

## Environments

| Environment | Backend | GPU | Description |
|---|---|---|---|
| `simple_spread` | **VMAS** | Yes | Agents spread to cover distinct landmarks |
| `simple_formation` | MPE2 | No | Agents form a circle around a landmark |
| `simple_line` | MPE2 | No | Agents distribute evenly between two landmarks |

## Usage

See `arguments.py` for the full list of command-line arguments.

### GPU Training (simple_spread via VMAS)

```bash
# Train 3 agents on GPU, 32 parallel environments
python main.py --env-name simple_spread --num-agents 3 --entity-mp --success-bonus --save-dir 0

# Train 6 agents, 64 parallel environments
python main.py --env-name simple_spread --num-agents 6 --num-envs 64 --entity-mp --success-bonus --save-dir 0
```

Key VMAS arguments:

| Argument | Default | Description |
|---|---|---|
| `--num-envs` | 32 | Number of parallel GPU environments |
| `--success-bonus` | False | +5 reward when all agents cover landmarks |
| `--entity-mp` | False | Enable entity message passing (required for transfer) |

### CPU Training (formation / line via MPE2)

```bash
python main.py --env-name simple_formation --num-agents 4 --num-processes 32 --save-dir 0
python main.py --env-name simple_line --num-agents 5 --num-processes 32 --save-dir 0
```

### Curriculum / Transfer Training

Train sequentially from `N` to `M` agents:

```bash
# Step 1: train 5 agents
python main.py --env-name simple_spread --num-agents 5 --entity-mp --success-bonus --save-dir stage_5

# Step 2: transfer to 10 agents from the 5-agent checkpoint
python main.py --env-name simple_spread --num-agents 10 --entity-mp --success-bonus \
    --continue-training --load-dir ../marlsave/save_new/stage_5/ep400.pt --save-dir stage_10
```

For automated curriculum training, edit `automate.py` and run:

```bash
python automate.py --env-name simple_spread --entity-mp --success-bonus --save-dir 0
```

### Inference / Testing

Test trained checkpoints on the VMAS environment:

```bash
python vmas_env/test_inference.py
```

### Save Format

Checkpoints are saved as `.pt` files containing:

```python
{
    'models': [state_dict, ...],  # one per policy
    'ob_rms': (mean, var) or None,
}
```

## Results

Trained models (curriculum learning) are in `marlsave/save_new/`. Evaluation videos in `videos/`.

## Contact

For queries, raise an issue or contact the authors at sumit.sks4@gmail.com or agarwalaks30@gmail.com.

## License

MIT License.

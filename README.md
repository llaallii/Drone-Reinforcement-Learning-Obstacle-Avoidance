# 🚁 Drone Forest Navigation with Reinforcement Learning

## 🎯 Research Overview

A solo research project exploring **Attention-Based Reinforcement Learning for Autonomous Drone Navigation in Forest Environments**. Starting with simple environments and progressively building toward complex forest navigation with novel RL approaches.

### 🏗️ Current Status: Foundation Phase
- ✅ Project structure created
- 🔄 Environment development in progress
- ⏳ Baseline implementation planned
- ⏳ Novel method development planned

### 🎪 Baby Steps Approach
This project follows a gradual complexity increase:
1. **Simple Environment** → Basic obstacle avoidance
2. **Medium Complexity** → Multi-obstacle navigation  
3. **Complex Forest** → Dense vegetation navigation
4. **Novel Contributions** → Attention mechanisms + hierarchical control

## 🚀 Quick Start (Current Phase)

### Installation
```bash
# Clone repository
git clone https://github.com/llaallii/Drone-Reinforcement-Learning-Obstacle-Avoidance.git
cd drone_forest_navigation_rl

# Create environment
conda create -n drone_rl python=3.11
conda activate drone_rl

# Install basics (start small!)
pip install genesis-world torch gymnasium matplotlib jupyter

# Install in dev mode
pip install -e .
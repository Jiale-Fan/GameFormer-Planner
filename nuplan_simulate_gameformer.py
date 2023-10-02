from nuplan.planning.script.run_simulation import main as main_simulation
import os
import hydra
from nuplan.planning.script.utils import set_default_path
import logging
import Planner.planner

SPLIT='val14_split'
CHALLENGE='open_loop_boxes' # open_loop_boxes, closed_loop_nonreactive_agents, closed_loop_reactive_agents
CHECKPOINT='/data1/nuplan/jiale/exp/exp/training/training_autobots_ego_model/2023.09.26.17.43.27/checkpoints/last.ckpt'
MODEL='autobots_model'

logging.getLogger('numba').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# If set, use the env. variable to overwrite the default dataset and experiment paths
set_default_path()

# If set, use the env. variable to overwrite the Hydra config
CONFIG_PATH = "../nuplan-devkit/nuplan/planning/script/config/simulation"

# if os.environ.get('NUPLAN_HYDRA_CONFIG_PATH') is not None:
#     CONFIG_PATH = os.path.join('../../../../', CONFIG_PATH)

# if os.path.basename(CONFIG_PATH) != 'simulation':
#     CONFIG_PATH = os.path.join(CONFIG_PATH, 'simulation')
CONFIG_NAME = 'default_simulation'


# Initialize configuration management system
hydra.core.global_hydra.GlobalHydra.instance().clear()  # reinitialize hydra if already initialized
hydra.initialize(CONFIG_PATH)

cfg = hydra.compose(config_name=CONFIG_NAME, overrides=[
        f'+simulation={CHALLENGE}',
        'planner=gameformer_planner',

        f'scenario_filter={SPLIT}',
        'scenario_builder=nuplan',
        'hydra.searchpath=["file:///home/jiale/GameFormer-Planner/configs", "pkg://nuplan.planning.script.config.common", "pkg://nuplan.planning.script.experiments"]'
    ])


main_simulation(cfg)





# hydra.searchpath="[pkg://viplan.planning.script.config.common, pkg://viplan.planning.script.config.training, pkg://viplan.planning.script.experiments, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]"
# tensorboard --logdir=/data1/nuplan/jiale/exp/exp/training/training_autobots_ego_model
from typing import List, Optional

import numpy as np
import numpy.typing as npt
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.maps.abstract_map_objects import LaneGraphEdgeMapObject
from nuplan.planning.simulation.planner.abstract_planner import PlannerInput
from nuplan.planning.simulation.trajectory.interpolated_trajectory import (
    InterpolatedTrajectory,
)
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from tuplan_garage.planning.simulation.planner.pdm_planner.abstract_pdm_planner import (
    AbstractPDMPlanner,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.observation.pdm_observation import (
    PDMObservation,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.proposal.batch_idm_policy import (
    BatchIDMPolicy,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.proposal.pdm_generator import (
    PDMGenerator,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.proposal.pdm_proposal import (
    PDMProposalManager,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import (
    PDMScorer,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import (
    PDMSimulator,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_emergency_brake import (
    PDMEmergencyBrake,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
    parallel_discrete_path,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_path import PDMPath

# new imports
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from tuplan_garage.planning.simulation.planner.pdm_planner.observation.pdm_observation_utils import (
    get_drivable_area_map,
)

from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization
from typing import Any, List, Optional, Type
from nuplan.planning.simulation.observation.observation_type import Observation
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
import torch

count_no_route = 0

class MetricsComputer(AbstractPDMPlanner):
    """
    Metrics computer modified on the basis of tuplan-garage's AbstractPDMClosedPlanner
    """

    def __init__(
        self,
        map_radius: float = 60.0,
    ):
        """
        Constructor for AbstractPDMClosedPlanner
        :param trajectory_sampling: Sampling parameters for final trajectory
        :param proposal_sampling: Sampling parameters for proposals
        :param idm_policies: BatchIDMPolicy class
        :param lateral_offsets: centerline offsets for proposals (optional)
        :param map_radius: radius around ego to consider
        """

        super(MetricsComputer, self).__init__(map_radius)
        self.proposal_sampling = TrajectorySampling(num_poses=80, interval_length=0.1) # TODO check num poses
        self._scorer = PDMScorer(self.proposal_sampling)
        self._simulator = PDMSimulator(self.proposal_sampling)

    def compute_metrics_batch(self, trajectories: torch.Tensor, scenarios: List[NuPlanScenario]) -> np.float64:
        trajectory_list = np.split(trajectories, trajectories.shape[0])
        metrics_list = [self.compute_metrics(trajectory, scenario) for trajectory, scenario in zip(trajectory_list, scenarios)]
        mean_metrics = np.mean(metrics_list)
        return mean_metrics, metrics_list

    def compute_metrics(self, trajectory: torch.Tensor, scenario: NuPlanScenario) -> np.float64:
        self._map_api = scenario.map_api
        
        # prepare ego states and observation 
        simulation_history_buffer_duration = 2
        buffer_size = int(simulation_history_buffer_duration / scenario.database_interval + 1)

        history = SimulationHistoryBuffer.initialize_from_scenario(
                buffer_size=buffer_size, 
                scenario=scenario, 
                observation_type=DetectionsTracks) 

        ego_state, observation = history.current_state
        self._drivable_area_map = get_drivable_area_map(
            self._map_api, ego_state, self._map_radius
        )

        # prepare route dicts
        self._load_route_dicts(scenario._route_roadblock_ids)
        if len(self._route_lane_dict) == 0:
            global count_no_route
            count_no_route += 1
            print("No route lane dict found: " + str(count_no_route))
            return np.array([0.0])

        # prepare observation
        trajectory_sampling = TrajectorySampling(num_poses=80, interval_length=0.1)
        _observation = PDMObservation(
                trajectory_sampling, self.proposal_sampling, self._map_radius
            )

        # update observation
        _observation.update(
            ego_state,
            observation,
            list(scenario.get_traffic_light_status_at_iteration(0)),
            self._route_lane_dict,
        )

        # prepare centerline
        current_lane = self._get_starting_lane(ego_state)
        centerline_discrete_path = self._get_discrete_centerline(current_lane)
        _centerline = PDMPath(centerline_discrete_path)

        # get simulated proposal
        proposals_array = np.zeros((trajectory.shape[0], trajectory.shape[1]+1, 11))
        proposals_array[:, 1:, :3] = coordinates_transform(trajectory.cpu().detach().numpy(), ego_state)
        # proposal must be in universal coordinates
        proposals_array[:, 0, :3] = np.array([ego_state.rear_axle.x, ego_state.rear_axle.y, ego_state.rear_axle.heading])

        simulated_proposals_array = self._simulator.simulate_proposals( # Simulation uncorrect, maybe coordinates inconsistency
            proposals_array, ego_state
        ) # [15, 41, 11]

        # TODO DEBUG tracking performance

        proposal_scores = self._scorer.score_proposals(
                    simulated_proposals_array, # TODO: check dimensions
                    ego_state,
                    _observation,
                    _centerline,
                    self._route_lane_dict,
                    self._drivable_area_map,
                    scenario.map_api,
                    )

        return proposal_scores



    # fake methods
    def name(self) -> str:
        """
        :return string describing name of this planner.
        """
        pass

    def initialize(self, initialization: PlannerInitialization) -> None:
        """
        Initialize planner
        :param initialization: Initialization class.
        """
        pass


    def observation_type(self) -> Type[Observation]:
        """
        :return Type of observation that is expected in compute_trajectory.
        """
        pass

    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        """
        Computes the ego vehicle trajectory.
        :param current_input: List of planner inputs for which trajectory needs to be computed.
        :return: Trajectories representing the predicted ego's position in future
        """
        pass

def coordinates_transform(proposal: npt.NDArray, ego_state: EgoState):
    '''
    Transform proposal from ego coordinates to universal coordinates
    '''
    ego_state_array = np.array([ego_state.rear_axle.x, ego_state.rear_axle.y, ego_state.rear_axle.heading])
    theta = ego_state.rear_axle.heading
    rotation_mat = np.array([[np.cos(theta), np.sin(theta)],
                             [-np.sin(theta), np.cos(theta)]])
    new_proposal = np.zeros(proposal.shape)
    new_proposal[..., :2] = np.matmul(proposal[..., :2], rotation_mat)
    new_proposal[..., :2] += ego_state_array[:2]
    new_proposal[..., 2] = proposal[..., 2] + theta
    return new_proposal

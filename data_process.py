import os
import argparse
from tqdm import tqdm
from common_utils import *
from GameFormer.data_utils import *
import matplotlib.pyplot as plt
from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioMapping
from hydra import initialize, compose
import hydra.utils
import pickle
import math
from shapely import Point, LineString
from Planner.planner_utils import *
from Planner.state_lattice_path_planner import LatticePlanner
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks

# define data processor
class DataProcessor(object):
    def __init__(self, scenarios):
        self._scenarios = scenarios

        self.past_time_horizon = 2 # [seconds]
        self.num_past_poses = 10 * self.past_time_horizon 
        self.future_time_horizon = 8 # [seconds]
        self.num_future_poses = 10 * self.future_time_horizon
        self.num_agents = 20

        self._map_features = ['LANE', 'ROUTE_LANES', 'CROSSWALK'] # name of map features to be extracted.
        self._max_elements = {'LANE': 40, 'ROUTE_LANES': 10, 'CROSSWALK': 5} # maximum number of elements to extract per feature layer.
        self._max_points = {'LANE': 50, 'ROUTE_LANES': 50, 'CROSSWALK': 30} # maximum number of points per feature to extract per feature layer.
        self._radius = 60 # [m] query radius scope relative to the current pose.
        self._interpolation_method = 'linear' # Interpolation method to apply when interpolating to maintain fixed size map elements.

        self._target_speed = 13.0 # [m/s]
        self._max_path_length = 50 # [m]
        self.proposals_num = 12
        self.subsampling_ratio = 10

    def get_ego_agent(self):
        self.anchor_ego_state = self.scenario.initial_ego_state
        
        past_ego_states = self.scenario.get_ego_past_trajectory(
            iteration=0, num_samples=self.num_past_poses, time_horizon=self.past_time_horizon
        )

        sampled_past_ego_states = list(past_ego_states) + [self.anchor_ego_state]
        past_ego_states_tensor = sampled_past_ego_states_to_tensor(sampled_past_ego_states)

        past_time_stamps = list(
            self.scenario.get_past_timestamps(
                iteration=0, num_samples=self.num_past_poses, time_horizon=self.past_time_horizon
            )
        ) + [self.scenario.start_time]

        past_time_stamps_tensor = sampled_past_timestamps_to_tensor(past_time_stamps)

        return past_ego_states_tensor, past_time_stamps_tensor
    
    def get_neighbor_agents(self):
        present_tracked_objects = self.scenario.initial_tracked_objects.tracked_objects
        past_tracked_objects = [
            tracked_objects.tracked_objects
            for tracked_objects in self.scenario.get_past_tracked_objects(
                iteration=0, time_horizon=self.past_time_horizon, num_samples=self.num_past_poses
            )
        ]

        sampled_past_observations = past_tracked_objects + [present_tracked_objects]
        past_tracked_objects_tensor_list, past_tracked_objects_types = \
              sampled_tracked_objects_to_tensor_list(sampled_past_observations)

        return past_tracked_objects_tensor_list, past_tracked_objects_types

    def get_map(self):        
        ego_state = self.scenario.initial_ego_state
        ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)
        route_roadblock_ids = self.scenario.get_route_roadblock_ids()
        traffic_light_data = self.scenario.get_traffic_light_status_at_iteration(0)

        coords, traffic_light_data = get_neighbor_vector_set_map(
            self.map_api, self._map_features, ego_coords, self._radius, route_roadblock_ids, traffic_light_data
        )

        vector_map = map_process(ego_state.rear_axle, coords, traffic_light_data, self._map_features, 
                                 self._max_elements, self._max_points, self._interpolation_method)

        return vector_map

    def get_ego_agent_future(self):
        current_absolute_state = self.scenario.initial_ego_state

        trajectory_absolute_states = self.scenario.get_ego_future_trajectory(
            iteration=0, num_samples=self.num_future_poses, time_horizon=self.future_time_horizon
        )

        # Get all future poses of the ego relative to the ego coordinate system
        trajectory_relative_poses = convert_absolute_to_relative_poses(
            current_absolute_state.rear_axle, [state.rear_axle for state in trajectory_absolute_states]
        )

        return trajectory_relative_poses
    
    def get_neighbor_agents_future(self, agent_index):
        current_ego_state = self.scenario.initial_ego_state
        present_tracked_objects = self.scenario.initial_tracked_objects.tracked_objects

        # Get all future poses of of other agents
        future_tracked_objects = [
            tracked_objects.tracked_objects
            for tracked_objects in self.scenario.get_future_tracked_objects(
                iteration=0, time_horizon=self.future_time_horizon, num_samples=self.num_future_poses
            )
        ]

        sampled_future_observations = [present_tracked_objects] + future_tracked_objects
        future_tracked_objects_tensor_list, _ = sampled_tracked_objects_to_tensor_list(sampled_future_observations)
        agent_futures = agent_future_process(current_ego_state, future_tracked_objects_tensor_list, self.num_agents, agent_index)

        return agent_futures
    
    def plot_scenario(self, data):
        # Create map layers
        create_map_raster(data['lanes'], data['crosswalks'], data['route_lanes'])

        # Create agent layers
        create_ego_raster(data['ego_agent_past'][-1])
        create_agents_raster(data['neighbor_agents_past'][:, -1])

        # Draw past and future trajectories
        draw_trajectory(data['ego_agent_past'], data['neighbor_agents_past'])
        draw_trajectory(data['ego_agent_future'], data['neighbor_agents_future'])

        plt.gca().set_aspect('equal')
        plt.tight_layout()
        # plt.show()
        # save plot
        plt.savefig(f"figures/{data['map_name']}_{data['token']}.png")

    def save_to_disk(self, dir, data):
        np.savez(f"{dir}/{data['map_name']}_{data['token']}.npz", **data)

    def _initialize_route_plan(self):
        route_roadblock_ids=self.scenario._route_roadblock_ids
        self._route_roadblocks = []

        for id_ in route_roadblock_ids:
            block = self.map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK)
            block = block or self.map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK_CONNECTOR)
            self._route_roadblocks.append(block)

        self._candidate_lane_edge_ids = [
            edge.id for block in self._route_roadblocks if block for edge in block.interior_edges
        ]

    def _get_path_proposals(self):
        '''
        get all possible path proposals (modified from _get_reference_path)
        '''
        solution_exists = True # assume to be true

        simulation_history_buffer_duration = 2
        buffer_size = int(simulation_history_buffer_duration / self.scenario.database_interval + 1)
        # maybe buffer_size = 1 is enough

        history = SimulationHistoryBuffer.initialize_from_scenario(
                buffer_size=buffer_size, 
                scenario=self.scenario, 
                observation_type=DetectionsTracks) 

        ego_state, observation = history.current_state
        traffic_light_data = self.scenario.get_traffic_light_status_at_iteration(0)

        # Get starting block
        starting_block = None
        min_target_speed = 3
        max_target_speed = 15
        cur_point = (ego_state.rear_axle.x, ego_state.rear_axle.y)
        closest_distance = math.inf

        # get _route_roadblocks
        self._initialize_route_plan()
        self._route_roadblocks # should be ready
        # initialize lattice planner
        self._path_planner = LatticePlanner(self._candidate_lane_edge_ids, self._max_path_length)

        for block in self._route_roadblocks:
            for edge in block.interior_edges:
                distance = edge.polygon.distance(Point(cur_point))
                if distance < closest_distance:
                    starting_block = block
                    closest_distance = distance

            if np.isclose(closest_distance, 0):
                break
            
        # In case the ego vehicle is not on the route, return None
        if closest_distance > 5:
            solution_exists = False

        # ref_paths = self._path_planner.plan_path_proposals(ego_state, starting_block, observation)

        # Get reference path, handle exception
        try:
            ref_paths = self._path_planner.plan_path_proposals(ego_state, starting_block, observation)
        except:
            solution_exists = False

        if not solution_exists:
            return np.zeros((self.proposals_num, 1200//self.subsampling_ratio, 7))
        
        # ref_path = self.post_process(optimal_path, ego_state)

        annotated_paths = []
        for ref_path_tuple in ref_paths:
            ref_path = self._path_planner.post_process(ref_path_tuple[0], ego_state)
            # Annotate red light to occupancy
            occupancy = np.zeros(shape=(ref_path.shape[0], 1))
            for data in traffic_light_data:
                id_ = str(data.lane_connector_id)
                if data.status == TrafficLightStatusType.RED and id_ in self._candidate_lane_edge_ids:
                    lane_conn = self.map_api.get_map_object(id_, SemanticMapLayer.LANE_CONNECTOR)
                    conn_path = lane_conn.baseline_path.discrete_path
                    conn_path = np.array([[p.x, p.y] for p in conn_path])
                    red_light_lane = transform_to_ego_frame(conn_path, ego_state)
                    occupancy = annotate_occupancy(occupancy, ref_path, red_light_lane)

            # Annotate max speed along the reference path
            target_speed = starting_block.interior_edges[0].speed_limit_mps or self._target_speed
            target_speed = np.clip(target_speed, min_target_speed, max_target_speed)
            max_speed = annotate_speed(ref_path, target_speed)

            # Finalize reference path
            ref_path = np.concatenate([ref_path, max_speed, occupancy], axis=-1) # [x, y, theta, k, v_max, occupancy]
            if len(ref_path) < MAX_LEN * 10:
                ref_path = np.append(ref_path, np.repeat(ref_path[np.newaxis, -1], MAX_LEN*10-len(ref_path), axis=0), axis=0)

            annotated_paths.append(ref_path.astype(np.float32))


        stacked_paths = np.stack(annotated_paths[:self.proposals_num], axis=0) # [num_paths, 1200, 6]
        # TODO subsample the path proposals
        stacked_paths = stacked_paths[:, ::self.subsampling_ratio, :] # [num_paths, 120, 6]
        # extend feature dimension as valid mask
        stacked_paths = np.concatenate([stacked_paths, np.ones((stacked_paths.shape[0], stacked_paths.shape[1], 1))], axis=-1)
        # pad to proposals number
        if stacked_paths.shape[0] < self.proposals_num:
            stacked_paths = np.concatenate([stacked_paths, np.zeros((self.proposals_num-stacked_paths.shape[0], stacked_paths.shape[1], stacked_paths.shape[2]))], axis=0)
        
        return stacked_paths

    def work(self, save_dir, debug=False):
        for scenario in tqdm(self._scenarios[:10]):
            map_name = scenario._map_name
            token = scenario.token
            self.scenario: NuPlanScenario = scenario
            self.map_api = scenario.map_api        

            # get agent past tracks
            ego_agent_past, time_stamps_past = self.get_ego_agent()
            neighbor_agents_past, neighbor_agents_types = self.get_neighbor_agents()
            ego_agent_past, neighbor_agents_past, neighbor_indices = \
                agent_past_process(ego_agent_past, time_stamps_past, neighbor_agents_past, neighbor_agents_types, self.num_agents)

            # get vector set map
            vector_map = self.get_map()

            # get agent future tracks
            ego_agent_future = self.get_ego_agent_future()
            neighbor_agents_future = self.get_neighbor_agents_future(neighbor_indices)

            # ------------------------------------------------------------------
            proposals = self._get_path_proposals() ## [num_paths, 120, 6] array, or None

            # gather data
            data = {"map_name": map_name, "token": token, "ego_agent_past": ego_agent_past, "ego_agent_future": ego_agent_future,
                    "neighbor_agents_past": neighbor_agents_past, "neighbor_agents_future": neighbor_agents_future}
            data.update(vector_map)
            data.update({"path_proposals": proposals})

            # data["path_proposals"] = proposals # [12, 120, 7]

            # visualization
            if debug:
                self.plot_scenario(data)

            pass
            # save to disk
            self.save_to_disk(save_dir, data)

    def work_save_scenario(self, save_dir, debug=False):
        for scenario in tqdm(self._scenarios):
            pickle.dump(scenario, open(f"{save_dir}/{scenario._map_name}_{scenario.token}.pkl", "wb"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data Processing')
    parser.add_argument('--data_path', type=str, help='path to raw data')
    parser.add_argument('--map_path', type=str, help='path to map data')
    parser.add_argument('--save_path', type=str, help='path to save processed data')
    parser.add_argument('--scenarios_per_type', type=int, default=1000, help='number of scenarios per type')
    parser.add_argument('--total_scenarios', default=None, help='limit total number of scenarios')
    parser.add_argument('--shuffle_scenarios', type=bool, default=False, help='shuffle scenarios')
    parser.add_argument('--debug', action="store_true", help='if visualize the data output', default=False)
    args = parser.parse_args()

    # create save folder
    os.makedirs(args.save_path, exist_ok=True)
 
    # get scenarios
    map_version = "nuplan-maps-v1.0"    
    sensor_root = None
    db_files = None
    scenario_mapping = ScenarioMapping(scenario_map=get_scenario_map(), subsample_ratio_override=0.5)
    builder = NuPlanScenarioBuilder(args.data_path, args.map_path, sensor_root, db_files, map_version, scenario_mapping=scenario_mapping)
    
    #------------------------------------------------------------------
    # Modified code
    # Instead of using the filter given with GameFormer, use train split "train_140k" to be consistent with the results of other models
    
    with initialize(config_path="./configs/scenario_filter"):
    # # Load the configuration file
        cfg = compose(config_name="train150k_split")
    scenario_filter = hydra.utils.instantiate(cfg)

    
    #------------------------------------------------------------------
    
    
    scenario_filter = ScenarioFilter(*get_filter_parameters(args.scenarios_per_type, args.total_scenarios, args.shuffle_scenarios))
    worker = SingleMachineParallelExecutor(use_process_pool=True)
    scenarios = builder.get_scenarios(scenario_filter, worker)
    print(f"Total number of scenarios: {len(scenarios)}")
    
    # process data
    del worker, builder, scenario_filter, scenario_mapping
    processor = DataProcessor(scenarios)

    # processor.work(args.save_path, debug=args.debug)
    processor.work(args.save_path, debug=True)
    # processor.work_save_scenario(args.save_path, debug=args.debug)



# !! NOTE this script only saves scenario objects now 
'''
python data_process.py \
--data_path $NUPLAN_DATA_ROOT/nuplan-v1.1/splits/trainval \
--map_path $NUPLAN_MAPS_ROOT \
--save_path $NUPLAN_EXP_ROOT/GameFormer/processed_data_proposals_test


'''
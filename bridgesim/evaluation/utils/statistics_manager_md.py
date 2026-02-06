#!/usr/bin/env python
#
# This work is inspired by the CARLA AD Leaderboard statistics_manager.py
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module contains an offline statistics manager for MetaDrive scenarios,
mimicking the output format of the CARLA AD Leaderboard.
"""

import json
import math
from pathlib import Path
from datetime import datetime
from metadrive.constants import TerminationState

# --- Constants from CARLA statistics_manager.py ---
#

# Mapping of MetaDrive infraction attributes to CARLA Leaderboard JSON keys
# We omit STOP_INFRACTION, VEHICLE_BLOCKED, and YIELD_TO_EMERGENCY_VEHICLE
# as they are not directly available in BaseVehicleState
INFRACTION_MAPPING = {
    'crash_building': 'collisions_layout',
    'crash_sidewalk': 'collisions_layout',
    'crash_human': 'collisions_pedestrian',
    'crash_vehicle': 'collisions_vehicle',
    'crash_object': 'collisions_static',
    'red_light': 'red_light',
    'on_white_continuous_line': 'outside_route_lanes',
    'on_yellow_continuous_line': 'outside_route_lanes',
}

# Penalties to apply for each infraction.
# Values are taken directly from PENALTY_VALUE_DICT in statistics_manager.py
PENALTY_VALUES = {
    'collisions_layout': 0.65,       # Mapped from COLLISION_STATIC
    'collisions_pedestrian': 0.5,
    'collisions_vehicle': 0.6,
    'collisions_static': 0.65,       # Mapped from COLLISION_STATIC
    'red_light': 0.7,                # Mapped from TRAFFIC_LIGHT_INFRACTION
    'outside_route_lanes': 0.8,      # Using STOP_INFRACTION value as a stand-in
                                     # since the original is percentage-based.
    'route_timeout': 0.7             # Mapped from SCENARIO_TIMEOUT
}

# Other constants for JSON formatting
ENTRY_STATUS_VALUES = ['Started', 'Finished', 'Rejected', 'Crashed', 'Invalid']
ELIGIBLE_VALUES = {'Started': False, 'Finished': True, 'Rejected': False, 'Crashed': False, 'Invalid': False}
ROUND_DIGITS = 3
ROUND_DIGITS_SCORE = 6

class OfflineStatisticsManager(object):
    """
    Gathers data at runtime and saves it in a format compatible with the
    CARLA AD Leaderboard's simulation_results.json.
    """

    def __init__(self, output_path: Path, scenario_name: str, scenario_data: dict):
        self.output_path = output_path
        self.scenario_name = scenario_name
        self.save_path = self.output_path / "evaluation_results.json"
        
        # Get scenario metadata
        sdc_id = scenario_data['metadata']['sdc_id']
        self.total_steps = len(scenario_data['tracks'][sdc_id]['state']['position'])
        self.track_length = scenario_data['metadata'].get('track_length', 0.0)
        
        # Create a unique save name, similar to CARLA's leaderboard_evaluator.py
        currentDateAndTime = datetime.now()
        currentTime = currentDateAndTime.strftime("%m_%d_%H_%M_%S")
        save_name = f"{scenario_name}_{currentTime}"

        # Initialize the route record
        self.record = self._create_route_record(scenario_name, save_name)
        self.infractions_logged = set() # To prevent double-counting
        
        self.finalized = False

        self.prev_flags = {key: False for key in INFRACTION_MAPPING.keys()}
        
        # NEW: Accumulators for continuous infractions
        self.off_road_distance = 0.0
        self.last_position = None

    def _create_route_record(self, route_name, save_name):
        """Creates the basic route data structure."""
        #
        record = {}
        record['index'] = 0
        record['route_id'] = route_name
        record['scenario_name'] = route_name
        record['weather_id'] = 0 # Placeholder
        record['save_name'] = save_name
        record['status'] = 'Started'
        record['entry_status'] = 'Started'
        record['num_infractions'] = 0
        
        # Initialize infraction lists
        record['infractions'] = {
            'collisions_layout': [],
            'collisions_pedestrian': [],
            'collisions_vehicle': [],
            'collisions_static': [],
            'red_light': [],
            'stop_infraction': [], # Omitted, but key kept for format compatibility
            'outside_route_lanes': [],
            'min_speed_infractions': [], # Omitted
            'yield_emergency_vehicle_infractions': [], # Omitted
            'scenario_timeouts': [], # Omitted
            'route_dev': [], # Omitted
            'vehicle_blocked': [], # Omitted
            'route_timeout': []
        }

        record['scores'] = {
            'score_route': 0,
            'score_penalty': 1.0, # Start at 1.0
            'score_composed': 0
        }

        record['meta'] = {
            'route_length': self.track_length,
            'duration_game': 0,
            'duration_system': 0, # Not applicable in offline eval
        }
        
        return record

    def update_per_step(self, agent, info, frame_id):
        if self.finalized:
            return

        current_pos = agent.position # MetaDrive (x, y) array
        
        # Calculate distance moved this step (for off-road accumulation)
        step_distance = 0.0
        if self.last_position is not None:
            step_distance = math.dist(current_pos, self.last_position)
        self.last_position = current_pos

        # Check infractions
        for attr, key in INFRACTION_MAPPING.items():
            is_active = getattr(agent, attr, False)
            
            # --- HANDLE OFF-ROAD (Continuous Integration) ---
            if key == 'outside_route_lanes':
                if is_active:
                    # Do NOT penalize yet. Just accumulate distance.
                    self.off_road_distance += step_distance
                    
                    # Optional: Log a message only on the *first* frame it happens
                    if not self.prev_flags[attr]:
                         self.record['infractions'][key].append(
                             f"Agent went off-road at frame {frame_id}"
                         )

            # --- HANDLE COLLISIONS (Edge Detection) ---
            elif 'collisions' in key:
                # Only register if it transitioned from False -> True (Rising Edge)
                if is_active and not self.prev_flags[attr]:
                    message = f"Infraction '{key}' triggered at frame {frame_id}"
                    self.record['infractions'][key].append(message)
                    
                    # Apply penalty ONCE per collision event
                    penalty = PENALTY_VALUES.get(key, 1.0)
                    self.record['scores']['score_penalty'] *= penalty

            # --- HANDLE RED LIGHT (Edge Detection) ---
            elif key == 'red_light':
                # Similar to collisions, usually counts once per violation event
                if is_active and not self.prev_flags[attr]:
                    self.record['infractions'][key].append(f"Red light violation at frame {frame_id}")
                    self.record['scores']['score_penalty'] *= PENALTY_VALUES.get(key, 1.0)

            # Update previous flag state
            self.prev_flags[attr] = is_active
            
        # Update total infraction count (based on list length)
        self.record['num_infractions'] = sum(
            len(v) for v in self.record['infractions'].values()
        )

    def finalize_route(self, agent, info, terminated, truncated, route_completion_override=None):
        """
        Calculates final scores and status at the end of the episode.
        'info' is the step info from ScenarioEnv.step().

        Args:
            agent: The ego agent
            info: Step info dict from ScenarioEnv.step()
            terminated: Whether episode terminated
            truncated: Whether episode was truncated
            route_completion_override: Pre-computed route completion value (if None, uses info['route_completion'])
        """
        if self.finalized:
            return

        # 1. Get Route Completion Score
        if route_completion_override is not None:
            route_completion = route_completion_override
        else:
            # Fallback to raw route_completion from info
            route_completion = info.get('route_completion', 0.0)

        self.record['scores']['score_route'] = round(
            min(route_completion * 100, 100.0), ROUND_DIGITS_SCORE
        )

        # 2. Get Metadata
        # Assuming 10Hz simulation step (0.1s)
        self.record['meta']['duration_game'] = round(
            agent.engine.episode_step * 0.1, ROUND_DIGITS
        )

        # 3. Apply Off-Road Penalty (Percentage Based)
        # This matches CARLA's "increases" penalty type logic where value=0.
        # Formula: score *= (1 - percentage_off_road / 100)
        if self.track_length > 0:
            percentage_off_road = (self.off_road_distance / self.track_length) * 100
        else:
            percentage_off_road = 0.0

        if percentage_off_road > 0:
            # Format the message exactly like CARLA
            msg = (f"Agent went outside the route lanes for about {self.off_road_distance:.3f} meters "
                   f"({percentage_off_road:.2f}% of the completed route)")
            
            # If we created a placeholder in update_per_step, replace it. 
            # Otherwise, append the new detailed message.
            if self.record['infractions']['outside_route_lanes']:
                self.record['infractions']['outside_route_lanes'][0] = msg
            else:
                self.record['infractions']['outside_route_lanes'].append(msg)

            # Apply the penalty multiplier
            # If you drove 5% off-road, the multiplier is 0.95
            off_road_multiplier = max(0.0, (1 - (percentage_off_road / 100.0)))
            self.record['scores']['score_penalty'] *= off_road_multiplier

        # 4. Determine final status
        if info.get(TerminationState.SUCCESS):
            self.record['status'] = 'Perfect' if self.record['num_infractions'] == 0 else 'Completed'
            self.record['entry_status'] = 'Finished'
        
        elif truncated and not info.get(TerminationState.SUCCESS):
            # Scenario truncated (max_step reached)
            self.record['status'] = 'Failed - Route timeout'
            self.record['entry_status'] = 'Finished' # Still "Finished" as it didn't crash
            
            # Only add timeout infraction if not already present
            if not self.record['infractions']['route_timeout']:
                self.record['infractions']['route_timeout'].append(
                    f"Scenario timed out at step {agent.engine.episode_step}"
                )
            
            # Apply timeout penalty
            self.record['scores']['score_penalty'] *= PENALTY_VALUES.get('route_timeout', 0.7)

        elif terminated:
            # Scenario terminated due to a crash
            reason = "Unknown"
            if info.get(TerminationState.CRASH_VEHICLE): reason = "Crash Vehicle"
            elif info.get(TerminationState.CRASH_OBJECT): reason = "Crash Object"
            elif info.get(TerminationState.CRASH_HUMAN): reason = "Crash Human"
            elif info.get(TerminationState.CRASH_BUILDING): reason = "Crash Building"
            elif info.get(TerminationState.CRASH_SIDEWALK): reason = "Crash Sidewalk"
            elif info.get(TerminationState.OUT_OF_ROAD): reason = "Out of Road"
            
            self.record['status'] = f'Failed - {reason}'
            self.record['entry_status'] = 'Crashed'
        
        else:
            # Fallback (e.g., error or manual stop)
            self.record['status'] = 'Failed - Unknown'
            self.record['entry_status'] = 'Invalid'

        # 5. Calculate final composed score
        # Round the penalty score before composition
        self.record['scores']['score_penalty'] = round(
            self.record['scores']['score_penalty'], ROUND_DIGITS_SCORE
        )
        
        # Composed score = Route Score * Penalty Multiplier
        self.record['scores']['score_composed'] = round(
            max(self.record['scores']['score_route'] * self.record['scores']['score_penalty'], 0.0), 
            ROUND_DIGITS_SCORE
        )
        
        self.finalized = True

    def get_save_path(self):
        return self.save_path

    def save_to_json(self):
        """
        Writes the results into the endpoint, mimicking the full JSON structure
        from statistics_manager.py.
        """
        if not self.finalized:
            print("Warning: Saving statistics before route is finalized. Results may be incomplete.")
        
        results = {}
        results['entry_status'] = self.record['entry_status']
        results['eligible'] = ELIGIBLE_VALUES.get(results['entry_status'], False)
        results['sensors'] = [] # Placeholder

        # Create global record (for a single run)
        global_record = {}
        global_record['index'] = 0
        global_record['route_id'] = -1
        global_record['status'] = self.record['status']
        global_record['infractions'] = {
            k: len(v) for k, v in self.record['infractions'].items()
        }
        global_record['scores_mean'] = self.record['scores']
        global_record['scores_std_dev'] = {k: 0 for k in self.record['scores']}
        global_record['meta'] = self.record['meta']
        global_record['meta']['exceptions'] = []
        if 'Failed' in self.record['status']:
            global_record['meta']['exceptions'].append(
                (self.record['route_id'], 0, self.record['status'])
            )

        # Create checkpoint structure
        checkpoint = {}
        checkpoint['global_record'] = global_record
        checkpoint['progress'] = [1, 1] # 1 of 1 routes
        checkpoint['records'] = [self.record]
        results['_checkpoint'] = checkpoint

        # Add top-level values and labels
        km_driven = (self.record['meta']['route_length'] / 1000) * (self.record['scores']['score_route'] / 100)
        km_driven = max(km_driven, 0.001)

        results['values'] = [
            str(global_record['scores_mean']['score_composed']),
            str(global_record['scores_mean']['score_route']),
            str(global_record['scores_mean']['score_penalty']),
            str(round(global_record['infractions']['collisions_pedestrian'] / km_driven, ROUND_DIGITS)),
            str(round(global_record['infractions']['collisions_vehicle'] / km_driven, ROUND_DIGITS)),
            str(round(global_record['infractions']['collisions_layout'] / km_driven, ROUND_DIGITS)),
            str(round(global_record['infractions']['red_light'] / km_driven, ROUND_DIGITS)),
            str(round(global_record['infractions']['stop_infraction'] / km_driven, ROUND_DIGITS)),
            str(round(global_record['infractions']['outside_route_lanes'] / km_driven, ROUND_DIGITS)),
            str(round(global_record['infractions']['route_dev'] / km_driven, ROUND_DIGITS)),
            str(round(global_record['infractions']['route_timeout'] / km_driven, ROUND_DIGITS)),
            str(round(global_record['infractions']['vehicle_blocked'] / km_driven, ROUND_DIGITS)),
            str(round(global_record['infractions']['yield_emergency_vehicle_infractions'] / km_driven, ROUND_DIGITS)),
            str(round(global_record['infractions']['scenario_timeouts'] / km_driven, ROUND_DIGITS)),
            str(round(global_record['infractions']['min_speed_infractions'] / km_driven, ROUND_DIGITS)),
        ]
        results['labels'] = [
            "Avg. driving score", "Avg. route completion", "Avg. infraction penalty",
            "Collisions with pedestrians", "Collisions with vehicles", "Collisions with layout",
            "Red lights infractions", "Stop sign infractions", "Off-road infractions",
            "Route deviations", "Route timeouts", "Agent blocked",
            "Yield emergency vehicles infractions", "Scenario timeouts", "Min speed infractions"
        ]

        # Save the file
        try:
            with open(self.save_path, 'w') as f:
                json.dump(results, f, indent=4)
        except Exception as e:
            print(f"Error saving statistics to JSON: {e}")

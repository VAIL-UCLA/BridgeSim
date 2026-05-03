from collections import deque
import numpy as np

class PID(object):
	def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, n=20):
		self._K_P = K_P
		self._K_I = K_I
		self._K_D = K_D

		self._window = deque([0 for _ in range(n)], maxlen=n)
		self._max = 0.0
		self._min = 0.0

	def step(self, error):
		self._window.append(error)
		self._max = max(self._max, abs(error))
		self._min = -abs(self._max)

		if len(self._window) >= 2:
			integral = np.mean(self._window)
			derivative = (self._window[-1] - self._window[-2])
		else:
			integral = 0.0
			derivative = 0.0

		return self._K_P * error + self._K_I * integral + self._K_D * derivative



class PIDController(object):
    
    def __init__(self, turn_KP=3, turn_KI=0.0, turn_KD=0.0, turn_n=10, speed_KP=5.0, speed_KI=0.0,speed_KD=0.0, speed_n = 10, max_throttle=1.0, brake_speed=0.5, brake_ratio=2.0, clip_delta=1.0, aim_dist=4.0, angle_thresh=0.3, dist_thresh=10):

        self.turn_controller = PID(K_P=turn_KP, K_I=turn_KI, K_D=turn_KD, n=turn_n)
        self.speed_controller = PID(K_P=speed_KP, K_I=speed_KI, K_D=speed_KD, n=speed_n)
        # Values from MetaDrive PID controller
        # self.turn_controller = PID(K_P=1.7, K_I=0.01, K_D=3.5, n=turn_n)
        # self.speed_controller = PID(K_P=0.3, K_I=0.002, K_D=0.05, n=speed_n)
        self.max_throttle = max_throttle
        self.brake_speed = brake_speed
        self.brake_ratio = brake_ratio
        self.clip_delta = clip_delta
        self.aim_dist = aim_dist
        self.angle_thresh = angle_thresh
        self.dist_thresh = dist_thresh

    def control_pid(self, waypoints, speed, target, waypoint_dt=None, target_speed=None):
        ''' Predicts vehicle control with a PID controller.
        Args:
            waypoints (tensor): output of self.plan()
            speed (tensor): speedometer input
            target_speed (float, optional): explicit longitudinal target speed (m/s).
                If provided, overrides waypoint-spacing-derived desired speed.
        '''

        # iterate over vectors between predicted waypoints
        num_pairs = len(waypoints) - 1
        best_norm = 1e5
        desired_speed = 0
        aim = waypoints[0]
        for i in range(num_pairs):
            # magnitude of vectors, used for speed (only when target_speed not provided)
            if target_speed is None:
                desired_speed += np.linalg.norm(
                        waypoints[i+1] - waypoints[i]) * 8.0 / num_pairs

            # norm of vector midpoints, used for steering
            norm = np.linalg.norm((waypoints[i+1] + waypoints[i]) / 2.0)
            if abs(self.aim_dist-best_norm) > abs(self.aim_dist-norm):
                aim = waypoints[i]
                best_norm = norm

        if target_speed is not None:
            desired_speed = float(target_speed)

        aim_last = waypoints[-1] - waypoints[-2]

        angle = np.degrees(np.pi / 2 - np.arctan2(aim[1], aim[0])) / 90
        angle_last = np.degrees(np.pi / 2 - np.arctan2(aim_last[1], aim_last[0])) / 90
        angle_target = np.degrees(np.pi / 2 - np.arctan2(target[1], target[0])) / 90

        # choice of point to aim for steering, removing outlier predictions
        # use target point if it has a smaller angle or if error is large
        # predicted point otherwise
        # (reduces noise in eg. straight roads, helps with sudden turn commands)
        use_target_to_aim = np.abs(angle_target) < np.abs(angle)
        use_target_to_aim = use_target_to_aim or (np.abs(angle_target-angle_last) > self.angle_thresh and target[1] < self.dist_thresh)
        if use_target_to_aim:
            angle_final = angle_target
        else:
            angle_final = angle

        steer = self.turn_controller.step(angle_final)
        steer = np.clip(steer, -1.0, 1.0)

        brake = desired_speed < self.brake_speed or (speed / desired_speed) > self.brake_ratio

        delta = np.clip(desired_speed - speed, 0.0, self.clip_delta)
        throttle = self.speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, self.max_throttle)
        throttle = throttle if not brake else 0.0

        metadata = {
            'speed': float(speed),
            'steer': float(steer),
            'throttle': float(throttle),
            'brake': float(brake),
            # 'wp_4': tuple(waypoints[3].astype(np.float64)),
            # 'wp_3': tuple(waypoints[2].astype(np.float64)),
            # 'wp_2': tuple(waypoints[1].astype(np.float64)),
            # 'wp_1': tuple(waypoints[0].astype(np.float64)),
            'aim': tuple(aim.astype(np.float64)),
            'target': tuple(target.astype(np.float64)),
            'desired_speed': float(desired_speed),
            'angle': float(angle.astype(np.float64)),
            'angle_last': float(angle_last.astype(np.float64)),
            'angle_target': float(angle_target.astype(np.float64)),
            'angle_final': float(angle_final.astype(np.float64)),
            'delta': float(delta.astype(np.float64)),
        }

        return steer, throttle, brake, metadata


class PurePursuitController(object):
    """
    Pure Pursuit Controller for trajectory tracking.

    Based on the algorithm from PythonRobotics:
    https://github.com/AtsushiSakai/PythonRobotics/blob/master/PathTracking/pure_pursuit/pure_pursuit.py

    The Pure Pursuit algorithm computes steering based on a look-ahead point
    using the bicycle model kinematics:
        delta = atan2(2 * L * sin(alpha) / Lf, 1.0)

    Where:
        - alpha: angle between vehicle heading and look-ahead point
        - L: wheelbase
        - Lf: look-ahead distance (can be adaptive based on speed)
    """

    def __init__(
        self,
        # Vehicle parameters
        wheelbase: float = 2.85,
        # Look-ahead parameters
        look_ahead_dist: float = 4.0,
        min_look_ahead: float = 2.0,
        look_ahead_gain: float = 0.1,  # k in Lf = k * v + Lfc
        # Steering limits
        max_steer: float = np.radians(40),  # 40 degrees max steering angle
        # Speed control parameters (PID)
        speed_KP: float = 0.9,
        speed_KI: float = 0.1,
        speed_KD: float = 0.01,
        speed_n: int = 10,
        max_throttle: float = 1.0,
        brake_speed: float = 0.5,
        brake_ratio: float = 2.0,
        clip_delta: float = 1.0,
    ):
        """
        Initialize Pure Pursuit Controller.

        Args:
            wheelbase: Vehicle wheelbase [m].
            look_ahead_dist: Base look-ahead distance [m].
            min_look_ahead: Minimum look-ahead distance [m].
            look_ahead_gain: Gain for speed-adaptive look-ahead (Lf = k*v + Lfc).
            max_steer: Maximum steering angle [rad].
            speed_KP/KI/KD: PID gains for speed control.
            speed_n: PID window size for speed control.
            max_throttle: Maximum throttle output.
            brake_speed: Speed threshold for braking.
            brake_ratio: Ratio threshold for braking.
            clip_delta: Maximum speed error for throttle computation.
        """
        # Vehicle parameters
        self.wheelbase = wheelbase

        # Look-ahead parameters
        self.look_ahead_dist = look_ahead_dist  # Lfc (base look-ahead)
        self.min_look_ahead = min_look_ahead
        self.look_ahead_gain = look_ahead_gain  # k

        # Steering limits
        self.max_steer = max_steer

        # Speed control (reuse PID for throttle/brake)
        self.speed_controller = PID(K_P=speed_KP, K_I=speed_KI, K_D=speed_KD, n=speed_n)
        self.max_throttle = max_throttle
        self.brake_speed = brake_speed
        self.brake_ratio = brake_ratio
        self.clip_delta = clip_delta

    def control_pid(self, waypoints, speed, target, waypoint_dt=0.1, target_speed=None):
        """
        Compute control commands using Pure Pursuit for steering.

        This method is compatible with PIDController.control_pid() interface.

        Args:
            waypoints: (N, 2) array of future waypoints in ego frame (Y-forward, X-right).
            speed: Current vehicle speed [m/s].
            target: Target point (typically waypoints[-1]).

        Returns:
            steer: Steering command [-1, 1].
            throttle: Throttle command [0, max_throttle].
            brake: Brake flag (boolean).
            metadata: Dictionary with debug information.
        """
        waypoints = np.asarray(waypoints)
        target = np.asarray(target)

        # 1. Compute adaptive look-ahead distance
        # Lf = k * v + Lfc (speed-adaptive look-ahead)
        Lf = self.look_ahead_gain * speed + self.look_ahead_dist
        Lf = max(Lf, self.min_look_ahead)

        # 2. Find the look-ahead point (closest to Lf distance)
        aim, aim_idx = self._find_look_ahead_point(waypoints, Lf)

        # 3. Compute Pure Pursuit steering
        steer, alpha = self._pure_pursuit_steering(aim, Lf)

        # 4. Compute desired speed from waypoint spacing
        desired_speed = self._compute_desired_speed(waypoints, waypoint_dt=waypoint_dt)

        # 5. Compute throttle/brake using PID (same as PIDController)
        brake = desired_speed < self.brake_speed or (speed / max(desired_speed, 0.01)) > self.brake_ratio

        delta = np.clip(desired_speed - speed, 0.0, self.clip_delta)
        throttle = self.speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, self.max_throttle)
        throttle = throttle if not brake else 0.0

        # Build metadata
        metadata = {
            'speed': float(speed),
            'steer': float(steer),
            'throttle': float(throttle),
            'brake': float(brake),
            'aim': tuple(aim.astype(np.float64)),
            'aim_idx': int(aim_idx),
            'target': tuple(target.astype(np.float64)),
            'desired_speed': float(desired_speed),
            'look_ahead_dist': float(Lf),
            'alpha': float(alpha),
            'delta': float(delta),
        }

        return steer, throttle, brake, metadata

    def _find_look_ahead_point(self, waypoints, Lf):
        """
        Find the waypoint closest to the look-ahead distance.

        Args:
            waypoints: (N, 2) array of waypoints in ego frame.
            Lf: Look-ahead distance [m].

        Returns:
            aim: The look-ahead point (x, y).
            aim_idx: Index of the selected waypoint.
        """
        best_idx = 0
        best_dist_diff = float('inf')

        for i, wp in enumerate(waypoints):
            dist = np.linalg.norm(wp)
            dist_diff = abs(dist - Lf)
            if dist_diff < best_dist_diff:
                best_dist_diff = dist_diff
                best_idx = i

        return waypoints[best_idx], best_idx

    def _pure_pursuit_steering(self, aim, Lf):
        """
        Compute steering angle using Pure Pursuit algorithm.

        In ego frame (Y-forward, X-right):
        - aim[0] is x (lateral offset, positive = right)
        - aim[1] is y (forward distance, positive = forward)

        The angle alpha is the angle from the rear axle to the look-ahead point.
        In ego frame with Y-forward convention:
            alpha = atan2(x, y)

        Steering angle:
            delta = atan2(2 * L * sin(alpha) / Lf, 1.0)

        Args:
            aim: Look-ahead point (x, y) in ego frame.
            Lf: Look-ahead distance [m].

        Returns:
            steer: Normalized steering command [-1, 1].
            alpha: Angle to look-ahead point [rad].
        """
        x = aim[0]  # lateral offset
        y = aim[1]  # forward distance

        # Compute alpha: angle to look-ahead point from ego heading
        # In Y-forward frame, alpha = atan2(x, y)
        alpha = np.arctan2(x, y)

        # Pure Pursuit steering formula
        # delta = atan2(2 * L * sin(alpha) / Lf, 1.0)
        # This simplifies to: delta = atan(2 * L * sin(alpha) / Lf)

        # Actual look-ahead distance to the point
        actual_Lf = np.sqrt(x*x + y*y)
        if actual_Lf < 0.1:
            actual_Lf = Lf  # Use target Lf if point is very close

        # Compute steering angle [rad]
        delta = np.arctan2(2.0 * self.wheelbase * np.sin(alpha), actual_Lf)

        # Clip to max steering angle
        delta = np.clip(delta, -self.max_steer, self.max_steer)

        # Normalize to [-1, 1] for MetaDrive
        # MetaDrive expects steering in [-1, 1] range
        steer = delta / self.max_steer

        return steer, alpha

    def _compute_desired_speed(self, waypoints, waypoint_dt=0.1):
        """
        Compute desired speed from waypoint spacing.

        Uses nearby waypoints (first ~1 second) to estimate immediate desired speed,
        rather than averaging over the entire trajectory.

        Args:
            waypoints: (N, 2) array of waypoints.
            waypoint_dt: Time between waypoints [s]. Default 0.1s (10Hz).

        Returns:
            desired_speed: Estimated desired speed [m/s].
        """
        num_pairs = len(waypoints) - 1
        if num_pairs <= 0:
            return 0.0

        # Use first ~1 second of waypoints (10 points at 10Hz)
        # This gives more responsive speed control
        max_pairs = min(num_pairs, 10)

        total_dist = 0.0
        for i in range(max_pairs):
            total_dist += np.linalg.norm(waypoints[i+1] - waypoints[i])

        # Speed = distance / time
        desired_speed = total_dist / (max_pairs * waypoint_dt)

        return desired_speed

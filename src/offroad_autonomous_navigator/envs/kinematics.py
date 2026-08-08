import math

from offroad_autonomous_navigator.envs.schemas import VehicleAction, VehicleState


class BicycleModel:
    """Kinematic bicycle model for Ackermann-steered vehicles.

    Simplifies the vehicle to two virtual wheels (front, rear) connected
    by a rigid axle of length `wheelbase`, integrated via explicit Euler.
    """
    def __init__(self, wheelbase: float, 
                max_steering_angle: float, 
                max_acceleration: float,
                max_speed: float) -> None:
        self.wheelbase = wheelbase
        self.max_steering_angle = max_steering_angle
        self.max_acceleration = max_acceleration
        self.max_speed = max_speed
    def step(self, state: VehicleState, action: VehicleAction, dt: float) -> VehicleState:
        '''
        Update the vehicle's state based on the current state, action, 
        and time step using the bicycle model equations.
        '''
        # Check dt is positive
        if dt <= 0:
            raise ValueError("Time step dt must be positive.")

        # Clamp the steering angle and acceleration to their respective limits
        steering_angle = max(-self.max_steering_angle, 
                            min(self.max_steering_angle, action.steering_angle))
        acceleration = max(-self.max_acceleration, 
                            min(self.max_acceleration, action.acceleration))

        # Update the vehicle's state using the bicycle model equations
        v_new = state.v + acceleration * dt
        v_clamped = max(-self.max_speed, min(self.max_speed, v_new))

        theta_new = state.theta + (v_clamped / self.wheelbase) * math.tan(steering_angle) * dt
        theta_clamped = (theta_new + math.pi) % (2 * math.pi) - math.pi

        x_new = state.x + v_clamped * math.cos(theta_clamped) * dt
        y_new = state.y + v_clamped * math.sin(theta_clamped) * dt



        return VehicleState(x=x_new, y=y_new, theta=theta_clamped, v=v_clamped)
# ============================================================================
# MOTOR CONTROLLER
# ============================================================================
from typing import Tuple
import lgpio
from .configuration import RobotConfig
import time
import math

class MotorController:
    """Low-level motor control using lgpio"""
    def __init__(self, cfg: RobotConfig):
        self.cfg = cfg
        self.handle = lgpio.gpiochip_open(0)
        # Initialize all pins
        for pin in cfg.motor_left_pins + cfg.motor_right_pins:
            lgpio.gpio_claim_output(self.handle, pin)
        # Start PWM on enable pins
        lgpio.tx_pwm(self.handle, cfg.motor_left_pins[2], cfg.pwm_frequency, 0)
        lgpio.tx_pwm(self.handle, cfg.motor_right_pins[2], cfg.pwm_frequency, 0)
        print("✓ Motor controller initialized")

    def set_motor(self, pins: Tuple[int, int, int], direction: int, speed: float):
        """
        Set motor direction and speed.
        Args:
            pins: (IN1, IN2, EN) pin tuple
            direction: 1 (forward), -1 (backward), 0 (stop)
            speed: 0-100 (PWM duty cycle percentage)
        """
        in1, in2, en = pins
        speed = max(0, min(100, speed))  # Clamp to 0-100
        if direction == 1:  # Forward
            lgpio.gpio_write(self.handle, in1, 1)
            lgpio.gpio_write(self.handle, in2, 0)
            lgpio.tx_pwm(self.handle, en, self.cfg.pwm_frequency, int(speed))
        elif direction == -1:  # Backward
            lgpio.gpio_write(self.handle, in1, 0)
            lgpio.gpio_write(self.handle, in2, 1)
            lgpio.tx_pwm(self.handle, en, self.cfg.pwm_frequency, int(speed))
        else:  # Stop
            lgpio.gpio_write(self.handle, in1, 0)
            lgpio.gpio_write(self.handle, in2, 0)
            lgpio.tx_pwm(self.handle, en, self.cfg.pwm_frequency, 0)
    def set_motor_speed(self, speed,heading):
      print(heading, speed)
      angle = 180 * heading / math.pi
      angle = max(-90, min(90, angle))
      time_delay = math.fabs(angle)/30
      left_dir = -1 if angle < 0 else 1
      right_dir = 1 if angle < 0 else -1
      if angle < 15:
        self.set_motor(self.cfg.motor_left_pins, -1 if angle < 0 else 1, speed)
        self.set_motor(self.cfg.motor_right_pins,  -1 if angle < 0 else 1, speed)
        return
      self.set_motor(self.cfg.motor_left_pins, left_dir, 60 if left_dir == 1 else 90)
      self.set_motor(self.cfg.motor_right_pins, right_dir, 60 if right_dir == 1 else 90)
      time.sleep(time_delay+0.2)

    def set_wheel_speeds(self, left_speed: float, right_speed: float):
        """
        Set differential drive wheel speeds.

        Args:
            left_speed: -1.0 to 1.0 (negative = backward)
            right_speed: -1.0 to 1.0 (negative = backward)
        """
        # Convert to direction and PWM
        left_dir = 1 if left_speed > 0 else (-1 if left_speed < 0 else 0)
        right_dir = 1 if right_speed > 0 else (-1 if right_speed < 0 else 0)

        left_pwm = abs(left_speed) * 100
        right_pwm = abs(right_speed) * 100

        self.set_motor(self.cfg.motor_left_pins, left_dir, left_pwm)
        self.set_motor(self.cfg.motor_right_pins, right_dir, right_pwm)

    def stop(self):
        """Emergency stop"""
        self.set_motor(self.cfg.motor_left_pins, 0, 0)
        self.set_motor(self.cfg.motor_right_pins, 0, 0)

    def cleanup(self):
        """Cleanup GPIO"""
        self.stop()
        lgpio.gpiochip_close(self.handle)
        print("Motor controller cleaned up")

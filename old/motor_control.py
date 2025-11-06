# motor_control.py
import lgpio
from config import WHEEL_RADIUS, WHEEL_BASE, MAX_SPEED, MAX_ANGULAR_SPEED

class MotorController:
    def __init__(self, left_pins=(16, 25, 12), right_pins=(5, 26, 13)):
        """Initialize L293D motor driver with lgpio."""
        self.handle = lgpio.gpiochip_open(0)
        self.left_pins = left_pins
        self.right_pins = right_pins
        for pin in left_pins + right_pins:
            lgpio.gpio_claim_output(self.handle, pin)
        lgpio.tx_pwm(self.handle, left_pins[2], 100, 0)
        lgpio.tx_pwm(self.handle, right_pins[2], 100, 0)
        self.left_pwm_pin = left_pins[2]
        self.right_pwm_pin = right_pins[2]

    def set_velocity(self, v, w):
        """Set linear and angular velocity (v in m/s, w in rad/s)."""
        vl = v - (w * WHEEL_BASE / 2)
        vr = v + (w * WHEEL_BASE / 2)
        max_wheel_speed = MAX_SPEED + (MAX_ANGULAR_SPEED * WHEEL_BASE / 2)
        left_duty = max(min((vl / max_wheel_speed) * 100, 100), -100)
        right_duty = max(min((vr / max_wheel_speed) * 100, 100), -100)

        if left_duty >= 0:
            lgpio.gpio_write(self.handle, self.left_pins[0], 1)
            lgpio.gpio_write(self.handle, self.left_pins[1], 0)
            lgpio.tx_pwm(self.handle, self.left_pwm_pin, 100, left_duty)
        else:
            lgpio.gpio_write(self.handle, self.left_pins[0], 0)
            lgpio.gpio_write(self.handle, self.left_pins[1], 1)
            lgpio.tx_pwm(self.handle, self.left_pwm_pin, 100, -left_duty)

        if right_duty >= 0:
            lgpio.gpio_write(self.handle, self.right_pins[0], 1)
            lgpio.gpio_write(self.handle, self.right_pins[1], 0)
            lgpio.tx_pwm(self.handle, self.right_pwm_pin, 100, right_duty)
        else:
            lgpio.gpio_write(self.handle, self.right_pins[0], 0)
            lgpio.gpio_write(self.handle, self.right_pins[1], 1)
            lgpio.tx_pwm(self.handle, self.right_pwm_pin, 100, -right_duty)

    def stop(self):
        """Stop both motors."""
        lgpio.tx_pwm(self.handle, self.left_pwm_pin, 100, 0)
        lgpio.tx_pwm(self.handle, self.right_pwm_pin, 100, 0)
        lgpio.gpiochip_close(self.handle)

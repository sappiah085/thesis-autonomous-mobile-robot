# test_motors.py
import lgpio
import time
import math

# GPIO pin assignments
MOTOR_LEFT_PINS = (16, 25, 12)  # IN1, IN2, EN1 (PWM) - right motor
MOTOR_RIGHT_PINS = (5, 26, 13)  # IN3, IN4, EN2 (PWM) - left motor


def setup_motors():
    """Initialize GPIO pins for L293D motor control."""
    handle = lgpio.gpiochip_open(0)

    # Set all pins as outputs
    for pin in MOTOR_LEFT_PINS + MOTOR_RIGHT_PINS:
        lgpio.gpio_claim_output(handle, pin)

    # Initialize PWM on enable pins (1000 Hz for smoother operation)
    lgpio.tx_pwm(handle, MOTOR_LEFT_PINS[2], 1000, 0)  # 1000 Hz, 0% duty
    lgpio.tx_pwm(handle, MOTOR_RIGHT_PINS[2], 1000, 0)  # 1000 Hz, 0% duty

    return handle


def set_motor(handle, motor_pins, direction, duty_cycle):
    """Set motor direction and speed.

    Args:
        direction: 1 (forward), -1 (backward), 0 (stop)
        duty_cycle: 0 to 100 (PWM percentage)
    """
    in1, in2, en = motor_pins

    if direction == 1:  # Forward
        lgpio.gpio_write(handle, in1, 1)
        lgpio.gpio_write(handle, in2, 0)
        lgpio.tx_pwm(handle, en, 100, duty_cycle)
    elif direction == -1:  # Backward
        lgpio.gpio_write(handle, in1, 0)
        lgpio.gpio_write(handle, in2, 1)
        lgpio.tx_pwm(handle, en, 100, duty_cycle)
    else:  # Stop
        lgpio.gpio_write(handle, in1, 0)
        lgpio.gpio_write(handle, in2, 0)
        lgpio.tx_pwm(handle, en, 100, 0)


def stop(handle):
    """Stop both motors."""
    set_motor(handle, MOTOR_LEFT_PINS, 0, 0)
    set_motor(handle, MOTOR_RIGHT_PINS, 0, 0)


def forward(handle, speed):
    """Move forward at specified speed (0-100)."""
    set_motor(handle, MOTOR_LEFT_PINS, 1, speed)
    set_motor(handle, MOTOR_RIGHT_PINS, 1, speed)


def backward(handle, speed):
    """Move backward at specified speed (0-100)."""
    set_motor(handle, MOTOR_LEFT_PINS, -1, speed)
    set_motor(handle, MOTOR_RIGHT_PINS, -1, speed)


def turn_angle(handle, angle):
    angle = max(-90, min(90, angle))
    time_delay = math.fabs(angle)/30
    left_dir = -1 if angle < 0 else 1
    right_dir = 1 if angle < 0 else -1
    set_motor(handle,MOTOR_RIGHT_PINS, right_dir, 60 if right_dir == 1 else 90)
    set_motor(handle,MOTOR_LEFT_PINS, left_dir, 60 if left_dir == 1 else 90)
    time.sleep(time_delay+0.2)

def test_motors():
    DESIRED_SPEED = 20
    handle = setup_motors()
    print("Starting motor test...")
    try:
        forward(handle, DESIRED_SPEED)
        time.sleep(1)
        # turn_angle(handle, -25)
        # forward(handle, DESIRED_SPEED)
        # time.sleep(1)
        print("Stop")
        stop(handle)
        print("\nTest completed successfully!")
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    finally:
        print("Cleaning up...")
        set_motor(handle, MOTOR_LEFT_PINS, 0, 0)
        set_motor(handle, MOTOR_RIGHT_PINS, 0, 0)
        lgpio.gpiochip_close(handle)


if __name__ == "__main__":
    test_motors()

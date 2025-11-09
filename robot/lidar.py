#!/usr/bin/env python3
"""
LD19 LIDAR Driver - Compatible with LidarInterface
Supports LD19, LD14, and similar UART-based 360° LIDARs
"""

import numpy as np
import serial
import struct
import threading
import time
import math
from typing import Tuple
from abc import ABC, abstractmethod


class LidarInterface(ABC):
    """Abstract base class for LIDAR sensors"""
    @abstractmethod
    def get_scan(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            ranges: Array of distance measurements (meters)
            angles: Array of corresponding angles (radians, robot frame)
        """
        pass

    @abstractmethod
    def start(self):
        """Initialize and start the LIDAR"""
        pass

    @abstractmethod
    def stop(self):
        """Stop and cleanup the LIDAR"""
        pass


class LD19Lidar(LidarInterface):
    """
    Driver for LD19/LD14 360° UART LIDAR

    Packet Format:
    - Header: 0x54
    - Length: 0x2C (44 bytes payload)
    - Speed: rotation speed (deg/s)
    - Start angle: angle * 100 (degrees)
    - 12 measurements: (distance_mm, confidence)
    - End angle: angle * 100 (degrees)
    - Timestamp: milliseconds
    - CRC: checksum
    """

    # Packet constants
    PACKET_LENGTH = 47
    MEASUREMENT_COUNT = 12
    MESSAGE_FORMAT = "<xBHH" + "HB" * MEASUREMENT_COUNT + "HHB"

    def __init__(self, port: str = '/dev/ttyAMA0', baudrate: int = 230400,
                 min_confidence: int = 50):
        """
        Initialize LD19 LIDAR

        Args:
            port: Serial port path
            baudrate: Serial baudrate (230400 for LD19)
            min_confidence: Minimum confidence value to accept (0-255)
        """
        try:
            self.port = port
            self.baudrate = baudrate
            self.min_confidence = min_confidence
            self.serial = serial.Serial(port, baudrate, timeout=0.5)

            self._running = False
            self._latest_scan = None
            self._lock = threading.Lock()

            print(f"✓ LD19 LIDAR connected on {port} @ {baudrate} baud")
        except ImportError:
            raise ImportError("pyserial not installed. Run: pip install pyserial")
        except Exception as e:
            raise RuntimeError(f"Failed to connect to LD19 LIDAR: {e}")

    def _parse_packet(self, data: bytes) -> list:
        """
        Parse a single LIDAR packet

        Returns:
            List of (angle_deg, distance_m, confidence) tuples
        """
        try:
            # Unpack the packet
            unpacked = struct.unpack(self.MESSAGE_FORMAT, data)
            length, speed, start_angle = unpacked[0:3]
            pos_data = unpacked[3:-3]
            stop_angle, timestamp, crc = unpacked[-3:]

            # Scale angles from centidegrees to degrees
            start_angle = float(start_angle) / 100.0
            stop_angle = float(stop_angle) / 100.0

            # Handle angle wraparound
            if stop_angle < start_angle:
                stop_angle += 360.0

            # Calculate angle step
            step_size = (stop_angle - start_angle) / (self.MEASUREMENT_COUNT - 1)

            # Extract measurements
            measurements = []
            for i in range(self.MEASUREMENT_COUNT):
                angle = start_angle + step_size * i
                distance_mm = pos_data[i * 2]
                confidence = pos_data[i * 2 + 1]

                # Filter by confidence and valid range
                if confidence >= self.min_confidence and distance_mm > 0:
                    distance_m = distance_mm / 1000.0
                    measurements.append((angle, distance_m, confidence))

            return measurements

        except Exception as e:
            print(f"Packet parse error: {e}")
            return []

    def _scan_loop(self):
        """
        Background thread to continuously read LIDAR packets
        Implements the same state machine logic as the original working code
        """
        scan_buffer = []
        data = b''
        state = 0  # 0=SYNC0, 1=SYNC1, 2=SYNC2, 3=LOCKED

        while self._running:
            try:
                # SYNC0: Find 1st header byte (0x54)
                if state == 0:
                    data = b''
                    scan_buffer = []
                    byte = self.serial.read(1)
                    if len(byte) > 0 and byte[0] == 0x54:
                        data = b'\x54'
                        state = 1

                # SYNC1: Find 2nd header byte (0x2C = length field)
                elif state == 1:
                    byte = self.serial.read(1)
                    if len(byte) > 0:
                        if byte[0] == 0x2C:
                            data += b'\x2C'
                            state = 2
                        else:
                            state = 0

                # SYNC2: Read remainder of packet
                elif state == 2:
                    remaining = self.serial.read(self.PACKET_LENGTH - 2)
                    data += remaining

                    if len(data) != self.PACKET_LENGTH:
                        state = 0
                        continue

                    measurements = self._parse_packet(data)
                    scan_buffer.extend(measurements)
                    state = 3  # Move to LOCKED state

                # LOCKED: Synchronized, read full packets directly
                elif state == 3:
                    data = self.serial.read(self.PACKET_LENGTH)

                    # Verify sync is maintained
                    if len(data) != self.PACKET_LENGTH or data[0] != 0x54:
                        print("WARNING: Serial sync lost - resynchronizing")
                        state = 0
                        continue

                    measurements = self._parse_packet(data)
                    scan_buffer.extend(measurements)

                    # After ~1 rotation (480+ measurements), update scan
                    if len(scan_buffer) >= 480:
                        # Convert to numpy arrays
                        angles_deg = np.array([m[0] for m in scan_buffer])
                        ranges_m = np.array([m[1] for m in scan_buffer])

                        # Convert angles to radians in robot frame [-pi, pi]
                        # (0° = forward, counterclockwise positive)
                        angles_rad = np.radians(angles_deg) + np.pi* 0.475

                        with self._lock:
                            self._latest_scan = (ranges_m, angles_rad)

                        # Keep in LOCKED state, clear buffer for next rotation
                        scan_buffer = []

            except Exception as e:
                if self._running:
                    print(f"LD19 scan error: {e}")
                    state = 0  # Reset to SYNC0 on error
                time.sleep(0.01)

    def start(self):
        """Start LIDAR scanning"""
        if self._running:
            print("LD19 already running")
            return

        self._running = True
        self._scan_thread = threading.Thread(target=self._scan_loop, daemon=True)
        self._scan_thread.start()
        print("LD19 LIDAR scanning started")

    def get_scan(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get latest scan data

        Returns:
            Tuple of (ranges, angles) where:
            - ranges: np.ndarray of distances in meters
            - angles: np.ndarray of angles in radians [-pi, pi]

        This matches the exact return format of RPLidarA1, YDLidarX2,
        and PlaceholderLidar for drop-in compatibility.
        """
        with self._lock:
            if self._latest_scan is None:
                # Return default scan matching other drivers
                return np.array([5.0] * 360), np.linspace(-math.pi, math.pi, 360)
            return self._latest_scan

    def stop(self):
        """Stop LIDAR and cleanup"""
        if not self._running:
            return

        self._running = False

        if hasattr(self, '_scan_thread'):
            self._scan_thread.join(timeout=2.0)

        self.serial.close()
        print("LD19 LIDAR stopped")


    def draw(self, ax):
        """
        Draw LIDAR scan on matplotlib axis
        Compatible with occupancy grid visualization

        Args:
            ax: matplotlib axis object to draw on
        """
        ranges, angles = self.get_scan()

        # Convert to cartesian coordinates
        x = ranges * np.sin(angles)
        y = ranges * np.cos(angles)

        # Draw scan points
        ax.scatter(x, y, c='red', s=1, alpha=0.5, label='LIDAR scan')

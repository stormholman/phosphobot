import time
import json
import requests
import math

# --- Configuration ---
# Replace with the IP address of the machine running phosphobot
PHOSPHOBOT_IP = "192.168.178.191"
# Replace with the port phosphobot is running on
PHOSPHOBOT_PORT = 80
# How often to read joint angles (in seconds)
READ_FREQUENCY = 0.1  # 10 Hz

JOINTS_READ_URL = f"http://{PHOSPHOBOT_IP}:{PHOSPHOBOT_PORT}/joints/read"


def rad_to_deg(radians):
    """Convert radians to degrees."""
    return [math.degrees(angle) for angle in radians]


def read_joint_angles():
    """
    Makes a POST request to read joint angles and returns the response.
    """
    try:
        response = requests.post(JOINTS_READ_URL, timeout=5)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error reading joint angles: {e}")
        return None


def main():
    """
    Continuously reads and displays joint angles at the specified frequency.
    """
    print(f"Reading joint angles from {JOINTS_READ_URL}")
    print(f"Update frequency: {1/READ_FREQUENCY:.1f} Hz")
    print("Press Ctrl+C to exit\n")
    
    try:
        while True:
            data = read_joint_angles()
            
            if data and "angles" in data:
                angles_rad = data["angles"]
                angles_deg = rad_to_deg(angles_rad)
                
                # Format angles to 2 decimal places
                formatted_rad = [f"{angle:.2f}" for angle in angles_rad]
                formatted_deg = [f"{angle:.2f}" for angle in angles_deg]
                
                print(f"Joint angles - Radians: {formatted_rad} | Degrees: {formatted_deg}", end="\r")
            
            time.sleep(READ_FREQUENCY)
            
    except KeyboardInterrupt:
        print("\nExiting.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
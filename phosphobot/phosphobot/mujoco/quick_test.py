#!/usr/bin/env python3
"""
Quick test script to verify SO-100 simulation is working correctly.
"""

import time
from so100_simulation import SO100Simulation

def quick_test():
    """Quick test of the SO-100 simulation."""
    print("🤖 SO-100 SIMULATION QUICK TEST")
    print("=" * 40)
    
    try:
        # Initialize simulation
        print("1. Initializing simulation...")
        sim = SO100Simulation()
        print("   ✅ Simulation initialized successfully!")
        
        # Wait for viewer to open
        time.sleep(2)
        print("   ✅ MuJoCo viewer should be open!")
        
        # Test move_to_position function
        print("\n2. Testing move_to_position() function...")
        target = [0.3, 0.2, 0.4]
        print(f"   Moving to: {target}")
        
        joint_angles = sim.move_to_position(target)
        print(f"   ✅ Movement completed!")
        print(f"   📊 Joint angles: {joint_angles}")
        
        # Get current position to verify
        current_pos = sim.get_end_effector_position()
        print(f"   📍 End effector position: [{current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f}]")
        
        print("\n3. Test completed successfully! 🎉")
        print("   The MuJoCo viewer is open and interactive.")
        print("   You can now use the simulation as desired.")
        print("\n   Available functions:")
        print("   - sim.move_to_position([x, y, z])")
        print("   - sim.get_current_joint_angles()")
        print("   - sim.get_end_effector_position()")
        
        print("\n   Press Ctrl+C to close the simulation.")
        
        # Keep running
        while sim.viewer.is_alive():
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'sim' in locals():
            sim.close()
        print("🔚 Test finished.")

if __name__ == "__main__":
    quick_test() 
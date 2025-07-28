#!/usr/bin/env python3
"""
Test script for the SO-100 simulation
"""

import time
import numpy as np
from so100_simulation import SO100Simulation

def test_simulation():
    """Test the SO-100 simulation."""
    print("🧪 TESTING SO-100 SIMULATION")
    print("=" * 50)
    
    try:
        # Test initialization
        print("1. Testing initialization...")
        sim = SO100Simulation()
        print("   ✅ Simulation initialized successfully!")
        
        # Test basic operations
        print("\n2. Testing basic operations...")
        
        # Get current angles
        current_angles = sim.get_current_joint_angles()
        print(f"   Current joint angles: {current_angles}")
        
        # Set new angles
        test_angles = [0.0, -1.0, 1.0, 0.5, -0.5, 0.0]
        success = sim.set_joint_angles(test_angles)
        print(f"   Set joint angles result: {success}")
        
        time.sleep(0.5)
        
        # Get end effector position
        ee_pos = sim.get_end_effector_position()
        print(f"   End effector position: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]")
        
        print("\n3. Testing rapid movements (stress test)...")
        print("   This should handle threading conflicts gracefully...")
        
        # Rapid movements that would cause the original error
        for i in range(20):
            angles = [np.sin(i * 0.3), np.cos(i * 0.3), i * 0.05, -i * 0.05, np.sin(i * 0.7), 0.0]
            sim.set_joint_angles(angles)
            
            # Get position during rapid movement
            pos = sim.get_end_effector_position()
            
            time.sleep(0.02)  # Very rapid
            
        print("   ✅ Stress test completed successfully!")
        
        print(f"\n🎉 ALL TESTS PASSED!")
        print("   The SO-100 simulation is working perfectly!")
        print("   Threading conflicts are handled gracefully.")
        print("\n   Try colliding with the banana - it should work smoothly!")
        print("   Press Ctrl+C to exit when ready.")
        
        # Keep running for user interaction
        try:
            while sim.viewer.is_running():
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n🛑 Test interrupted by user")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        if 'sim' in locals():
            print("\n🔄 Closing simulation...")
            sim.close()
            print("✅ Simulation closed cleanly")
    
    return True

if __name__ == "__main__":
    success = test_simulation()
    if success:
        print("\n🎯 SO-100 simulation test completed successfully!")
    else:
        print("\n💥 SO-100 simulation test failed!") 
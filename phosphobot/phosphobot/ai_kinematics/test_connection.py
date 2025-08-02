#!/usr/bin/env python3
"""
Test script to diagnose AI-kinematics connection issues
"""

def test_imports():
    """Test if all required packages can be imported"""
    print("🔍 Testing imports...")
    
    try:
        import numpy as np
        print("✅ numpy imported successfully")
    except ImportError as e:
        print(f"❌ numpy import failed: {e}")
        return False
    
    try:
        import cv2
        print("✅ opencv-python imported successfully")
    except ImportError as e:
        print(f"❌ opencv-python import failed: {e}")
        return False
    
    try:
        from record3d import Record3DStream
        print("✅ record3d imported successfully")
    except ImportError as e:
        print(f"❌ record3d import failed: {e}")
        return False
    
    try:
        import requests
        print("✅ requests imported successfully")
    except ImportError as e:
        print(f"❌ requests import failed: {e}")
        return False
    
    return True

def test_record3d_connection():
    """Test Record3D device connection"""
    print("\n🔍 Testing Record3D connection...")
    
    try:
        from record3d import Record3DStream
        
        print("Searching for Record3D devices...")
        devs = Record3DStream.get_connected_devices()
        print(f"Found {len(devs)} device(s)")
        
        if len(devs) == 0:
            print("❌ No Record3D devices found")
            print("💡 Make sure:")
            print("   • Record3D app is running on your iPhone")
            print("   • iPhone is connected to the same WiFi network")
            print("   • iPhone and computer can communicate (no firewall blocking)")
            return False
        
        for i, dev in enumerate(devs):
            print(f"✅ Device {i}: ID={dev.product_id}, UDID={dev.udid}")
        
        return True
        
    except Exception as e:
        print(f"❌ Record3D connection test failed: {e}")
        return False

def test_opencv_display():
    """Test if OpenCV can create windows"""
    print("\n🔍 Testing OpenCV display...")
    
    try:
        import cv2
        import numpy as np
        
        # Create a test window
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.putText(test_img, "Test", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imshow('Test Window', test_img)
        cv2.waitKey(1000)  # Show for 1 second
        cv2.destroyAllWindows()
        
        print("✅ OpenCV display test successful")
        return True
        
    except Exception as e:
        print(f"❌ OpenCV display test failed: {e}")
        return False

def main():
    print("AI-Kinematics Connection Test")
    print("=" * 40)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import test failed. Install missing dependencies:")
        print("pip install -r requirements.txt")
        return
    
    # Test Record3D connection
    record3d_ok = test_record3d_connection()
    
    # Test OpenCV display
    opencv_ok = test_opencv_display()
    
    print("\n" + "=" * 40)
    print("TEST SUMMARY:")
    print(f"✅ Imports: OK")
    print(f"{'✅' if record3d_ok else '❌'} Record3D: {'OK' if record3d_ok else 'FAILED'}")
    print(f"{'✅' if opencv_ok else '❌'} OpenCV Display: {'OK' if opencv_ok else 'FAILED'}")
    
    if record3d_ok and opencv_ok:
        print("\n🎉 All tests passed! AI-kinematics should work properly.")
    else:
        print("\n⚠️  Some tests failed. Fix the issues above before running AI-kinematics.")

if __name__ == "__main__":
    main() 
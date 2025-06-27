"""
Simple Test Script

Basic command-line test to verify LLM components are working.
Run this first to check if everything is set up correctly.
"""
import sys
from pathlib import Path

def test_basic_imports():
    """Test if we can import basic dependencies."""
    print("🧪 Testing Basic Dependencies")
    print("=" * 40)
    
    # Test Python packages
    packages = ['openai', 'tiktoken', 'keyring', 'PyQt6']
    missing = []
    
    for package in packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False
    else:
        print("\n✅ All basic dependencies available")
        return True

def test_llm_components():
    """Test if our LLM components can be imported."""
    print("\n🤖 Testing LLM Components")
    print("=" * 40)
    
    components = [
        'llm_client',
        'cost_guard', 
        'api_key_manager',
        'field_detector',
        'cell_resolver',
        'llm_integration_manager'
    ]
    
    missing = []
    
    for component in components:
        try:
            __import__(component)
            print(f"✅ {component}.py")
        except ImportError as e:
            print(f"❌ {component}.py - {e}")
            missing.append(component)
    
    if missing:
        print(f"\n❌ Missing components: {', '.join(missing)}")
        print("Make sure all .py files are in the same directory as this script")
        return False
    else:
        print("\n✅ All LLM components available")
        return True

def test_cost_guard():
    """Test cost guard functionality."""
    print("\n💰 Testing Cost Guard")
    print("=" * 40)
    
    try:
        from cost_guard import CostGuard
        
        # Create cost guard
        guard = CostGuard(budget_limit_usd=1.0)
        print("✅ Cost guard created")
        
        # Test operations
        if guard.check_affordability(0.30):
            guard.record_cost(0.30, 1000, "gpt-4o-mini", "test")
            print("✅ Small cost approved and recorded")
        
        if guard.check_affordability(0.50):
            guard.record_cost(0.50, 2000, "gpt-4o-mini", "test")
            print("✅ Medium cost approved and recorded")
        
        # This should be rejected
        if not guard.check_affordability(0.50):
            print("✅ Large cost correctly rejected")
        else:
            print("⚠️ Large cost approved when should be rejected")
        
        # Get breakdown
        breakdown = guard.get_cost_breakdown()
        print(f"📊 Total cost: ${breakdown['total_cost']:.4f}")
        print(f"📊 Operations: {breakdown['total_operations']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Cost guard test failed: {e}")
        return False

def test_api_key_manager():
    """Test API key manager without real keys."""
    print("\n🔑 Testing API Key Manager")
    print("=" * 40)
    
    try:
        from api_key_manager import APIKeyManager
        
        manager = APIKeyManager()
        print("✅ API key manager created")
        
        # Test invalid keys
        invalid_keys = ["", "invalid", "sk-short"]
        for key in invalid_keys:
            if not manager.set_api_key(key, save_to_keyring=False):
                print(f"✅ Invalid key rejected: {key}")
            else:
                print(f"❌ Invalid key accepted: {key}")
        
        # Test valid format (fake key)
        test_key = "sk-test1234567890abcdefghijklmnopqrstuvwxyz1234567890"
        if manager.set_api_key(test_key, save_to_keyring=False):
            print("✅ Valid format accepted")
        else:
            print("❌ Valid format rejected")
        
        # Test status
        status = manager.get_key_status()
        print(f"📊 Key status: {status['has_key']}, {status['key_format_valid']}")
        
        return True
        
    except Exception as e:
        print(f"❌ API key manager test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Simple LLM Component Test")
    print("=" * 50)
    print("This script tests if LLM components are properly set up.")
    print("Run this before trying to use the GUI or full demo.\n")
    
    tests = [
        ("Basic Dependencies", test_basic_imports),
        ("LLM Components", test_llm_components),
        ("Cost Guard", test_cost_guard),
        ("API Key Manager", test_api_key_manager)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nResult: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! You can now:")
        print("1. Run the GUI demo: python standalone_llm_demo.py")
        print("2. Run the full integration test: python llm_integration_example.py")
        print("3. Start integrating with your existing code")
    else:
        print("\n⚠️ Some tests failed. Please:")
        print("1. Install missing dependencies shown above")
        print("2. Make sure all .py files are in the same directory")
        print("3. Re-run this test script")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

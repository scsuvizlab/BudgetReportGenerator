"""
LLM Integration Example & Testing

Demonstrates the complete LLM integration workflow and provides testing capabilities.
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any

# Import existing modules (these would be your actual modules)
try:
    from template_document import TemplateParser, TemplateDocument
    from budget_book import BudgetParser, BudgetBook
    from session_state import SessionState
    from document_generator import DocumentGenerator
except ImportError:
    print("Warning: Could not import existing modules. This is expected if running standalone.")
    # Create mock classes for testing
    class TemplateDocument:
        def __init__(self):
            self.content = "Sample template with {PI_Salary_Total} and {Travel_Cost}"
            self.placeholders = []
    
    class BudgetBook:
        def __init__(self):
            self.cells = []
            self.sheets = ["Budget"]

# Import LLM modules
from llm_client import LLMClient
from cost_guard import CostGuard
from api_key_manager import APIKeyManager
from field_detector import FieldDetector
from cell_resolver import CellResolver
from llm_integration_manager import LLMIntegrationManager

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LLMIntegrationDemo:
    """
    Comprehensive demonstration of LLM integration capabilities.
    
    This class shows how all the LLM components work together
    and provides testing methods for validation.
    """
    
    def __init__(self):
        """Initialize the demo."""
        self.llm_manager = LLMIntegrationManager()
        self.test_results = {}
        self.demo_config_path = Path("demo_llm_config.json")
    
    def create_demo_config(self, api_key: str = None) -> Path:
        """
        Create a demo configuration file.
        
        Args:
            api_key: Optional API key to include
            
        Returns:
            Path to the created config file
        """
        config = {
            "openai_api_key": api_key or "your-api-key-here",
            "default_model": "gpt-4o-mini",
            "cost_limit_usd": 2.0,
            "auto_improve_enabled": True,
            "heuristics_first": True
        }
        
        with open(self.demo_config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Demo configuration created: {self.demo_config_path}")
        print("Edit this file to add your actual OpenAI API key.")
        return self.demo_config_path
    
    def test_api_key_management(self, api_key: str = None) -> bool:
        """
        Test API key management functionality.
        
        Args:
            api_key: API key to test (optional)
            
        Returns:
            True if tests pass
        """
        print("\n=== Testing API Key Management ===")
        
        try:
            api_manager = APIKeyManager()
            
            # Test key format validation
            print("Testing key format validation...")
            
            # Test invalid formats
            invalid_keys = ["", "invalid", "sk-short", "not-a-key"]
            for invalid_key in invalid_keys:
                result = api_manager.set_api_key(invalid_key, save_to_keyring=False)
                if result:
                    print(f"  ❌ Invalid key accepted: {invalid_key}")
                    return False
                else:
                    print(f"  ✅ Invalid key rejected: {invalid_key}")
            
            # Test valid format (but fake key)
            test_key = "sk-test1234567890abcdefghijklmnopqrstuvwxyz1234567890"
            result = api_manager.set_api_key(test_key, save_to_keyring=False)
            if result:
                print(f"  ✅ Valid format accepted")
            else:
                print(f"  ❌ Valid format rejected")
                return False
            
            # Test status reporting
            status = api_manager.get_key_status()
            print(f"  Key status: {status}")
            
            # Test with real key if provided
            if api_key:
                print("Testing real API key...")
                api_manager.set_api_key(api_key, save_to_keyring=False)
                is_valid = api_manager.validate_current_key()
                print(f"  Real key validation: {'✅ Valid' if is_valid else '❌ Invalid'}")
                return is_valid
            
            print("  ⚠️ No real API key provided - skipping validation test")
            return True
            
        except Exception as e:
            print(f"  ❌ API key management test failed: {e}")
            return False
    
    def test_cost_tracking(self) -> bool:
        """Test cost tracking and budget enforcement."""
        print("\n=== Testing Cost Tracking ===")
        
        try:
            cost_guard = CostGuard(budget_limit_usd=1.0, warning_threshold=0.8)
            
            # Test budget checking
            print("Testing budget enforcement...")
            
            # Should be affordable
            if not cost_guard.check_affordability(0.30):
                print("  ❌ Small cost rejected incorrectly")
                return False
            print("  ✅ Small cost accepted")
            
            # Record some usage
            cost_guard.record_cost(0.25, 1000, "gpt-4o-mini", "test_operation")
            cost_guard.record_cost(0.35, 1200, "gpt-4o-mini", "test_operation")
            
            # Should trigger warning
            cost_guard.record_cost(0.30, 800, "gpt-4o-mini", "test_operation")
            
            # Should be rejected (would exceed budget)
            if cost_guard.check_affordability(0.50):
                print("  ❌ Large cost accepted when should be rejected")
                return False
            print("  ✅ Large cost correctly rejected")
            
            # Test breakdown
            breakdown = cost_guard.get_cost_breakdown()
            print(f"  Cost breakdown: ${breakdown['total_cost']:.4f}")
            print(f"  Budget utilization: {breakdown['budget_utilization']:.1%}")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Cost tracking test failed: {e}")
            return False
    
    def test_llm_client(self, api_key: str) -> bool:
        """
        Test LLM client functionality.
        
        Args:
            api_key: Valid OpenAI API key
            
        Returns:
            True if tests pass
        """
        print("\n=== Testing LLM Client ===")
        
        try:
            client = LLMClient(api_key, "gpt-4o-mini")
            
            # Test cost estimation
            test_prompt = "Analyze this simple budget template."
            estimated_cost = client.estimate_cost(test_prompt, max_tokens=100)
            print(f"  Estimated cost: ${estimated_cost:.4f}")
            
            # Test API key validation
            print("  Testing API key validation...")
            is_valid = client.validate_api_key()
            if not is_valid:
                print("  ❌ API key validation failed")
                return False
            print("  ✅ API key validated")
            
            # Test simple LLM call
            print("  Testing LLM call...")
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Reply with just the word 'success'"}
            ]
            
            response = client.call(messages, max_tokens=10)
            
            if response.success and "success" in response.content.lower():
                print("  ✅ LLM call successful")
                print(f"  Response: {response.content}")
                print(f"  Cost: ${response.cost_usd:.4f}")
                print(f"  Tokens: {response.usage['total_tokens']}")
                return True
            else:
                print(f"  ❌ LLM call failed: {response.error_message}")
                return False
            
        except Exception as e:
            print(f"  ❌ LLM client test failed: {e}")
            return False
    
    def test_field_detection(self, api_key: str) -> bool:
        """
        Test field detection functionality.
        
        Args:
            api_key: Valid OpenAI API key
            
        Returns:
            True if tests pass
        """
        print("\n=== Testing Field Detection ===")
        
        try:
            client = LLMClient(api_key, "gpt-4o-mini")
            detector = FieldDetector(client)
            
            # Create a sample template
            template_content = """
            Budget Justification
            
            Principal Investigator Salary: {PI_Salary_Total}
            Equipment Costs: {Equipment_Total}
            Travel Expenses: {Travel_Domestic_Total}
            
            Total Project Cost: {Total_Budget}
            """
            
            # Create mock template document
            template = TemplateDocument()
            template.content = template_content
            
            print("  Analyzing template with LLM...")
            suggestions = detector.analyze_template(template)
            
            if suggestions:
                print(f"  ✅ Found {len(suggestions)} field suggestions")
                for suggestion in suggestions:
                    print(f"    - {suggestion.field_name}: {suggestion.description}")
                    print(f"      Type: {suggestion.data_type}, Confidence: {suggestion.confidence:.2f}")
                
                # Check if we got reasonable suggestions
                field_names = [s.field_name for s in suggestions]
                expected_fields = ["PI_Salary_Total", "Equipment_Total", "Travel_Domestic_Total", "Total_Budget"]
                
                found_count = sum(1 for field in expected_fields if any(field in fn for fn in field_names))
                if found_count >= 2:  # At least 2 out of 4
                    print(f"  ✅ Found {found_count}/4 expected fields")
                    return True
                else:
                    print(f"  ⚠️ Only found {found_count}/4 expected fields")
                    return False
            else:
                print("  ❌ No field suggestions generated")
                return False
                
        except Exception as e:
            print(f"  ❌ Field detection test failed: {e}")
            return False
    
    def test_full_integration(self, api_key: str) -> bool:
        """
        Test the full LLM integration workflow.
        
        Args:
            api_key: Valid OpenAI API key
            
        Returns:
            True if tests pass
        """
        print("\n=== Testing Full Integration ===")
        
        try:
            # Initialize LLM manager
            print("  Initializing LLM manager...")
            success = self.llm_manager.initialize_with_api_key(
                api_key=api_key,
                budget_limit=1.0,
                default_model="gpt-4o-mini"
            )
            
            if not success:
                print("  ❌ LLM manager initialization failed")
                return False
            print("  ✅ LLM manager initialized")
            
            # Create sample template and budget
            template = TemplateDocument()
            template.content = """
            Budget Justification
            PI Salary: {PI_Salary_Total}
            Travel: {Travel_Total}
            Equipment: {Equipment_Total}
            """
            
            budget = BudgetBook()
            # Add some mock budget cells
            # (In real usage, these would come from parsing an Excel file)
            
            # Test template analysis
            print("  Analyzing template...")
            suggestions = self.llm_manager.analyze_template_fields(template)
            print(f"    Found {len(suggestions)} suggestions")
            
            # Test usage summary
            usage = self.llm_manager.get_usage_summary()
            print(f"  Usage summary:")
            print(f"    Total cost: ${usage['total_cost']:.4f}")
            print(f"    Enhanced mappings: {usage['enhanced_mappings']}")
            print(f"    Field suggestions: {usage['field_suggestions']}")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Full integration test failed: {e}")
            return False
    
    def run_interactive_demo(self):
        """Run an interactive demonstration."""
        print("🤖 LLM Integration Interactive Demo")
        print("=" * 50)
        
        # Check for config file
        if self.demo_config_path.exists():
            print(f"Found config file: {self.demo_config_path}")
            try:
                with open(self.demo_config_path, 'r') as f:
                    config = json.load(f)
                api_key = config.get('openai_api_key', '')
            except Exception as e:
                print(f"Error reading config: {e}")
                api_key = ''
        else:
            print("No config file found.")
            api_key = ''
        
        # Get API key from user if needed
        if not api_key or api_key == "your-api-key-here":
            print("\nTo run the full demo, you need an OpenAI API key.")
            print("You can either:")
            print("1. Create a config file with your API key")
            print("2. Enter your API key now (it won't be saved)")
            print("3. Run limited tests without API key")
            
            choice = input("\nEnter choice (1/2/3): ").strip()
            
            if choice == "1":
                self.create_demo_config()
                print("\nEdit the config file and run the demo again.")
                return
            elif choice == "2":
                api_key = input("Enter your OpenAI API key: ").strip()
            else:
                api_key = None
        
        # Run tests
        print("\nRunning tests...")
        
        # Test 1: API Key Management (always runs)
        self.test_results['api_key_management'] = self.test_api_key_management(api_key)
        
        # Test 2: Cost Tracking (always runs)
        self.test_results['cost_tracking'] = self.test_cost_tracking()
        
        if api_key:
            # Test 3: LLM Client
            self.test_results['llm_client'] = self.test_llm_client(api_key)
            
            # Test 4: Field Detection
            if self.test_results['llm_client']:
                self.test_results['field_detection'] = self.test_field_detection(api_key)
            
            # Test 5: Full Integration
            if self.test_results.get('field_detection', False):
                self.test_results['full_integration'] = self.test_full_integration(api_key)
        else:
            print("\nSkipping LLM tests (no API key provided)")
        
        # Summary
        self.print_test_summary()
    
    def print_test_summary(self):
        """Print a summary of all test results."""
        print("\n" + "=" * 50)
        print("🧪 TEST SUMMARY")
        print("=" * 50)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name.replace('_', ' ').title()}: {status}")
        
        print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("🎉 All tests passed! LLM integration is ready to use.")
        else:
            print("⚠️ Some tests failed. Check the output above for details.")
        
        # Usage recommendations
        if self.test_results.get('llm_client', False):
            print("\n💡 RECOMMENDATIONS:")
            print("- LLM integration is working correctly")
            print("- You can now enable LLM features in the main application")
            print("- Monitor costs using the built-in cost tracking")
            print("- Start with gpt-4o-mini model for cost efficiency")
        elif not any(self.test_results.values()):
            print("\n💡 NEXT STEPS:")
            print("- Get an OpenAI API key from platform.openai.com")
            print("- Add billing information to your OpenAI account")
            print("- Run this demo again with your API key")
    
    def cleanup_demo_files(self):
        """Clean up demo files."""
        if self.demo_config_path.exists():
            self.demo_config_path.unlink()
            print(f"Cleaned up: {self.demo_config_path}")


def main():
    """Main demo function."""
    demo = LLMIntegrationDemo()
    
    try:
        demo.run_interactive_demo()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
    finally:
        # Ask about cleanup
        try:
            cleanup = input("\nClean up demo files? (y/n): ").strip().lower()
            if cleanup == 'y':
                demo.cleanup_demo_files()
        except:
            pass  # Ignore cleanup errors


if __name__ == "__main__":
    main()

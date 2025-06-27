"""
API Key Debug Script

Specifically debug the API key validation issue.
"""
import sys
import traceback
import logging
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
    QWidget, QPushButton, QLabel, QTextEdit, QMessageBox,
    QGroupBox, QLineEdit
)

# Configure detailed logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class APIKeyDebugWindow(QMainWindow):
    """Debug API key validation specifically."""
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the debug UI."""
        self.setWindowTitle("🔑 API Key Debug")
        self.setMinimumSize(800, 600)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Title
        title = QLabel("🔑 API Key Validation Debug")
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin: 10px;")
        layout.addWidget(title)
        
        # API Key section
        api_group = QGroupBox("API Key Analysis")
        api_layout = QVBoxLayout()
        
        api_input_layout = QHBoxLayout()
        self.api_key_input = QLineEdit()
        self.api_key_input.setPlaceholderText("sk-your-openai-api-key-here")
        self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        
        self.analyze_button = QPushButton("Analyze API Key")
        self.analyze_button.clicked.connect(self.analyze_api_key)
        
        self.test_manager_button = QPushButton("Test Manager Setting")
        self.test_manager_button.clicked.connect(self.test_manager_setting)
        
        api_input_layout.addWidget(QLabel("API Key:"))
        api_input_layout.addWidget(self.api_key_input)
        api_input_layout.addWidget(self.analyze_button)
        api_input_layout.addWidget(self.test_manager_button)
        
        api_layout.addLayout(api_input_layout)
        api_group.setLayout(api_layout)
        layout.addWidget(api_group)
        
        # Debug output
        debug_group = QGroupBox("Debug Output")
        debug_layout = QVBoxLayout()
        
        self.debug_output = QTextEdit()
        self.debug_output.setReadOnly(True)
        
        clear_button = QPushButton("Clear Output")
        clear_button.clicked.connect(self.debug_output.clear)
        
        debug_layout.addWidget(self.debug_output)
        debug_layout.addWidget(clear_button)
        debug_group.setLayout(debug_layout)
        layout.addWidget(debug_group)
    
    def log_to_output(self, message):
        """Add message to debug output."""
        self.debug_output.append(message)
        self.debug_output.ensureCursorVisible()
        print(message)
    
    def analyze_api_key(self):
        """Analyze the API key format in detail."""
        api_key = self.api_key_input.text().strip()
        if not api_key:
            self.log_to_output("❌ Please enter an API key first")
            return
        
        self.log_to_output("🔍 Analyzing API Key Format")
        self.log_to_output("=" * 40)
        
        # Basic properties
        self.log_to_output(f"Length: {len(api_key)} characters")
        self.log_to_output(f"Starts with 'sk-': {api_key.startswith('sk-')}")
        self.log_to_output(f"First 10 chars: {api_key[:10]}...")
        self.log_to_output(f"Last 4 chars: ...{api_key[-4:]}")
        
        # Character analysis
        allowed_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-")
        key_chars = set(api_key)
        invalid_chars = key_chars - allowed_chars
        
        self.log_to_output(f"Total unique characters: {len(key_chars)}")
        self.log_to_output(f"All characters allowed: {len(invalid_chars) == 0}")
        
        if invalid_chars:
            self.log_to_output(f"Invalid characters found: {invalid_chars}")
            for char in invalid_chars:
                self.log_to_output(f"  - '{char}' (ASCII: {ord(char)})")
        
        # Test our validation function manually
        self.log_to_output("\n🧪 Manual Validation Tests:")
        
        # Test 1: Empty check
        if not api_key or not api_key.strip():
            self.log_to_output("❌ Empty key test: FAILED")
        else:
            self.log_to_output("✅ Empty key test: PASSED")
        
        # Test 2: Format check
        cleaned_key = api_key.strip()
        if not cleaned_key.startswith("sk-"):
            self.log_to_output("❌ 'sk-' prefix test: FAILED")
        else:
            self.log_to_output("✅ 'sk-' prefix test: PASSED")
        
        # Test 3: Length check
        if len(cleaned_key) < 40:
            self.log_to_output(f"❌ Length test: FAILED (needs ≥40, got {len(cleaned_key)})")
        else:
            self.log_to_output(f"✅ Length test: PASSED ({len(cleaned_key)} chars)")
        
        # Test 4: Character set check
        if not set(cleaned_key).issubset(allowed_chars):
            self.log_to_output("❌ Character set test: FAILED")
        else:
            self.log_to_output("✅ Character set test: PASSED")
        
        # Overall manual validation
        manual_valid = (
            api_key and api_key.strip() and
            cleaned_key.startswith("sk-") and
            len(cleaned_key) >= 40 and
            set(cleaned_key).issubset(allowed_chars)
        )
        
        self.log_to_output(f"\n🎯 Manual validation result: {'✅ VALID' if manual_valid else '❌ INVALID'}")
    
    def test_manager_setting(self):
        """Test the actual manager API key setting process."""
        api_key = self.api_key_input.text().strip()
        if not api_key:
            self.log_to_output("❌ Please enter an API key first")
            return
        
        self.log_to_output("\n🔧 Testing Manager API Key Setting")
        self.log_to_output("=" * 50)
        
        try:
            # Import and create API key manager
            self.log_to_output("1. Importing APIKeyManager...")
            from api_key_manager import APIKeyManager
            
            self.log_to_output("2. Creating APIKeyManager instance...")
            manager = APIKeyManager()
            
            self.log_to_output("3. Testing set_api_key with save_to_keyring=False...")
            result = manager.set_api_key(api_key, save_to_keyring=False)
            
            if result:
                self.log_to_output("✅ set_api_key returned True")
                
                # Test getting the key back
                self.log_to_output("4. Testing get_api_key...")
                retrieved_key = manager.get_api_key()
                if retrieved_key:
                    self.log_to_output("✅ get_api_key returned key")
                    self.log_to_output(f"   Key matches: {retrieved_key == api_key}")
                else:
                    self.log_to_output("❌ get_api_key returned None")
                
                # Test key status
                self.log_to_output("5. Testing get_key_status...")
                status = manager.get_key_status()
                self.log_to_output(f"   Status: {status}")
                
            else:
                self.log_to_output("❌ set_api_key returned False")
                
                # Let's try to figure out why
                self.log_to_output("\n🔍 Debugging why set_api_key failed...")
                
                # Test each validation step manually
                cleaned_key = api_key.strip()
                
                # Check empty
                if not api_key or not api_key.strip():
                    self.log_to_output("   Reason: Empty API key")
                    return
                
                # Check format validation
                if not manager._is_valid_key_format(cleaned_key):
                    self.log_to_output("   Reason: _is_valid_key_format returned False")
                    
                    # Dig deeper into format validation
                    self.log_to_output("   Detailed format validation:")
                    
                    if not cleaned_key.startswith("sk-"):
                        self.log_to_output("     - Missing 'sk-' prefix")
                    else:
                        self.log_to_output("     ✅ Has 'sk-' prefix")
                    
                    if len(cleaned_key) < 40:
                        self.log_to_output(f"     - Too short: {len(cleaned_key)} < 40")
                    else:
                        self.log_to_output(f"     ✅ Length OK: {len(cleaned_key)} >= 40")
                    
                    allowed_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-")
                    if not set(cleaned_key).issubset(allowed_chars):
                        invalid_chars = set(cleaned_key) - allowed_chars
                        self.log_to_output(f"     - Invalid characters: {invalid_chars}")
                    else:
                        self.log_to_output("     ✅ All characters valid")
                else:
                    self.log_to_output("   _is_valid_key_format returned True, but set_api_key still failed")
        
        except Exception as e:
            self.log_to_output(f"❌ Error testing manager: {str(e)}")
            self.log_to_output("Full traceback:")
            self.log_to_output(traceback.format_exc())
    
    def test_with_different_approaches(self):
        """Test different approaches to setting the API key."""
        api_key = self.api_key_input.text().strip()
        if not api_key:
            self.log_to_output("❌ Please enter an API key first")
            return
        
        self.log_to_output("\n🧪 Testing Different Approaches")
        self.log_to_output("=" * 40)
        
        try:
            from api_key_manager import APIKeyManager
            
            # Approach 1: With keyring
            self.log_to_output("Approach 1: With keyring (default)")
            manager1 = APIKeyManager()
            result1 = manager1.set_api_key(api_key, save_to_keyring=True)
            self.log_to_output(f"  Result: {result1}")
            
            # Approach 2: Without keyring
            self.log_to_output("Approach 2: Without keyring")
            manager2 = APIKeyManager()
            result2 = manager2.set_api_key(api_key, save_to_keyring=False)
            self.log_to_output(f"  Result: {result2}")
            
            # Approach 3: Direct assignment (bypass validation)
            self.log_to_output("Approach 3: Direct assignment")
            manager3 = APIKeyManager()
            manager3._cached_key = api_key
            retrieved = manager3.get_api_key()
            self.log_to_output(f"  Retrieved key: {retrieved == api_key}")
            
        except Exception as e:
            self.log_to_output(f"❌ Error in approaches test: {str(e)}")


def main():
    """Run the API key debug."""
    app = QApplication(sys.argv)
    
    window = APIKeyDebugWindow()
    window.show()
    
    # Add additional test to menu
    menubar = window.menuBar()
    test_menu = menubar.addMenu("Tests")
    
    test_approaches_action = test_menu.addAction("Test Different Approaches")
    test_approaches_action.triggered.connect(window.test_with_different_approaches)
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

"""
Debug LLM Demo

Enhanced version with detailed error reporting to identify initialization issues.
"""
import sys
import traceback
import logging
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
    QWidget, QPushButton, QLabel, QTextEdit, QMessageBox,
    QGroupBox, QLineEdit, QProgressBar
)
from PyQt6.QtCore import Qt, QTimer

# Configure detailed logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DebugLLMWindow(QMainWindow):
    """Debug version with detailed error reporting."""
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.test_api_key_direct()
    
    def setup_ui(self):
        """Set up the debug UI."""
        self.setWindowTitle("🔍 Debug LLM Integration")
        self.setMinimumSize(900, 700)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Title
        title = QLabel("🔍 Debug LLM Integration")
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin: 10px;")
        layout.addWidget(title)
        
        # API Key section
        api_group = QGroupBox("API Key Testing")
        api_layout = QVBoxLayout()
        
        api_input_layout = QHBoxLayout()
        self.api_key_input = QLineEdit()
        self.api_key_input.setPlaceholderText("sk-your-openai-api-key-here")
        self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        
        self.test_direct_button = QPushButton("Test Direct LLM Call")
        self.test_direct_button.clicked.connect(self.test_api_key_direct)
        
        self.test_manager_button = QPushButton("Test LLM Manager")
        self.test_manager_button.clicked.connect(self.test_llm_manager)
        
        api_input_layout.addWidget(QLabel("API Key:"))
        api_input_layout.addWidget(self.api_key_input)
        api_input_layout.addWidget(self.test_direct_button)
        api_input_layout.addWidget(self.test_manager_button)
        
        api_layout.addLayout(api_input_layout)
        api_group.setLayout(api_layout)
        layout.addWidget(api_group)
        
        # Debug output
        debug_group = QGroupBox("Debug Output")
        debug_layout = QVBoxLayout()
        
        self.debug_output = QTextEdit()
        self.debug_output.setReadOnly(True)
        self.debug_output.setFont(self.debug_output.font())  # Monospace
        
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
        print(message)  # Also print to console
    
    def test_api_key_direct(self):
        """Test API key with direct LLM client."""
        api_key = self.api_key_input.text().strip()
        if not api_key:
            api_key = "test-key-for-import-testing"
        
        self.log_to_output("🧪 Testing Direct LLM Client")
        self.log_to_output("=" * 50)
        
        try:
            self.log_to_output("1. Importing llm_client...")
            from llm_client import LLMClient
            self.log_to_output("✅ llm_client imported successfully")
            
            if api_key and api_key != "test-key-for-import-testing":
                self.log_to_output("2. Creating LLM client...")
                client = LLMClient(api_key, "gpt-4o-mini")
                self.log_to_output("✅ LLM client created")
                
                self.log_to_output("3. Testing API key validation...")
                is_valid = client.validate_api_key()
                if is_valid:
                    self.log_to_output("✅ API key is valid")
                    
                    self.log_to_output("4. Testing simple LLM call...")
                    messages = [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Reply with just 'Hello World'"}
                    ]
                    
                    response = client.call(messages, max_tokens=10)
                    if response.success:
                        self.log_to_output(f"✅ LLM call successful: {response.content}")
                        self.log_to_output(f"💰 Cost: ${response.cost_usd:.6f}")
                    else:
                        self.log_to_output(f"❌ LLM call failed: {response.error_message}")
                else:
                    self.log_to_output("❌ API key validation failed")
            else:
                self.log_to_output("⚠️ No API key provided - skipping validation tests")
            
        except Exception as e:
            self.log_to_output(f"❌ Direct LLM test failed: {str(e)}")
            self.log_to_output("Full traceback:")
            self.log_to_output(traceback.format_exc())
    
    def test_llm_manager(self):
        """Test LLM integration manager step by step."""
        api_key = self.api_key_input.text().strip()
        if not api_key:
            self.log_to_output("❌ Please enter an API key first")
            return
        
        self.log_to_output("\n🔧 Testing LLM Integration Manager")
        self.log_to_output("=" * 50)
        
        try:
            # Step 1: Import manager
            self.log_to_output("1. Importing LLMIntegrationManager...")
            from llm_integration_manager import LLMIntegrationManager
            self.log_to_output("✅ LLMIntegrationManager imported")
            
            # Step 2: Create manager
            self.log_to_output("2. Creating manager instance...")
            manager = LLMIntegrationManager()
            self.log_to_output("✅ Manager instance created")
            
            # Step 3: Check initial state
            self.log_to_output("3. Checking initial state...")
            self.log_to_output(f"   - Initialized: {manager.is_initialized}")
            self.log_to_output(f"   - API key manager: {manager.api_key_manager is not None}")
            self.log_to_output(f"   - Cost guard: {manager.cost_guard is not None}")
            self.log_to_output(f"   - LLM client: {manager.llm_client is not None}")
            
            # Step 4: Initialize with API key
            self.log_to_output("4. Initializing with API key...")
            success = manager.initialize_with_api_key(
                api_key=api_key,
                budget_limit=1.0,
                default_model="gpt-4o-mini"
            )
            
            if success:
                self.log_to_output("✅ Manager initialization successful")
                
                # Step 5: Check post-initialization state
                self.log_to_output("5. Checking post-initialization state...")
                self.log_to_output(f"   - Initialized: {manager.is_initialized}")
                self.log_to_output(f"   - Cost guard: {manager.cost_guard is not None}")
                self.log_to_output(f"   - LLM client: {manager.llm_client is not None}")
                self.log_to_output(f"   - Field detector: {manager.field_detector is not None}")
                self.log_to_output(f"   - Cell resolver: {manager.cell_resolver is not None}")
                
                # Step 6: Test usage summary
                self.log_to_output("6. Testing usage summary...")
                usage = manager.get_usage_summary()
                self.log_to_output(f"   - Total cost: ${usage.get('total_cost', 0):.4f}")
                self.log_to_output(f"   - Initialized: {usage.get('initialized', False)}")
                
                # Step 7: Test simple template analysis
                self.log_to_output("7. Testing template analysis...")
                
                # Create a mock template
                class MockTemplate:
                    def __init__(self):
                        self.content = "Budget template with {PI_Salary} and {Equipment_Cost}"
                
                template = MockTemplate()
                suggestions = manager.analyze_template_fields(template)
                self.log_to_output(f"✅ Template analysis complete: {len(suggestions)} suggestions")
                
                for suggestion in suggestions:
                    self.log_to_output(f"   - {suggestion.field_name}: {suggestion.description}")
                
                self.log_to_output("\n🎉 All manager tests passed!")
                
            else:
                self.log_to_output("❌ Manager initialization failed")
                
        except Exception as e:
            self.log_to_output(f"❌ LLM Manager test failed: {str(e)}")
            self.log_to_output("Full traceback:")
            self.log_to_output(traceback.format_exc())
    
    def test_individual_components(self):
        """Test each component individually."""
        self.log_to_output("\n🔬 Testing Individual Components")
        self.log_to_output("=" * 50)
        
        components = [
            ('cost_guard', 'CostGuard'),
            ('api_key_manager', 'APIKeyManager'),
            ('field_detector', 'FieldDetector'),
            ('cell_resolver', 'CellResolver')
        ]
        
        for module_name, class_name in components:
            try:
                self.log_to_output(f"Testing {module_name}...")
                module = __import__(module_name)
                cls = getattr(module, class_name)
                
                if module_name == 'cost_guard':
                    instance = cls(budget_limit_usd=1.0)
                elif module_name == 'api_key_manager':
                    instance = cls()
                else:
                    # field_detector and cell_resolver need an LLM client
                    # We'll test import only for now
                    self.log_to_output(f"✅ {module_name} class available")
                    continue
                
                self.log_to_output(f"✅ {module_name} created successfully")
                
            except Exception as e:
                self.log_to_output(f"❌ {module_name} failed: {str(e)}")


def main():
    """Run the debug demo."""
    app = QApplication(sys.argv)
    
    window = DebugLLMWindow()
    window.show()
    
    # Add a test button to menu bar
    menubar = window.menuBar()
    test_menu = menubar.addMenu("Debug")
    
    test_components_action = test_menu.addAction("Test Individual Components")
    test_components_action.triggered.connect(window.test_individual_components)
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

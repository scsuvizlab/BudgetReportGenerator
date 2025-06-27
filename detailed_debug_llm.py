"""
Detailed Debug LLM Demo

Very detailed debugging to catch the exact error in manager initialization.
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


class DetailedDebugWindow(QMainWindow):
    """Very detailed debug version."""
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the debug UI."""
        self.setWindowTitle("🔍 Detailed Debug LLM Integration")
        self.setMinimumSize(900, 700)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Title
        title = QLabel("🔍 Detailed Debug LLM Integration")
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin: 10px;")
        layout.addWidget(title)
        
        # API Key section
        api_group = QGroupBox("API Key Testing")
        api_layout = QVBoxLayout()
        
        api_input_layout = QHBoxLayout()
        self.api_key_input = QLineEdit()
        self.api_key_input.setPlaceholderText("sk-your-openai-api-key-here")
        self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        
        self.test_step_by_step_button = QPushButton("Debug Manager Step-by-Step")
        self.test_step_by_step_button.clicked.connect(self.debug_manager_step_by_step)
        
        self.test_components_button = QPushButton("Test All Components")
        self.test_components_button.clicked.connect(self.test_all_components)
        
        api_input_layout.addWidget(QLabel("API Key:"))
        api_input_layout.addWidget(self.api_key_input)
        api_input_layout.addWidget(self.test_step_by_step_button)
        api_input_layout.addWidget(self.test_components_button)
        
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
    
    def debug_manager_step_by_step(self):
        """Debug the manager initialization step by step."""
        api_key = self.api_key_input.text().strip()
        if not api_key:
            self.log_to_output("❌ Please enter an API key first")
            return
        
        self.log_to_output("🔧 Debugging LLM Integration Manager Step by Step")
        self.log_to_output("=" * 60)
        
        try:
            # Step 1: Import LLM Integration Manager
            self.log_to_output("STEP 1: Importing LLMIntegrationManager...")
            from llm_integration_manager import LLMIntegrationManager
            self.log_to_output("✅ LLMIntegrationManager imported successfully")
            
            # Step 2: Create manager instance
            self.log_to_output("\nSTEP 2: Creating manager instance...")
            manager = LLMIntegrationManager()
            self.log_to_output("✅ Manager instance created")
            
            # Step 3: Manually recreate the initialization process
            self.log_to_output("\nSTEP 3: Manual initialization process...")
            
            # 3a: Check if API key is valid
            self.log_to_output("3a. Checking API key format...")
            if not api_key or not api_key.strip():
                self.log_to_output("❌ API key is empty")
                return
            self.log_to_output("✅ API key has content")
            
            # 3b: Try to set API key in manager
            self.log_to_output("3b. Setting API key in manager...")
            try:
                success = manager.api_key_manager.set_api_key(api_key)
                if success:
                    self.log_to_output("✅ API key set successfully")
                else:
                    self.log_to_output("❌ Failed to set API key")
                    return
            except Exception as e:
                self.log_to_output(f"❌ Error setting API key: {str(e)}")
                self.log_to_output(traceback.format_exc())
                return
            
            # 3c: Create cost guard
            self.log_to_output("3c. Creating cost guard...")
            try:
                from cost_guard import CostGuard
                manager.cost_guard = CostGuard(budget_limit_usd=1.0)
                self.log_to_output("✅ Cost guard created")
            except Exception as e:
                self.log_to_output(f"❌ Error creating cost guard: {str(e)}")
                self.log_to_output(traceback.format_exc())
                return
            
            # 3d: Create LLM client
            self.log_to_output("3d. Creating LLM client...")
            try:
                from llm_client import LLMClient
                manager.llm_client = LLMClient(api_key, "gpt-4o-mini")
                self.log_to_output("✅ LLM client created")
            except Exception as e:
                self.log_to_output(f"❌ Error creating LLM client: {str(e)}")
                self.log_to_output(traceback.format_exc())
                return
            
            # 3e: Test API key validation
            self.log_to_output("3e. Validating API key...")
            try:
                is_valid = manager.llm_client.validate_api_key()
                if is_valid:
                    self.log_to_output("✅ API key validation successful")
                else:
                    self.log_to_output("❌ API key validation failed")
                    return
            except Exception as e:
                self.log_to_output(f"❌ Error validating API key: {str(e)}")
                self.log_to_output(traceback.format_exc())
                return
            
            # 3f: Create field detector
            self.log_to_output("3f. Creating field detector...")
            try:
                from field_detector import FieldDetector
                manager.field_detector = FieldDetector(manager.llm_client)
                self.log_to_output("✅ Field detector created")
            except Exception as e:
                self.log_to_output(f"❌ Error creating field detector: {str(e)}")
                self.log_to_output(traceback.format_exc())
                return
            
            # 3g: Create cell resolver
            self.log_to_output("3g. Creating cell resolver...")
            try:
                from cell_resolver import CellResolver
                manager.cell_resolver = CellResolver(manager.llm_client)
                self.log_to_output("✅ Cell resolver created")
            except Exception as e:
                self.log_to_output(f"❌ Error creating cell resolver: {str(e)}")
                self.log_to_output(traceback.format_exc())
                return
            
            # 3h: Set initialization flag
            self.log_to_output("3h. Setting initialization flag...")
            manager.is_initialized = True
            self.log_to_output("✅ Manager marked as initialized")
            
            self.log_to_output("\n🎉 Manual initialization completed successfully!")
            
            # Step 4: Test the manager
            self.log_to_output("\nSTEP 4: Testing initialized manager...")
            usage = manager.get_usage_summary()
            self.log_to_output(f"✅ Usage summary: {usage}")
            
        except Exception as e:
            self.log_to_output(f"❌ Critical error in step-by-step debug: {str(e)}")
            self.log_to_output("Full traceback:")
            self.log_to_output(traceback.format_exc())
    
    def test_all_components(self):
        """Test all components individually to find issues."""
        self.log_to_output("🔬 Testing All Components Individually")
        self.log_to_output("=" * 50)
        
        # Test each component
        components_to_test = [
            ('llm_client', 'LLMClient'),
            ('cost_guard', 'CostGuard'), 
            ('api_key_manager', 'APIKeyManager'),
            ('field_detector', 'FieldDetector'),
            ('cell_resolver', 'CellResolver'),
            ('llm_integration_manager', 'LLMIntegrationManager')
        ]
        
        api_key = self.api_key_input.text().strip()
        
        for module_name, class_name in components_to_test:
            self.log_to_output(f"\nTesting {module_name}.{class_name}...")
            try:
                # Import module
                self.log_to_output(f"  Importing {module_name}...")
                module = __import__(module_name)
                self.log_to_output(f"  ✅ {module_name} imported")
                
                # Get class
                self.log_to_output(f"  Getting {class_name} class...")
                cls = getattr(module, class_name)
                self.log_to_output(f"  ✅ {class_name} class found")
                
                # Try to create instance
                self.log_to_output(f"  Creating {class_name} instance...")
                if module_name == 'llm_client' and api_key:
                    instance = cls(api_key, "gpt-4o-mini")
                elif module_name == 'cost_guard':
                    instance = cls(budget_limit_usd=1.0)
                elif module_name == 'api_key_manager':
                    instance = cls()
                elif module_name in ['field_detector', 'cell_resolver']:
                    # These need an LLM client
                    if api_key:
                        from llm_client import LLMClient
                        llm_client = LLMClient(api_key, "gpt-4o-mini")
                        instance = cls(llm_client)
                    else:
                        self.log_to_output(f"  ⚠️ Skipping {class_name} (needs API key)")
                        continue
                elif module_name == 'llm_integration_manager':
                    instance = cls()
                else:
                    instance = cls()
                
                self.log_to_output(f"  ✅ {class_name} instance created successfully")
                
                # Test basic functionality if possible
                if hasattr(instance, '__dict__'):
                    attrs = [attr for attr in dir(instance) if not attr.startswith('_')]
                    self.log_to_output(f"  📋 Available methods: {len(attrs)} methods")
                
            except Exception as e:
                self.log_to_output(f"  ❌ {module_name}.{class_name} failed: {str(e)}")
                self.log_to_output(f"  Full error: {traceback.format_exc()}")
    
    def test_actual_manager_call(self):
        """Test the actual manager initialization call to see the real error."""
        api_key = self.api_key_input.text().strip()
        if not api_key:
            self.log_to_output("❌ Please enter an API key first")
            return
        
        self.log_to_output("🎯 Testing Actual Manager Call")
        self.log_to_output("=" * 40)
        
        try:
            from llm_integration_manager import LLMIntegrationManager
            manager = LLMIntegrationManager()
            
            self.log_to_output("Calling manager.initialize_with_api_key()...")
            
            # This is the actual call that's failing - let's catch the specific error
            success = manager.initialize_with_api_key(
                api_key=api_key,
                budget_limit=1.0,
                default_model="gpt-4o-mini"
            )
            
            if success:
                self.log_to_output("✅ Manager initialization successful!")
            else:
                self.log_to_output("❌ Manager initialization returned False")
                
        except Exception as e:
            self.log_to_output(f"❌ Manager initialization threw exception: {str(e)}")
            self.log_to_output("Full traceback:")
            self.log_to_output(traceback.format_exc())


def main():
    """Run the detailed debug demo."""
    app = QApplication(sys.argv)
    
    window = DetailedDebugWindow()
    window.show()
    
    # Add test button to menu bar
    menubar = window.menuBar()
    test_menu = menubar.addMenu("Debug")
    
    test_actual_action = test_menu.addAction("Test Actual Manager Call")
    test_actual_action.triggered.connect(window.test_actual_manager_call)
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

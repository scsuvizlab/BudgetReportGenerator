"""
Standalone LLM Demo

A simple standalone PyQt6 application to test LLM integration components.
This can be run independently to verify everything is working.
"""
import sys
import logging
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
    QWidget, QPushButton, QLabel, QTextEdit, QMessageBox,
    QGroupBox, QLineEdit, QProgressBar
)
from PyQt6.QtCore import Qt, QTimer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LLMDemoWindow(QMainWindow):
    """Simple demo window to test LLM components."""
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        
        # Test if we can import LLM modules
        self.llm_available = self.test_imports()
        self.update_status()
    
    def setup_ui(self):
        """Set up the demo UI."""
        self.setWindowTitle("LLM Integration Demo")
        self.setMinimumSize(800, 600)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Title
        title = QLabel("🤖 LLM Integration Demo")
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin: 10px;")
        layout.addWidget(title)
        
        # Status section
        self.status_group = QGroupBox("Component Status")
        status_layout = QVBoxLayout()
        
        self.status_text = QTextEdit()
        self.status_text.setMaximumHeight(200)
        self.status_text.setReadOnly(True)
        
        status_layout.addWidget(self.status_text)
        self.status_group.setLayout(status_layout)
        layout.addWidget(self.status_group)
        
        # API Key section
        api_group = QGroupBox("API Key Testing")
        api_layout = QVBoxLayout()
        
        api_input_layout = QHBoxLayout()
        self.api_key_input = QLineEdit()
        self.api_key_input.setPlaceholderText("sk-your-openai-api-key-here")
        self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        
        self.test_key_button = QPushButton("Test API Key")
        self.test_key_button.clicked.connect(self.test_api_key)
        
        api_input_layout.addWidget(QLabel("API Key:"))
        api_input_layout.addWidget(self.api_key_input)
        api_input_layout.addWidget(self.test_key_button)
        
        self.api_result = QLabel("Enter API key and click Test")
        
        api_layout.addLayout(api_input_layout)
        api_layout.addWidget(self.api_result)
        api_group.setLayout(api_layout)
        layout.addWidget(api_group)
        
        # Demo section
        demo_group = QGroupBox("LLM Demo")
        demo_layout = QVBoxLayout()
        
        button_layout = QHBoxLayout()
        
        self.demo_button = QPushButton("Run Full Demo")
        self.demo_button.clicked.connect(self.run_demo)
        self.demo_button.setEnabled(False)
        
        self.cost_demo_button = QPushButton("Test Cost Tracking")
        self.cost_demo_button.clicked.connect(self.test_cost_tracking)
        
        button_layout.addWidget(self.demo_button)
        button_layout.addWidget(self.cost_demo_button)
        
        self.demo_output = QTextEdit()
        self.demo_output.setReadOnly(True)
        self.demo_output.setPlaceholderText("Demo output will appear here...")
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        demo_layout.addLayout(button_layout)
        demo_layout.addWidget(self.demo_output)
        demo_layout.addWidget(self.progress_bar)
        demo_group.setLayout(demo_layout)
        layout.addWidget(demo_group)
        
        # Instructions
        instructions = QLabel("""
Instructions:
1. First, check if all components imported correctly in the status section
2. If components are available, enter your OpenAI API key and test it
3. Run the full demo to test LLM functionality
4. The cost tracking demo works without an API key
        """)
        instructions.setStyleSheet("color: gray; font-size: 11px; margin: 10px;")
        layout.addWidget(instructions)
    
    def test_imports(self):
        """Test if we can import all LLM components."""
        status_lines = []
        all_available = True
        
        # Test basic dependencies
        try:
            import openai
            status_lines.append("✅ openai library available")
        except ImportError as e:
            status_lines.append(f"❌ openai library missing: {e}")
            all_available = False
        
        try:
            import tiktoken
            status_lines.append("✅ tiktoken library available")
        except ImportError as e:
            status_lines.append(f"❌ tiktoken library missing: {e}")
            all_available = False
        
        try:
            import keyring
            status_lines.append("✅ keyring library available")
        except ImportError as e:
            status_lines.append(f"❌ keyring library missing: {e}")
            all_available = False
        
        # Test our LLM components
        components = [
            'llm_client', 'cost_guard', 'api_key_manager', 
            'field_detector', 'cell_resolver', 'llm_integration_manager'
        ]
        
        for component in components:
            try:
                __import__(component)
                status_lines.append(f"✅ {component}.py available")
            except ImportError as e:
                status_lines.append(f"❌ {component}.py missing: {e}")
                all_available = False
        
        self.status_text.setPlainText("\n".join(status_lines))
        return all_available
    
    def update_status(self):
        """Update status display."""
        if self.llm_available:
            self.status_group.setTitle("Component Status - ✅ All Components Available")
            self.test_key_button.setEnabled(True)
        else:
            self.status_group.setTitle("Component Status - ❌ Missing Components")
            self.test_key_button.setEnabled(False)
            self.demo_output.setPlainText(
                "Some components are missing. Please check:\n"
                "1. Install dependencies: pip install openai tiktoken keyring\n"
                "2. Make sure all .py files are in the same directory\n"
                "3. Check the status section above for specific missing items"
            )
    
    def test_api_key(self):
        """Test the entered API key."""
        if not self.llm_available:
            QMessageBox.warning(self, "Components Missing", "Cannot test API key - missing components.")
            return
        
        api_key = self.api_key_input.text().strip()
        if not api_key:
            QMessageBox.warning(self, "No API Key", "Please enter an API key.")
            return
        
        self.test_key_button.setEnabled(False)
        self.test_key_button.setText("Testing...")
        self.api_result.setText("Testing API key...")
        
        try:
            # Import and test
            from llm_client import LLMClient
            
            client = LLMClient(api_key, "gpt-4o-mini")
            is_valid = client.validate_api_key()
            
            if is_valid:
                self.api_result.setText("✅ API key is valid!")
                self.api_result.setStyleSheet("color: green;")
                self.demo_button.setEnabled(True)
            else:
                self.api_result.setText("❌ API key validation failed")
                self.api_result.setStyleSheet("color: red;")
                self.demo_button.setEnabled(False)
        
        except Exception as e:
            self.api_result.setText(f"❌ Error testing API key: {str(e)}")
            self.api_result.setStyleSheet("color: red;")
            self.demo_button.setEnabled(False)
        
        finally:
            self.test_key_button.setEnabled(True)
            self.test_key_button.setText("Test API Key")
    
    def test_cost_tracking(self):
        """Test cost tracking without API calls."""
        if not self.llm_available:
            QMessageBox.warning(self, "Components Missing", "Cannot test cost tracking - missing components.")
            return
        
        try:
            from cost_guard import CostGuard
            
            output = []
            output.append("🧪 Testing Cost Tracking Component")
            output.append("=" * 40)
            
            # Create cost guard
            guard = CostGuard(budget_limit_usd=2.0, warning_threshold=0.8)
            output.append(f"✅ Created cost guard with $2.00 limit")
            
            # Test operations
            operations = [
                (0.50, 1000, "gpt-4o-mini", "template_analysis"),
                (0.75, 1500, "gpt-4o-mini", "field_detection"),
                (0.60, 1200, "gpt-4o-mini", "cell_resolution"),
                (0.80, 2000, "gpt-4o", "complex_analysis")  # This should be rejected
            ]
            
            for cost, tokens, model, operation in operations:
                if guard.check_affordability(cost):
                    guard.record_cost(cost, tokens, model, operation)
                    output.append(f"✅ ${cost:.2f} {operation} - Approved")
                else:
                    output.append(f"❌ ${cost:.2f} {operation} - Rejected (budget exceeded)")
            
            # Show final state
            breakdown = guard.get_cost_breakdown()
            output.append("")
            output.append("📊 Final State:")
            output.append(f"Total cost: ${breakdown['total_cost']:.4f}")
            output.append(f"Budget remaining: ${breakdown['budget_remaining']:.4f}")
            output.append(f"Utilization: {breakdown['budget_utilization']:.1%}")
            output.append(f"Operations: {breakdown['total_operations']}")
            output.append(f"Warnings issued: {breakdown['warnings_issued']}")
            
            self.demo_output.setPlainText("\n".join(output))
            
        except Exception as e:
            self.demo_output.setPlainText(f"❌ Cost tracking test failed: {str(e)}")
    
    def run_demo(self):
        """Run the full LLM demo."""
        api_key = self.api_key_input.text().strip()
        if not api_key:
            QMessageBox.warning(self, "No API Key", "Please enter and test an API key first.")
            return
        
        self.demo_button.setEnabled(False)
        self.demo_button.setText("Running Demo...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        
        try:
            from llm_integration_manager import LLMIntegrationManager
            
            output = []
            output.append("🤖 Running Full LLM Integration Demo")
            output.append("=" * 50)
            
            # Initialize manager
            manager = LLMIntegrationManager()
            success = manager.initialize_with_api_key(api_key, budget_limit=1.0)
            
            if success:
                output.append("✅ LLM Integration Manager initialized")
                
                # Test template analysis
                output.append("\n📝 Testing Template Analysis...")
                
                # Create a simple template for testing
                class MockTemplate:
                    def __init__(self):
                        self.content = """
                        Budget Justification Template
                        
                        Principal Investigator Salary: {PI_Salary_Total}
                        Equipment Costs: {Equipment_Total}
                        Travel Expenses: {Travel_Domestic_Total}
                        Supplies: {Supplies_Total}
                        
                        Total Project Cost: {Total_Budget}
                        """
                
                template = MockTemplate()
                suggestions = manager.analyze_template_fields(template)
                
                output.append(f"✅ Template analysis complete: {len(suggestions)} field suggestions")
                
                for suggestion in suggestions[:3]:  # Show first 3
                    output.append(f"  • {suggestion.field_name}: {suggestion.description}")
                
                if len(suggestions) > 3:
                    output.append(f"  • ... and {len(suggestions) - 3} more")
                
                # Show usage stats
                usage = manager.get_usage_summary()
                output.append(f"\n💰 Usage Statistics:")
                output.append(f"Total cost: ${usage['total_cost']:.4f}")
                output.append(f"Operations: {usage['total_operations']}")
                output.append(f"Field suggestions: {usage['field_suggestions']}")
                
                output.append("\n🎉 Demo completed successfully!")
                
            else:
                output.append("❌ Failed to initialize LLM manager")
            
            self.demo_output.setPlainText("\n".join(output))
            
        except Exception as e:
            self.demo_output.setPlainText(f"❌ Demo failed: {str(e)}\n\nPlease check that all components are properly installed.")
        
        finally:
            self.demo_button.setEnabled(True)
            self.demo_button.setText("Run Full Demo")
            self.progress_bar.setVisible(False)


def main():
    """Main function to run the demo."""
    app = QApplication(sys.argv)
    
    window = LLMDemoWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

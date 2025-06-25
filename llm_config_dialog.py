"""
LLM Configuration Dialog - Complete Fixed Version

Fixed to properly integrate with session state instead of directly with LLM manager.
"""
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QCheckBox,
    QGroupBox, QFormLayout, QTabWidget, QWidget, QProgressBar,
    QMessageBox, QFileDialog, QComboBox, QFrame, QTextBrowser
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont, QPalette

logger = logging.getLogger(__name__)


class LLMConfigDialog(QDialog):
    """
    Dialog for configuring LLM integration settings.
    
    FIXED: Now works properly with session state instead of LLM manager directly.
    """
    
    # Signal emitted when LLM is successfully configured
    llm_configured = pyqtSignal(bool)  # True if enabled, False if disabled
    
    def __init__(self, session_state=None, parent=None):
        """
        Initialize LLM configuration dialog.
        
        Args:
            session_state: Session state instance (CHANGED from llm_manager)
            parent: Parent widget
        """
        super().__init__(parent)
        self.session = session_state  # CHANGED: Store session instead of llm_manager
        
        # Initialize timers as None first
        self.usage_timer = None
        self.key_timer = None
        
        self.setup_ui()
        self.setup_timers()
        self.load_current_settings()
    
    def setup_ui(self):
        """Set up the dialog UI."""
        self.setWindowTitle("LLM Configuration")
        self.setMinimumSize(600, 500)
        self.setModal(True)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        
        # API Configuration Tab
        self.api_tab = self.create_api_tab()
        self.tab_widget.addTab(self.api_tab, "API Configuration")
        
        # Settings Tab
        self.settings_tab = self.create_settings_tab()
        self.tab_widget.addTab(self.settings_tab, "Model Settings")
        
        # Usage Tab
        self.usage_tab = self.create_usage_tab()
        self.tab_widget.addTab(self.usage_tab, "Usage & Costs")
        
        # Import/Export Tab
        self.import_tab = self.create_import_tab()
        self.tab_widget.addTab(self.import_tab, "Import/Export")
        
        # Main layout
        layout = QVBoxLayout()
        layout.addWidget(self.tab_widget)
        
        # Status bar
        self.status_label = QLabel("Configure your OpenAI API key to enable LLM features.")
        self.status_label.setWordWrap(True)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.test_button = QPushButton("Test Connection")
        self.test_button.clicked.connect(self.test_connection)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        
        self.save_button = QPushButton("Save & Enable")
        self.save_button.clicked.connect(self.save_and_enable)
        self.save_button.setDefault(True)
        
        button_layout.addWidget(self.test_button)
        button_layout.addStretch()
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.save_button)
        
        layout.addWidget(self.status_label)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def create_api_tab(self) -> QWidget:
        """Create the API configuration tab."""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # API Key Section
        api_group = QGroupBox("OpenAI API Key")
        api_layout = QFormLayout()
        
        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_input.setPlaceholderText("sk-...")
        self.api_key_input.textChanged.connect(self.on_api_key_changed)
        
        self.show_key_checkbox = QCheckBox("Show API Key")
        self.show_key_checkbox.toggled.connect(self.toggle_key_visibility)
        
        api_layout.addRow("API Key:", self.api_key_input)
        api_layout.addRow("", self.show_key_checkbox)
        
        api_group.setLayout(api_layout)
        
        # Key Status Section
        status_group = QGroupBox("Key Status")
        status_layout = QFormLayout()
        
        self.key_format_label = QLabel("Unknown")
        self.key_length_label = QLabel("0 characters")
        self.validation_label = QLabel("Not validated")
        
        status_layout.addRow("Format:", self.key_format_label)
        status_layout.addRow("Length:", self.key_length_label)
        status_layout.addRow("Validation:", self.validation_label)
        
        status_group.setLayout(status_layout)
        
        # Help Section
        help_group = QGroupBox("Getting Your API Key")
        help_layout = QVBoxLayout()
        
        help_text = QTextBrowser()
        help_text.setMaximumHeight(120)
        help_text.setHtml("""
        <p><b>To get your OpenAI API key:</b></p>
        <ol>
        <li>Visit <a href="https://platform.openai.com/api-keys">platform.openai.com/api-keys</a></li>
        <li>Sign in to your OpenAI account</li>
        <li>Click "Create new secret key"</li>
        <li>Copy the key (starts with "sk-")</li>
        <li>Paste it above</li>
        </ol>
        <p><b>Note:</b> You'll need billing set up to use the API.</p>
        """)
        
        help_layout.addWidget(help_text)
        help_group.setLayout(help_layout)
        
        layout.addWidget(api_group)
        layout.addWidget(status_group)
        layout.addWidget(help_group)
        layout.addStretch()
        
        tab.setLayout(layout)
        return tab
    
    def create_settings_tab(self) -> QWidget:
        """Create the model settings tab."""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Model Settings
        model_group = QGroupBox("Model Configuration")
        model_layout = QFormLayout()
        
        self.model_combo = QComboBox()
        self.model_combo.addItems(["gpt-4o-mini", "gpt-4o"])
        self.model_combo.setCurrentText("gpt-4o-mini")
        
        # Add cost information
        cost_info = QLabel("gpt-4o-mini: $0.15/$0.60 per 1M tokens (in/out)\ngpt-4o: $5.00/$15.00 per 1M tokens (in/out)")
        cost_info.setStyleSheet("color: gray; font-size: 10px;")
        
        model_layout.addRow("Default Model:", self.model_combo)
        model_layout.addRow("", cost_info)
        
        model_group.setLayout(model_layout)
        
        # Budget Settings
        budget_group = QGroupBox("Budget Control")
        budget_layout = QFormLayout()
        
        self.cost_limit_input = QDoubleSpinBox()
        self.cost_limit_input.setRange(0.10, 100.0)
        self.cost_limit_input.setSingleStep(0.50)
        self.cost_limit_input.setDecimals(2)
        self.cost_limit_input.setSuffix(" USD")
        self.cost_limit_input.setValue(5.0)
        
        warning_info = QLabel("Warning will be shown at 80% of limit")
        warning_info.setStyleSheet("color: gray; font-size: 10px;")
        
        budget_layout.addRow("Cost Limit:", self.cost_limit_input)
        budget_layout.addRow("", warning_info)
        
        budget_group.setLayout(budget_layout)
        
        # Advanced Settings
        advanced_group = QGroupBox("Advanced Options")
        advanced_layout = QFormLayout()
        
        self.auto_improve_checkbox = QCheckBox("Automatically improve low-confidence mappings")
        self.auto_improve_checkbox.setChecked(True)
        
        self.heuristics_first_checkbox = QCheckBox("Try heuristic matching before LLM")
        self.heuristics_first_checkbox.setChecked(True)
        
        advanced_layout.addRow(self.auto_improve_checkbox)
        advanced_layout.addRow(self.heuristics_first_checkbox)
        
        advanced_group.setLayout(advanced_layout)
        
        layout.addWidget(model_group)
        layout.addWidget(budget_group)
        layout.addWidget(advanced_group)
        layout.addStretch()
        
        tab.setLayout(layout)
        return tab
    
    def create_usage_tab(self) -> QWidget:
        """Create the usage monitoring tab."""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Current Session
        session_group = QGroupBox("Current Session")
        session_layout = QFormLayout()
        
        self.total_cost_label = QLabel("$0.00")
        self.remaining_budget_label = QLabel("$0.00")
        self.operations_label = QLabel("0")
        self.tokens_label = QLabel("0")
        
        session_layout.addRow("Total Cost:", self.total_cost_label)
        session_layout.addRow("Remaining Budget:", self.remaining_budget_label)
        session_layout.addRow("Operations:", self.operations_label)
        session_layout.addRow("Tokens Used:", self.tokens_label)
        
        session_group.setLayout(session_layout)
        
        # Usage Breakdown
        breakdown_group = QGroupBox("Usage Breakdown")
        breakdown_layout = QVBoxLayout()
        
        self.usage_breakdown = QTextBrowser()
        self.usage_breakdown.setMaximumHeight(150)
        
        breakdown_layout.addWidget(self.usage_breakdown)
        breakdown_group.setLayout(breakdown_layout)
        
        # Budget Progress
        progress_group = QGroupBox("Budget Progress")
        progress_layout = QVBoxLayout()
        
        self.budget_progress = QProgressBar()
        self.budget_progress.setRange(0, 100)
        self.budget_progress.setValue(0)
        
        self.reset_button = QPushButton("Reset Usage")
        self.reset_button.clicked.connect(self.reset_usage)
        
        progress_layout.addWidget(self.budget_progress)
        progress_layout.addWidget(self.reset_button)
        progress_group.setLayout(progress_layout)
        
        layout.addWidget(session_group)
        layout.addWidget(breakdown_group)
        layout.addWidget(progress_group)
        layout.addStretch()
        
        tab.setLayout(layout)
        return tab
    
    def create_import_tab(self) -> QWidget:
        """Create the import/export tab."""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # JSON Import/Export
        json_group = QGroupBox("Configuration File")
        json_layout = QVBoxLayout()
        
        import_layout = QHBoxLayout()
        self.import_button = QPushButton("Import from JSON")
        self.import_button.clicked.connect(self.import_config)
        import_layout.addWidget(self.import_button)
        import_layout.addStretch()
        
        export_layout = QHBoxLayout()
        self.export_button = QPushButton("Export to JSON")
        self.export_button.clicked.connect(self.export_config)
        
        self.include_key_checkbox = QCheckBox("Include API key in export")
        export_layout.addWidget(self.export_button)
        export_layout.addWidget(self.include_key_checkbox)
        export_layout.addStretch()
        
        json_layout.addLayout(import_layout)
        json_layout.addLayout(export_layout)
        json_group.setLayout(json_layout)
        
        # Example Configuration
        example_group = QGroupBox("Example Configuration File")
        example_layout = QVBoxLayout()
        
        example_text = QTextBrowser()
        example_text.setMaximumHeight(200)
        example_config = {
            "openai_api_key": "sk-your-actual-key-here",
            "default_model": "gpt-4o-mini",
            "cost_limit_usd": 5.0
        }
        example_text.setPlainText(json.dumps(example_config, indent=2))
        
        example_layout.addWidget(example_text)
        example_group.setLayout(example_layout)
        
        layout.addWidget(json_group)
        layout.addWidget(example_group)
        layout.addStretch()
        
        tab.setLayout(layout)
        return tab
    
    def setup_timers(self):
        """Set up update timers."""
        try:
            # Update usage display every 3 seconds
            self.usage_timer = QTimer()
            self.usage_timer.timeout.connect(self.update_usage_display)
            self.usage_timer.start(3000)
            
            # Update key status when typing stops
            self.key_timer = QTimer()
            self.key_timer.setSingleShot(True)
            self.key_timer.timeout.connect(self.update_key_status)
            
            logger.info("Timers initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize timers: {e}")
            # Create dummy timers to prevent crashes
            self.usage_timer = QTimer()
            self.key_timer = QTimer()
    
    def load_current_settings(self):
        """Load current settings into the dialog."""
        try:
            # FIXED: Load from session state instead of LLM manager
            if self.session:
                # Load API key from session config
                if self.session.config.openai_api_key:
                    self.api_key_input.setText(self.session.config.openai_api_key)
                
                # Load other settings
                self.model_combo.setCurrentText(self.session.config.default_model)
                self.cost_limit_input.setValue(self.session.config.cost_limit_usd)
                self.auto_improve_checkbox.setChecked(self.session.config.auto_improve_enabled)
                self.heuristics_first_checkbox.setChecked(self.session.config.heuristics_first)
            
            # Update key status
            self.update_key_status()
        except Exception as e:
            logger.error(f"Failed to load current settings: {e}")
    
    def on_api_key_changed(self):
        """Handle API key text changes."""
        try:
            if self.key_timer:
                self.key_timer.start(1000)  # Update after 1 second of no typing
        except Exception as e:
            logger.error(f"Error in on_api_key_changed: {e}")
            # Immediately update status if timer fails
            self.update_key_status()
    
    def toggle_key_visibility(self):
        """Toggle API key visibility."""
        try:
            if self.show_key_checkbox.isChecked():
                self.api_key_input.setEchoMode(QLineEdit.EchoMode.Normal)
            else:
                self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        except Exception as e:
            logger.error(f"Error toggling key visibility: {e}")
    
    def update_key_status(self):
        """Update API key status display."""
        try:
            api_key = self.api_key_input.text().strip()
            
            if not api_key:
                self.key_format_label.setText("No key entered")
                self.key_length_label.setText("0 characters")
                self.validation_label.setText("Not validated")
                return
            
            # Check format
            if api_key.startswith("sk-") and len(api_key) > 40:
                self.key_format_label.setText("✅ Valid format")
                self.key_format_label.setStyleSheet("color: green;")
            else:
                self.key_format_label.setText("❌ Invalid format")
                self.key_format_label.setStyleSheet("color: red;")
            
            self.key_length_label.setText(f"{len(api_key)} characters")
            
            # Validation status - FIXED: Check session state
            if self.session and self.session.llm_enabled:
                self.validation_label.setText("✅ Validated")
                self.validation_label.setStyleSheet("color: green;")
            else:
                self.validation_label.setText("❓ Not validated")
                self.validation_label.setStyleSheet("color: orange;")
        except Exception as e:
            logger.error(f"Error updating key status: {e}")
    
    def test_connection(self):
        """Test the API key connection."""
        api_key = self.api_key_input.text().strip()
        
        if not api_key:
            QMessageBox.warning(self, "No API Key", "Please enter an API key first.")
            return
        
        self.test_button.setEnabled(False)
        self.test_button.setText("Testing...")
        
        try:
            # FIXED: Test using session's validate_api_key method
            if self.session and hasattr(self.session, 'validate_api_key'):
                is_valid = self.session.validate_api_key(api_key)
                
                if is_valid:
                    QMessageBox.information(self, "Success", "API key is valid!")
                    self.status_label.setText("✅ API key validated successfully.")
                    self.validation_label.setText("✅ Validated")
                    self.validation_label.setStyleSheet("color: green;")
                else:
                    QMessageBox.warning(self, "Invalid Key", "API key validation failed.")
                    self.status_label.setText("❌ API key validation failed.")
            else:
                QMessageBox.warning(self, "No Session", "Session not available for testing.")
        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Connection test failed:\n{str(e)}")
            self.status_label.setText(f"❌ Test error: {str(e)}")
        
        finally:
            self.test_button.setEnabled(True)
            self.test_button.setText("Test Connection")
    
    def save_and_enable(self):
        """Save settings and enable LLM - FIXED VERSION."""
        api_key = self.api_key_input.text().strip()
        
        if not api_key:
            QMessageBox.warning(self, "No API Key", "Please enter an API key.")
            return
        
        try:
            # FIXED: Use session's initialize_llm method instead of LLM manager directly
            if self.session:
                # Update session configuration first
                self.session.config.openai_api_key = api_key
                self.session.config.default_model = self.model_combo.currentText()
                self.session.config.cost_limit_usd = self.cost_limit_input.value()
                self.session.config.auto_improve_enabled = self.auto_improve_checkbox.isChecked()
                self.session.config.heuristics_first = self.heuristics_first_checkbox.isChecked()
                
                # Initialize LLM through session state (this sets llm_enabled flag)
                success = self.session.initialize_llm(
                    api_key=api_key,
                    cost_limit=self.cost_limit_input.value()
                )
                
                if success:
                    QMessageBox.information(self, "Success", "LLM enabled successfully!")
                    self.llm_configured.emit(True)
                    self.accept()
                else:
                    QMessageBox.warning(self, "Failed", "Failed to initialize LLM client.")
            else:
                QMessageBox.warning(self, "No Session", "Session state not available.")
        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to enable LLM:\n{str(e)}")
            logger.error(f"LLM configuration error: {e}")
    
    def import_config(self):
        """Import configuration from JSON file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Import Configuration", "", "JSON Files (*.json)"
        )
        
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    config = json.load(f)
                
                if 'openai_api_key' in config:
                    self.api_key_input.setText(config['openai_api_key'])
                if 'default_model' in config:
                    self.model_combo.setCurrentText(config['default_model'])
                if 'cost_limit_usd' in config:
                    self.cost_limit_input.setValue(config['cost_limit_usd'])
                
                QMessageBox.information(self, "Success", "Configuration imported successfully.")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to import configuration:\n{str(e)}")
    
    def export_config(self):
        """Export configuration to JSON file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Configuration", "llm_config.json", "JSON Files (*.json)"
        )
        
        if file_path:
            try:
                config = {
                    'default_model': self.model_combo.currentText(),
                    'cost_limit_usd': self.cost_limit_input.value()
                }
                
                if self.include_key_checkbox.isChecked():
                    config['openai_api_key'] = self.api_key_input.text().strip()
                else:
                    config['openai_api_key'] = "[REDACTED - Import your key separately]"
                
                with open(file_path, 'w') as f:
                    json.dump(config, f, indent=2)
                
                QMessageBox.information(self, "Success", "Configuration exported successfully.")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export configuration:\n{str(e)}")
    
    def update_usage_display(self):
        """Update the usage display - FIXED VERSION."""
        try:
            # FIXED: Use session state instead of LLM manager
            if not self.session or not self.session.llm_enabled:
                return
            
            usage = self.session.get_llm_usage_summary()
            
            # Update labels
            self.total_cost_label.setText(f"${usage.get('total_cost', 0.0):.4f}")
            self.remaining_budget_label.setText(f"${usage.get('budget_remaining', 0.0):.4f}")
            self.operations_label.setText(str(usage.get('total_operations', 0)))
            self.tokens_label.setText(f"{usage.get('llm_tokens', 0):,}")
            
            # Update progress bar
            utilization = usage.get('budget_utilization', 0.0)
            self.budget_progress.setValue(int(utilization * 100))
            
            # Update breakdown text
            breakdown_text = f"""Operations: {usage.get('total_operations', 0)}
Cost: ${usage.get('total_cost', 0.0):.4f}
Tokens: {usage.get('llm_tokens', 0):,}
Budget: {utilization:.1%} used"""
            
            self.usage_breakdown.setPlainText(breakdown_text)
            
        except Exception as e:
            logger.error(f"Failed to update usage display: {e}")
    
    def reset_usage(self):
        """Reset usage tracking - FIXED VERSION."""
        reply = QMessageBox.question(
            self,
            "Reset Usage",
            "This will reset all usage tracking. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                # FIXED: Use session method instead of LLM manager
                if self.session and hasattr(self.session, 'llm_manager') and self.session.llm_manager:
                    self.session.llm_manager.reset_session()
                    # Also reset session-level tracking
                    self.session.total_tokens_used = 0
                    self.session.total_cost_usd = 0.0
                    
                    self.update_usage_display()
                    QMessageBox.information(self, "Reset", "Usage tracking has been reset.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to reset usage: {str(e)}")
    
    def closeEvent(self, event):
        """Handle dialog close."""
        try:
            # Stop timers when closing
            if self.usage_timer:
                self.usage_timer.stop()
            if self.key_timer:
                self.key_timer.stop()
        except Exception as e:
            logger.error(f"Error stopping timers: {e}")
        
        event.accept()


# Example usage
if __name__ == "__main__":
    import sys
    from PyQt6.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    
    # Create dialog without session (for testing)
    dialog = LLMConfigDialog()
    
    # Connect signal
    dialog.llm_configured.connect(lambda enabled: print(f"LLM configured: {enabled}"))
    
    # Show dialog
    result = dialog.exec()
    print(f"Dialog result: {result}")
    
    sys.exit(0)
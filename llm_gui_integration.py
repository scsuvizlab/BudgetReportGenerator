"""
GUI Integration for LLM Features

Enhanced PyQt6 components for OpenAI API key management and LLM analysis display.
These should be integrated into the existing main_window.py.
"""
import json
from pathlib import Path
from typing import Optional, Dict, Any

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QLineEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QCheckBox,
    QGroupBox, QFormLayout, QTabWidget, QWidget, QProgressBar,
    QMessageBox, QFileDialog, QTableWidget, QTableWidgetItem,
    QHeaderView, QComboBox, QSlider, QFrame, QSplitter
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QSettings
from PyQt6.QtGui import QFont, QIcon, QPixmap, QPalette


class APIKeyDialog(QDialog):
    """Dialog for OpenAI API key configuration and LLM settings."""
    
    def __init__(self, session_state, parent=None):
        super().__init__(parent)
        self.session = session_state
        self.settings = QSettings("BudgetTool", "BudgetJustification")
        
        self.setup_ui()
        self.load_settings()
    
    def setup_ui(self):
        """Set up the API key dialog UI."""
        self.setWindowTitle("LLM Configuration")
        self.setMinimumSize(500, 400)
        self.setModal(True)
        
        layout = QVBoxLayout()
        
        # API Key Section
        api_group = QGroupBox("OpenAI API Configuration")
        api_layout = QFormLayout()
        
        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_input.setPlaceholderText("sk-...")
        
        self.show_key_checkbox = QCheckBox("Show API Key")
        self.show_key_checkbox.toggled.connect(self.toggle_key_visibility)
        
        self.test_button = QPushButton("Test Connection")
        self.test_button.clicked.connect(self.test_api_key)
        
        api_layout.addRow("API Key:", self.api_key_input)
        api_layout.addRow("", self.show_key_checkbox)
        api_layout.addRow("", self.test_button)
        
        api_group.setLayout(api_layout)
        
        # Model Settings Section
        model_group = QGroupBox("Model Settings")
        model_layout = QFormLayout()
        
        self.model_combo = QComboBox()
        self.model_combo.addItems(["gpt-4o-mini", "gpt-4o"])
        self.model_combo.setCurrentText(self.session.config.default_model)
        
        self.cost_limit_input = QDoubleSpinBox()
        self.cost_limit_input.setRange(0.10, 50.0)
        self.cost_limit_input.setSingleStep(0.50)
        self.cost_limit_input.setDecimals(2)
        self.cost_limit_input.setSuffix(" USD")
        self.cost_limit_input.setValue(self.session.config.cost_limit_usd)
        
        model_layout.addRow("Default Model:", self.model_combo)
        model_layout.addRow("Cost Limit:", self.cost_limit_input)
        
        model_group.setLayout(model_layout)
        
        # Usage Display Section
        usage_group = QGroupBox("Current Usage")
        usage_layout = QFormLayout()
        
        self.usage_label = QLabel("Not initialized")
        self.remaining_budget_label = QLabel("$0.00")
        self.calls_label = QLabel("0")
        
        usage_layout.addRow("Total Cost:", self.usage_label)
        usage_layout.addRow("Remaining Budget:", self.remaining_budget_label)
        usage_layout.addRow("API Calls:", self.calls_label)
        
        usage_group.setLayout(usage_layout)
        
        # File Import Section
        import_group = QGroupBox("Configuration Import/Export")
        import_layout = QHBoxLayout()
        
        self.import_button = QPushButton("Import from JSON")
        self.import_button.clicked.connect(self.import_config)
        
        self.export_button = QPushButton("Export to JSON")
        self.export_button.clicked.connect(self.export_config)
        
        import_layout.addWidget(self.import_button)
        import_layout.addWidget(self.export_button)
        
        import_group.setLayout(import_layout)
        
        # Status Section
        self.status_label = QLabel("Enter your OpenAI API key to enable LLM features.")
        self.status_label.setWordWrap(True)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        
        self.save_button = QPushButton("Save & Enable LLM")
        self.save_button.clicked.connect(self.save_and_enable)
        self.save_button.setDefault(True)
        
        button_layout.addWidget(self.cancel_button)
        button_layout.addStretch()
        button_layout.addWidget(self.save_button)
        
        # Layout assembly
        layout.addWidget(api_group)
        layout.addWidget(model_group)
        layout.addWidget(usage_group)
        layout.addWidget(import_group)
        layout.addWidget(self.status_label)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        
        # Update usage display timer
        self.usage_timer = QTimer()
        self.usage_timer.timeout.connect(self.update_usage_display)
        self.usage_timer.start(2000)  # Update every 2 seconds
    
    def load_settings(self):
        """Load settings from QSettings."""
        api_key = self.settings.value("openai_api_key", "")
        self.api_key_input.setText(api_key)
        
        model = self.settings.value("default_model", "gpt-4o-mini")
        self.model_combo.setCurrentText(model)
        
        cost_limit = self.settings.value("cost_limit", 5.0, type=float)
        self.cost_limit_input.setValue(cost_limit)
    
    def toggle_key_visibility(self):
        """Toggle API key visibility."""
        if self.show_key_checkbox.isChecked():
            self.api_key_input.setEchoMode(QLineEdit.EchoMode.Normal)
        else:
            self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
    
    def test_api_key(self):
        """Test the API key connection."""
        api_key = self.api_key_input.text().strip()
        
        if not api_key:
            QMessageBox.warning(self, "No API Key", "Please enter an API key first.")
            return
        
        self.test_button.setEnabled(False)
        self.test_button.setText("Testing...")
        
        try:
            # Use session's validate_api_key method
            if hasattr(self.session, 'validate_api_key'):
                is_valid = self.session.validate_api_key(api_key)
            else:
                # Fallback test
                from llm_client import LLMClient
                test_client = LLMClient(api_key, "gpt-4o-mini")
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Reply with just 'OK'"}
                ]
                response = test_client.call_llm(messages, max_tokens=5)
                is_valid = "ok" in response.content.lower()
            
            if is_valid:
                QMessageBox.information(self, "Success", "API key is valid!")
                self.status_label.setText("✅ API key validated successfully.")
            else:
                QMessageBox.warning(self, "Invalid Key", "API key validation failed.")
                self.status_label.setText("❌ API key validation failed.")
        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"API key test failed:\n{str(e)}")
            self.status_label.setText(f"❌ Test error: {str(e)}")
        
        finally:
            self.test_button.setEnabled(True)
            self.test_button.setText("Test Connection")
    
    def save_and_enable(self):
        """Save settings and enable LLM."""
        api_key = self.api_key_input.text().strip()
        
        if not api_key:
            QMessageBox.warning(self, "No API Key", "Please enter an API key.")
            return
        
        # Update session configuration
        self.session.config.openai_api_key = api_key
        self.session.config.default_model = self.model_combo.currentText()
        self.session.config.cost_limit_usd = self.cost_limit_input.value()
        
        # Initialize LLM
        if hasattr(self.session, 'initialize_llm'):
            if self.session.initialize_llm(api_key):
                self.save_settings_to_file()
                QMessageBox.information(self, "Success", "LLM enabled successfully!")
                self.accept()
            else:
                QMessageBox.warning(self, "Failed", "Failed to initialize LLM client.")
        else:
            # For testing without full integration
            self.save_settings_to_file()
            QMessageBox.information(self, "Saved", "Settings saved. LLM integration pending.")
            self.accept()
    
    def save_settings_to_file(self):
        """Save settings to QSettings."""
        self.settings.setValue("openai_api_key", self.api_key_input.text().strip())
        self.settings.setValue("default_model", self.model_combo.currentText())
        self.settings.setValue("cost_limit", self.cost_limit_input.value())
    
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
                    'openai_api_key': self.api_key_input.text().strip(),
                    'default_model': self.model_combo.currentText(),
                    'cost_limit_usd': self.cost_limit_input.value()
                }
                
                with open(file_path, 'w') as f:
                    json.dump(config, f, indent=2)
                
                QMessageBox.information(self, "Success", "Configuration exported successfully.")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export configuration:\n{str(e)}")
    
    def update_usage_display(self):
        """Update the usage display."""
        if hasattr(self.session, 'get_llm_usage_summary'):
            usage = self.session.get_llm_usage_summary()
            
            if usage.get('enabled', False):
                self.usage_label.setText(f"${usage.get('total_cost', 0.0):.4f}")
                self.remaining_budget_label.setText(f"${usage.get('remaining_budget', 0.0):.4f}")
                self.calls_label.setText(str(usage.get('total_calls', 0)))
            else:
                self.usage_label.setText("Not initialized")
                self.remaining_budget_label.setText("$0.00")
                self.calls_label.setText("0")


class EnhancedFieldMappingPage:
    """Enhanced field mapping page with LLM analysis display.
    
    This shows the enhancements that should be added to the existing FieldMappingPage.
    """
    
    def setup_enhanced_ui(self):
        """Enhanced UI setup with LLM features."""
        # Add to existing setup_ui method
        
        # LLM Control Panel
        llm_group = QGroupBox("LLM Analysis")
        llm_layout = QHBoxLayout()
        
        self.llm_status_label = QLabel("LLM: Not initialized")
        
        self.configure_llm_button = QPushButton("Configure LLM")
        self.configure_llm_button.clicked.connect(self.configure_llm)
        
        self.analyze_button = QPushButton("Analyze with LLM")
        self.analyze_button.clicked.connect(self.run_llm_analysis)
        self.analyze_button.setEnabled(False)
        
        self.improve_button = QPushButton("Improve Low Confidence")
        self.improve_button.clicked.connect(self.improve_mappings)
        self.improve_button.setEnabled(False)
        
        self.cost_label = QLabel("Cost: $0.00")
        
        llm_layout.addWidget(self.llm_status_label)
        llm_layout.addWidget(self.configure_llm_button)
        llm_layout.addWidget(self.analyze_button)
        llm_layout.addWidget(self.improve_button)
        llm_layout.addStretch()
        llm_layout.addWidget(self.cost_label)
        
        llm_group.setLayout(llm_layout)
        
        # Add to existing layout (insert after controls_layout)
        self.layout().insertWidget(1, llm_group)
        
        # Enhance the existing table with LLM columns
        self.enhance_mapping_table()
    
    def enhance_mapping_table(self):
        """Add LLM-specific columns to the mapping table."""
        # Update table to include LLM analysis columns
        self.mapping_table.setColumnCount(8)  # Increased from 6
        self.mapping_table.setHorizontalHeaderLabels([
            "Field", "Current Value", "Source", "Confidence", "LLM Analysis", "Manual Value", "Notes"
        ])
        
        header = self.mapping_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch)
    
    def configure_llm(self):
        """Open LLM configuration dialog."""
        dialog = APIKeyDialog(self.session, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.update_llm_status()
    
    def update_llm_status(self):
        """Update LLM status display."""
        if hasattr(self.session, 'llm_enabled') and self.session.llm_enabled:
            self.llm_status_label.setText("LLM: ✅ Enabled")
            self.analyze_button.setEnabled(True)
            self.improve_button.setEnabled(True)
            
            # Update cost display
            if hasattr(self.session, 'get_llm_usage_summary'):
                usage = self.session.get_llm_usage_summary()
                self.cost_label.setText(f"Cost: ${usage.get('total_cost', 0.0):.4f}")
        else:
            self.llm_status_label.setText("LLM: ❌ Not enabled")
            self.analyze_button.setEnabled(False)
            self.improve_button.setEnabled(False)
            self.cost_label.setText("Cost: $0.00")
    
    def run_llm_analysis(self):
        """Run LLM analysis on all fields."""
        if not hasattr(self.session, 'enhanced_auto_match_fields'):
            QMessageBox.warning(self, "Not Available", "LLM analysis not available in this version.")
            return
        
        self.analyze_button.setEnabled(False)
        self.analyze_button.setText("Analyzing...")
        
        try:
            # Run enhanced auto-matching
            self.session.enhanced_auto_match_fields()
            
            # Update the table
            self.update_enhanced_content()
            
            QMessageBox.information(self, "Complete", "LLM analysis completed!")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"LLM analysis failed:\n{str(e)}")
        
        finally:
            self.analyze_button.setEnabled(True)
            self.analyze_button.setText("Analyze with LLM")
            self.update_llm_status()
    
    def improve_mappings(self):
        """Improve low-confidence mappings with LLM."""
        if not hasattr(self.session, 'improve_low_confidence_mappings'):
            QMessageBox.warning(self, "Not Available", "LLM improvement not available.")
            return
        
        self.improve_button.setEnabled(False)
        self.improve_button.setText("Improving...")
        
        try:
            improved_count = self.session.improve_low_confidence_mappings()
            
            if improved_count > 0:
                self.update_enhanced_content()
                QMessageBox.information(self, "Success", f"Improved {improved_count} mappings!")
            else:
                QMessageBox.information(self, "No Changes", "No mappings needed improvement.")
        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"LLM improvement failed:\n{str(e)}")
        
        finally:
            self.improve_button.setEnabled(True)
            self.improve_button.setText("Improve Low Confidence")
            self.update_llm_status()
    
    def update_enhanced_content(self):
        """Update table content with LLM analysis."""
        if not self.session.template or not self.session.budget:
            return
        
        mappings = self.session.mappings
        self.mapping_table.setRowCount(len(mappings))
        
        for row, (field_name, mapping) in enumerate(mappings.items()):
            # Existing columns
            field_item = QTableWidgetItem(field_name)
            field_item.setFlags(field_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.mapping_table.setItem(row, 0, field_item)
            
            value_item = QTableWidgetItem(mapping.display_value)
            value_item.setFlags(value_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.mapping_table.setItem(row, 1, value_item)
            
            source = "Manual" if mapping.is_manually_set else "Auto"
            # Check if this was enhanced by LLM
            if hasattr(self.session, 'llm_enhanced_mappings') and field_name in self.session.llm_enhanced_mappings:
                enhanced_match = self.session.llm_enhanced_mappings[field_name]
                source = f"{enhanced_match.source.title()}"
            
            source_item = QTableWidgetItem(source)
            source_item.setFlags(source_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.mapping_table.setItem(row, 2, source_item)
            
            confidence_item = QTableWidgetItem(f"{mapping.confidence:.2f}")
            confidence_item.setFlags(confidence_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.mapping_table.setItem(row, 3, confidence_item)
            
            # New LLM Analysis column
            llm_analysis = "N/A"
            if hasattr(self.session, 'llm_enhanced_mappings') and field_name in self.session.llm_enhanced_mappings:
                enhanced_match = self.session.llm_enhanced_mappings[field_name]
                llm_analysis = enhanced_match.reasoning[:100] + "..." if len(enhanced_match.reasoning) > 100 else enhanced_match.reasoning
            
            llm_item = QTableWidgetItem(llm_analysis)
            llm_item.setFlags(llm_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            llm_item.setToolTip(llm_analysis)  # Full text in tooltip
            self.mapping_table.setItem(row, 4, llm_item)
            
            # Manual value (editable)
            manual_value = mapping.manual_value if mapping.manual_value is not None else ""
            manual_item = QTableWidgetItem(str(manual_value))
            self.mapping_table.setItem(row, 5, manual_item)
            
            # Notes (editable)
            notes_item = QTableWidgetItem(mapping.notes)
            self.mapping_table.setItem(row, 6, notes_item)
        
        self.update_status()


# Usage instructions:
"""
TO INTEGRATE INTO EXISTING main_window.py:

1. Add APIKeyDialog class to main_window.py
2. Enhance the existing FieldMappingPage with the methods shown in EnhancedFieldMappingPage
3. Add LLM menu item to the main window menu bar
4. Call update_llm_status() when pages are loaded
5. Add LLM configuration option to the settings menu

Example menu integration:

def create_menu_bar(self):
    # ... existing menu code ...
    
    llm_menu = menubar.addMenu("LLM")
    
    configure_action = QAction("Configure OpenAI API", self)
    configure_action.triggered.connect(self.configure_llm)
    llm_menu.addAction(configure_action)
    
    analyze_action = QAction("Analyze Template", self)
    analyze_action.triggered.connect(self.analyze_with_llm)
    llm_menu.addAction(analyze_action)

def configure_llm(self):
    dialog = APIKeyDialog(self.wizard.session, self)
    dialog.exec()

def analyze_with_llm(self):
    # Trigger LLM analysis on current page
    current_page = self.wizard.currentPage()
    if hasattr(current_page, 'run_llm_analysis'):
        current_page.run_llm_analysis()
"""

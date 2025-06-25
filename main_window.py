"""
Main Window with LLM Integration - Complete Fixed Version

Fixed LLM configuration dialog integration to work properly with session state.
"""
import sys
import logging
import subprocess
import platform
from pathlib import Path
from typing import Optional, Dict, Any

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QWizard, QWizardPage, QLabel, QPushButton, QFileDialog, QTextEdit,
    QTableWidget, QTableWidgetItem, QHeaderView, QGroupBox, QFormLayout,
    QLineEdit, QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox,
    QProgressBar, QMessageBox, QMenuBar, QStatusBar, QSplitter,
    QTabWidget, QTextBrowser, QFrame
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, QSize
from PyQt6.QtGui import QAction, QIcon, QFont, QPalette

from session_state import SessionState
from llm_config_dialog import LLMConfigDialog  
from document_generator import DocumentGenerator

logger = logging.getLogger(__name__)


class BudgetJustificationWizard(QWizard):
    """Main wizard for the budget justification workflow."""
    
    def __init__(self, session: SessionState, parent=None):
        super().__init__(parent)
        self.session = session
        self.document_generator = DocumentGenerator()
        
        self.setWindowTitle("Budget Justification Automation Tool")
        self.setMinimumSize(1000, 700)
        self.setWizardStyle(QWizard.WizardStyle.ModernStyle)
        
        # Add wizard pages
        self.template_page = TemplateSelectionPage(self.session)
        self.budget_page = BudgetSelectionPage(self.session)
        self.mapping_page = FieldMappingPage(self.session)
        self.preview_page = PreviewPage(self.session, self.document_generator)
        
        self.addPage(self.template_page)
        self.addPage(self.budget_page)
        self.addPage(self.mapping_page)
        self.addPage(self.preview_page)
        
        # Connect signals
        self.mapping_page.llm_configured.connect(self.on_llm_configured)
    
    def on_llm_configured(self, enabled: bool):
        """Handle LLM configuration changes."""
        try:
            if enabled:
                self.session.enhanced_auto_match_fields()
                self.mapping_page.update_content()
        except Exception as e:
            logger.error(f"Error in LLM configuration: {e}")


class TemplateSelectionPage(QWizardPage):
    """Page for selecting template file."""
    
    def __init__(self, session: SessionState):
        super().__init__()
        self.session = session
        self.setup_ui()
    
    def setup_ui(self):
        self.setTitle("Select Template")
        self.setSubTitle("Choose your budget justification template file")
        
        layout = QVBoxLayout()
        
        # Template selection
        select_group = QGroupBox("Template File")
        select_layout = QVBoxLayout()
        
        self.file_label = QLabel("No template selected")
        self.file_label.setStyleSheet("padding: 10px; border: 1px solid gray; background: #f0f0f0;")
        
        self.select_button = QPushButton("Select Template File...")
        self.select_button.clicked.connect(self.select_template)
        
        select_layout.addWidget(self.file_label)
        select_layout.addWidget(self.select_button)
        select_group.setLayout(select_layout)
        
        # Template preview
        preview_group = QGroupBox("Template Preview")
        preview_layout = QVBoxLayout()
        
        self.preview_text = QTextBrowser()
        self.preview_text.setMaximumHeight(300)
        
        preview_layout.addWidget(self.preview_text)
        preview_group.setLayout(preview_layout)
        
        layout.addWidget(select_group)
        layout.addWidget(preview_group)
        layout.addStretch()
        
        self.setLayout(layout)
    
    def select_template(self):
        """Select template file."""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select Template",
                "",
                "Templates (*.docx *.md *.txt *.pdf);;Word Documents (*.docx);;Markdown (*.md);;Text Files (*.txt);;PDF Files (*.pdf)"
            )
            
            if file_path:
                path = Path(file_path)
                if self.session.load_template(path):
                    self.file_label.setText(f"Selected: {path.name}")
                    self.file_label.setStyleSheet("padding: 10px; border: 1px solid green; background: #e8f5e8;")
                    
                    # Show preview
                    preview_content = f"Template: {path.name}\n"
                    preview_content += f"Type: {self.session.template.source_type.upper()}\n"
                    preview_content += f"Placeholders: {len(self.session.template.placeholders)}\n\n"
                    
                    if len(self.session.template.content) > 1000:
                        preview_content += self.session.template.content[:1000] + "..."
                    else:
                        preview_content += self.session.template.content
                    
                    self.preview_text.setPlainText(preview_content)
                    
                    self.completeChanged.emit()
                else:
                    QMessageBox.warning(self, "Error", "Failed to load template file.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to select template: {str(e)}")
    
    def isComplete(self):
        return self.session.template is not None


class BudgetSelectionPage(QWizardPage):
    """Page for selecting budget file."""
    
    def __init__(self, session: SessionState):
        super().__init__()
        self.session = session
        self.setup_ui()
    
    def setup_ui(self):
        self.setTitle("Select Budget")
        self.setSubTitle("Choose your budget spreadsheet file")
        
        layout = QVBoxLayout()
        
        # Budget selection
        select_group = QGroupBox("Budget File")
        select_layout = QVBoxLayout()
        
        self.file_label = QLabel("No budget selected")
        self.file_label.setStyleSheet("padding: 10px; border: 1px solid gray; background: #f0f0f0;")
        
        self.select_button = QPushButton("Select Budget File...")
        self.select_button.clicked.connect(self.select_budget)
        
        select_layout.addWidget(self.file_label)
        select_layout.addWidget(self.select_button)
        select_group.setLayout(select_layout)
        
        # Budget info
        info_group = QGroupBox("Budget Information")
        self.info_layout = QFormLayout()
        
        self.sheets_label = QLabel("0")
        self.cells_label = QLabel("0")
        self.years_label = QLabel("None")
        
        self.info_layout.addRow("Sheets:", self.sheets_label)
        self.info_layout.addRow("Budget Cells:", self.cells_label)
        self.info_layout.addRow("Years Found:", self.years_label)
        
        info_group.setLayout(self.info_layout)
        
        layout.addWidget(select_group)
        layout.addWidget(info_group)
        layout.addStretch()
        
        self.setLayout(layout)
    
    def select_budget(self):
        """Select budget file."""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select Budget",
                "",
                "Spreadsheets (*.xlsx *.xls *.csv);;Excel Files (*.xlsx *.xls);;CSV Files (*.csv)"
            )
            
            if file_path:
                path = Path(file_path)
                if self.session.load_budget(path):
                    self.file_label.setText(f"Selected: {path.name}")
                    self.file_label.setStyleSheet("padding: 10px; border: 1px solid green; background: #e8f5e8;")
                    
                    # Update info
                    self.sheets_label.setText(str(len(self.session.budget.sheets)))
                    self.cells_label.setText(str(len(self.session.budget.cells)))
                    years = self.session.budget.get_years()
                    self.years_label.setText(", ".join(map(str, years)) if years else "None")
                    
                    self.completeChanged.emit()
                else:
                    QMessageBox.warning(self, "Error", "Failed to load budget file.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to select budget: {str(e)}")
    
    def isComplete(self):
        return self.session.budget is not None


class FieldMappingPage(QWizardPage):
    """Enhanced page for mapping template fields to budget values with LLM integration."""
    
    llm_configured = pyqtSignal(bool)
    
    def __init__(self, session: SessionState):
        super().__init__()
        self.session = session
        self._updating_content = False
        self.setup_ui()
        self.setup_timers()
    
    def setup_ui(self):
        self.setTitle("Field Mapping")
        self.setSubTitle("Map template fields to budget values")
        
        layout = QVBoxLayout()
        
        # LLM Control Panel
        self.llm_group = QGroupBox("🤖 LLM Analysis")
        llm_layout = QHBoxLayout()
        
        self.llm_status_label = QLabel("LLM: Not configured")
        self.llm_status_label.setStyleSheet("color: gray;")
        
        self.configure_llm_button = QPushButton("Configure LLM")
        self.configure_llm_button.clicked.connect(self.configure_llm)
        
        self.analyze_button = QPushButton("🧠 Analyze with LLM")
        self.analyze_button.clicked.connect(self.run_llm_analysis)
        self.analyze_button.setEnabled(False)
        
        self.improve_button = QPushButton("⚡ Improve Low Confidence")
        self.improve_button.clicked.connect(self.improve_mappings)
        self.improve_button.setEnabled(False)
        
        self.cost_label = QLabel("Cost: $0.00")
        self.cost_label.setStyleSheet("font-family: monospace;")
        
        self.budget_progress = QProgressBar()
        self.budget_progress.setRange(0, 100)
        self.budget_progress.setValue(0)
        self.budget_progress.setMaximumWidth(100)
        self.budget_progress.setVisible(False)
        
        llm_layout.addWidget(self.llm_status_label)
        llm_layout.addWidget(self.configure_llm_button)
        llm_layout.addWidget(self.analyze_button)
        llm_layout.addWidget(self.improve_button)
        llm_layout.addStretch()
        llm_layout.addWidget(self.cost_label)
        llm_layout.addWidget(self.budget_progress)
        
        self.llm_group.setLayout(llm_layout)
        
        # Controls
        controls_group = QGroupBox("Controls")
        controls_layout = QHBoxLayout()
        
        self.auto_match_button = QPushButton("Auto Match")
        self.auto_match_button.clicked.connect(self.auto_match)
        
        self.clear_button = QPushButton("Clear All")
        self.clear_button.clicked.connect(self.clear_all)
        
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.update_content)
        
        controls_layout.addWidget(self.auto_match_button)
        controls_layout.addWidget(self.clear_button)
        controls_layout.addWidget(self.refresh_button)
        controls_layout.addStretch()
        
        controls_group.setLayout(controls_layout)
        
        # Mapping table
        self.mapping_table = QTableWidget()
        self.setup_table()
        
        # Status
        self.status_label = QLabel("Ready")
        
        layout.addWidget(self.llm_group)
        layout.addWidget(controls_group)
        layout.addWidget(self.mapping_table)
        layout.addWidget(self.status_label)
        
        self.setLayout(layout)
    
    def setup_table(self):
        """Set up the mapping table with LLM columns."""
        self.mapping_table.setColumnCount(7)
        self.mapping_table.setHorizontalHeaderLabels([
            "Field", "Current Value", "Confidence", "Source", "LLM Analysis", "Manual Value", "Notes"
        ])
        
        header = self.mapping_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch)
        
        self.mapping_table.cellChanged.connect(self.on_cell_changed)
    
    def setup_timers(self):
        """Set up timers for LLM status updates."""
        try:
            self.llm_timer = QTimer()
            self.llm_timer.timeout.connect(self.update_llm_status)
            self.llm_timer.start(2000)  # Update every 2 seconds
        except Exception as e:
            logger.error(f"Failed to setup timers: {e}")
    
    def configure_llm(self):
        """Open LLM configuration dialog - FIXED VERSION."""
        try:
            dialog = LLMConfigDialog(
                session_state=self.session,  # FIXED: Pass session_state instead of llm_manager
                parent=self
            )
            dialog.llm_configured.connect(self.on_llm_configured)
            dialog.exec()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open LLM configuration: {str(e)}")
    
    def on_llm_configured(self, enabled: bool):
        """Handle LLM configuration changes."""
        try:
            if enabled:
                self.llm_configured.emit(True)
                self.update_llm_status()
        except Exception as e:
            logger.error(f"Error in on_llm_configured: {e}")
    
    def update_llm_status(self):
        """Update LLM status display."""
        try:
            # Check if session has LLM integration
            if not hasattr(self.session, 'llm_enabled') or not self.session.llm_enabled:
                self.llm_status_label.setText("LLM: Not configured")
                self.llm_status_label.setStyleSheet("color: gray;")
                self.cost_label.setText("Cost: $0.00")
                self.budget_progress.setValue(0)
                self.budget_progress.setVisible(False)
                self.analyze_button.setEnabled(False)
                self.improve_button.setEnabled(False)
                return

            usage = self.session.get_llm_usage_summary()
            
            self.llm_status_label.setText("LLM: ✅ Enabled")
            self.llm_status_label.setStyleSheet("color: green;")
            
            total_cost = usage.get('total_cost', 0.0)
            self.cost_label.setText(f"Cost: ${total_cost:.4f}")
            
            utilization = usage.get('budget_utilization', 0.0)
            self.budget_progress.setValue(int(utilization * 100))
            self.budget_progress.setVisible(True)
            
            self.analyze_button.setEnabled(True)
            self.improve_button.setEnabled(True)
            
            # Color-code based on usage
            if utilization > 0.8:
                self.cost_label.setStyleSheet("color: red; font-family: monospace;")
            elif utilization > 0.6:
                self.cost_label.setStyleSheet("color: orange; font-family: monospace;")
            else:
                self.cost_label.setStyleSheet("color: green; font-family: monospace;")
        
        except Exception as e:
            self.llm_status_label.setText("LLM: ⚠️ Error")
            self.llm_status_label.setStyleSheet("color: red;")
            logger.error(f"Error updating LLM status: {e}")
    
    def run_llm_analysis(self):
        """Run LLM analysis on all fields."""
        try:
            # Check if session has LLM capabilities
            if not hasattr(self.session, 'llm_enabled') or not self.session.llm_enabled:
                QMessageBox.warning(self, "LLM Not Available", "LLM analysis is not available.")
                return
            
            if not hasattr(self.session, 'enhanced_auto_match_fields'):
                QMessageBox.warning(self, "LLM Not Available", "Enhanced field matching is not available.")
                return
            
            self.analyze_button.setEnabled(False)
            self.analyze_button.setText("🧠 Analyzing...")
            
            self.session.enhanced_auto_match_fields()
            
            # Only update if not already updating
            if not self._updating_content:
                self.update_content()
            
            usage = self.session.get_llm_usage_summary()
            enhanced_count = usage.get('enhanced_mappings', 0)
            cost = usage.get('total_cost', 0.0)
            
            QMessageBox.information(
                self, 
                "Analysis Complete", 
                f"LLM analysis completed!\n\n"
                f"Enhanced mappings: {enhanced_count}\n"
                f"Total cost: ${cost:.4f}"
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Analysis Error", f"LLM analysis failed:\n{str(e)}")
        
        finally:
            self.analyze_button.setEnabled(True)
            self.analyze_button.setText("🧠 Analyze with LLM")
    
    def improve_mappings(self):
        """Improve low-confidence mappings with LLM."""
        try:
            # Check if session has LLM capabilities
            if not hasattr(self.session, 'llm_enabled') or not self.session.llm_enabled:
                QMessageBox.warning(self, "LLM Not Available", "LLM improvement is not available.")
                return
            
            if not hasattr(self.session, 'improve_low_confidence_mappings'):
                QMessageBox.warning(self, "LLM Not Available", "LLM improvement is not available.")
                return
            
            self.improve_button.setEnabled(False)
            self.improve_button.setText("⚡ Improving...")
            
            improved_count = self.session.improve_low_confidence_mappings()
            
            if improved_count > 0:
                # Only update if not already updating
                if not self._updating_content:
                    self.update_content()
                
                QMessageBox.information(
                    self, 
                    "Improvement Complete", 
                    f"Improved {improved_count} mappings with LLM analysis!"
                )
            else:
                QMessageBox.information(
                    self, 
                    "No Improvements", 
                    "No mappings needed improvement or could be improved."
                )
        
        except Exception as e:
            QMessageBox.critical(self, "Improvement Error", f"LLM improvement failed:\n{str(e)}")
        
        finally:
            self.improve_button.setEnabled(True)
            self.improve_button.setText("⚡ Improve Low Confidence")
    
    def auto_match(self):
        """Run auto-matching (enhanced if LLM enabled)."""
        try:
            if hasattr(self.session, 'llm_enabled') and self.session.llm_enabled and hasattr(self.session, 'enhanced_auto_match_fields'):
                self.session.enhanced_auto_match_fields()
            elif hasattr(self.session, '_auto_match_fields'):
                self.session._auto_match_fields()
            else:
                QMessageBox.warning(self, "Feature Not Available", "Auto-matching is not available.")
                return
            
            # Only update if not already updating
            if not self._updating_content:
                self.update_content()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Auto-matching failed:\n{str(e)}")
    
    def clear_all(self):
        """Clear all mappings."""
        try:
            for mapping in self.session.mappings.values():
                mapping.budget_cell = None
                mapping.manual_value = None
                mapping.is_manually_set = False
                mapping.confidence = 0.0
                mapping.notes = ""
            
            # Only update if not already updating
            if not self._updating_content:
                self.update_content()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to clear mappings:\n{str(e)}")
    
    def update_content(self):
        """Update table content."""
        # Prevent recursion
        if self._updating_content:
            return
        
        try:
            if not self.session.mappings:
                self.mapping_table.setRowCount(0)
                return
            
            # Set flag to prevent recursive updates
            self._updating_content = True
            
            # Temporarily disconnect the signal to prevent recursion
            self.mapping_table.cellChanged.disconnect()
            
            self.mapping_table.setRowCount(len(self.session.mappings))
            
            for row, (field_name, mapping) in enumerate(self.session.mappings.items()):
                # Field name
                field_item = QTableWidgetItem(field_name)
                field_item.setFlags(field_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.mapping_table.setItem(row, 0, field_item)
                
                # Current value
                value_item = QTableWidgetItem(mapping.display_value)
                value_item.setFlags(value_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.mapping_table.setItem(row, 1, value_item)
                
                # Confidence
                confidence_item = QTableWidgetItem(f"{mapping.confidence:.2f}")
                confidence_item.setFlags(confidence_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.mapping_table.setItem(row, 2, confidence_item)
                
                # Source
                source = "Manual" if mapping.is_manually_set else "Auto"
                if hasattr(self.session, 'enhanced_mappings') and field_name in self.session.enhanced_mappings:
                    enhanced = self.session.enhanced_mappings[field_name]
                    source = enhanced.source.title()
                    if enhanced.llm_cost > 0:
                        source += f" (${enhanced.llm_cost:.3f})"
                
                source_item = QTableWidgetItem(source)
                source_item.setFlags(source_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.mapping_table.setItem(row, 3, source_item)
                
                # LLM Analysis
                llm_analysis = "N/A"
                if hasattr(self.session, 'enhanced_mappings') and field_name in self.session.enhanced_mappings:
                    enhanced = self.session.enhanced_mappings[field_name]
                    llm_analysis = enhanced.reasoning[:80] + "..." if len(enhanced.reasoning) > 80 else enhanced.reasoning
                
                llm_item = QTableWidgetItem(llm_analysis)
                llm_item.setFlags(llm_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                llm_item.setToolTip(llm_analysis)
                self.mapping_table.setItem(row, 4, llm_item)
                
                # Manual value (editable)
                manual_value = mapping.manual_value if mapping.manual_value is not None else ""
                manual_item = QTableWidgetItem(str(manual_value))
                self.mapping_table.setItem(row, 5, manual_item)
                
                # Notes (editable)
                notes_item = QTableWidgetItem(mapping.notes)
                self.mapping_table.setItem(row, 6, notes_item)
            
            # Reconnect the signal
            self.mapping_table.cellChanged.connect(self.on_cell_changed)
            
            self.update_status()
            
        except Exception as e:
            logger.error(f"Error updating content: {e}")
        finally:
            # Always clear the flag
            self._updating_content = False
    
    def update_status(self):
        """Update status display."""
        try:
            summary = self.session.get_mapping_summary()
            
            status_parts = [
                f"Total: {summary['total_fields']}",
                f"Mapped: {summary['mapped_fields']}",
                f"Manual: {summary['manual_overrides']}",
                f"High Conf: {summary['high_confidence']}",
                f"Low Conf: {summary['low_confidence']}"
            ]
            
            if summary.get('llm_enabled', False):
                status_parts.append(f"LLM Enhanced: {summary.get('llm_enhanced_mappings', 0)}")
            
            self.status_label.setText(" | ".join(status_parts))
        except Exception as e:
            logger.error(f"Error updating status: {e}")
    
    def on_cell_changed(self, row: int, col: int):
        """Handle cell changes in the table."""
        # Prevent recursion
        if self._updating_content:
            return
        
        try:
            if col == 5:  # Manual value column
                item = self.mapping_table.item(row, 5)
                field_item = self.mapping_table.item(row, 0)
                
                if item and field_item:
                    field_name = field_item.text()
                    value_text = item.text().strip()
                    
                    if value_text:
                        try:
                            # Parse value (remove $ and commas)
                            clean_value = value_text.replace('$', '').replace(',', '')
                            value = float(clean_value)
                            self.session.set_manual_value(field_name, value, "Manual entry")
                            
                            # Update just this row's display instead of full refresh
                            self.update_single_row(row, field_name)
                            
                        except ValueError:
                            QMessageBox.warning(self, "Invalid Value", "Please enter a valid numeric value.")
                            # Reset to previous value
                            mapping = self.session.mappings.get(field_name)
                            if mapping and mapping.manual_value is not None:
                                item.setText(str(mapping.manual_value))
                            else:
                                item.setText("")
                    else:
                        self.session.clear_manual_value(field_name)
                        # Update just this row's display instead of full refresh
                        self.update_single_row(row, field_name)
            
            elif col == 6:  # Notes column
                item = self.mapping_table.item(row, 6)
                field_item = self.mapping_table.item(row, 0)
                
                if item and field_item:
                    field_name = field_item.text()
                    notes = item.text()
                    
                    if field_name in self.session.mappings:
                        self.session.mappings[field_name].notes = notes
        except Exception as e:
            logger.error(f"Error in cell changed: {e}")
    
    def update_single_row(self, row: int, field_name: str):
        """Update a single row in the table without full refresh."""
        try:
            if field_name not in self.session.mappings:
                return
            
            mapping = self.session.mappings[field_name]
            
            # Temporarily disconnect to avoid recursion
            self.mapping_table.cellChanged.disconnect()
            
            # Update current value
            value_item = self.mapping_table.item(row, 1)
            if value_item:
                value_item.setText(mapping.display_value)
            
            # Update confidence
            confidence_item = self.mapping_table.item(row, 2)
            if confidence_item:
                confidence_item.setText(f"{mapping.confidence:.2f}")
            
            # Update source
            source = "Manual" if mapping.is_manually_set else "Auto"
            source_item = self.mapping_table.item(row, 3)
            if source_item:
                source_item.setText(source)
            
            # Reconnect signal
            self.mapping_table.cellChanged.connect(self.on_cell_changed)
            
            # Update status bar
            self.update_status()
        except Exception as e:
            logger.error(f"Error updating single row: {e}")
            # Make sure to reconnect signal even if there's an error
            try:
                self.mapping_table.cellChanged.connect(self.on_cell_changed)
            except:
                pass
    
    def initializePage(self):
        """Initialize page when shown."""
        # Only update if not already updating
        if not self._updating_content:
            self.update_content()
        self.update_llm_status()
    
    def isComplete(self):
        """Check if page is complete."""
        return self.session.is_ready_for_generation()


class PreviewPage(QWizardPage):
    """Page for previewing and generating the final document."""
    
    def __init__(self, session: SessionState, document_generator: DocumentGenerator):
        super().__init__()
        self.session = session
        self.document_generator = document_generator
        self.setup_ui()
    
    def setup_ui(self):
        self.setTitle("Preview & Generate")
        self.setSubTitle("Preview your budget justification and generate the final document")
        
        layout = QVBoxLayout()
        
        # Controls
        controls_group = QGroupBox("Generation Options")
        controls_layout = QFormLayout()
        
        self.format_combo = QComboBox()
        self.format_combo.addItems(["DOCX (Word)", "Markdown"])
        self.format_combo.setCurrentText("DOCX (Word)")
        
        self.output_path_edit = QLineEdit()
        self.output_path_edit.setText(str(Path.home() / "Documents"))
        
        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self.browse_output_path)
        
        path_layout = QHBoxLayout()
        path_layout.addWidget(self.output_path_edit)
        path_layout.addWidget(self.browse_button)
        
        controls_layout.addRow("Format:", self.format_combo)
        controls_layout.addRow("Output Directory:", path_layout)
        
        controls_group.setLayout(controls_layout)
        
        # Preview
        preview_group = QGroupBox("Document Preview")
        preview_layout = QVBoxLayout()
        
        self.preview_text = QTextBrowser()
        
        preview_layout.addWidget(self.preview_text)
        preview_group.setLayout(preview_layout)
        
        # Generate button
        self.generate_button = QPushButton("Generate Document")
        self.generate_button.clicked.connect(self.generate_document)
        self.generate_button.setStyleSheet("QPushButton { font-size: 14px; font-weight: bold; padding: 10px; }")
        
        layout.addWidget(controls_group)
        layout.addWidget(preview_group)
        layout.addWidget(self.generate_button)
        
        self.setLayout(layout)
    
    def browse_output_path(self):
        """Browse for output directory."""
        try:
            directory = QFileDialog.getExistingDirectory(
                self, "Select Output Directory", self.output_path_edit.text()
            )
            
            if directory:
                self.output_path_edit.setText(directory)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to browse directory: {str(e)}")
    
    def generate_document(self):
        """Generate the final document."""
        try:
            # Update session config
            self.session.config.output_directory = Path(self.output_path_edit.text())
            format_text = self.format_combo.currentText()
            self.session.config.output_format = "docx" if "DOCX" in format_text else "md"
            
            # Generate document
            output_path = self.document_generator.generate_document(self.session)
            
            QMessageBox.information(
                self,
                "Success",
                f"Document generated successfully!\n\nSaved to: {output_path}"
            )
            
            # Offer to open the file
            reply = QMessageBox.question(
                self,
                "Open Document",
                "Would you like to open the generated document?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                try:
                    if platform.system() == "Windows":
                        subprocess.run(["start", str(output_path)], shell=True)
                    elif platform.system() == "Darwin":  # macOS
                        subprocess.run(["open", str(output_path)])
                    else:  # Linux
                        subprocess.run(["xdg-open", str(output_path)])
                except Exception as e:
                    logger.error(f"Failed to open document: {e}")
    
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate document:\n{str(e)}")
    
    def initializePage(self):
        """Initialize page when shown."""
        try:
            # Generate preview
            preview_content = self.document_generator.preview_content(self.session)
            self.preview_text.setPlainText(preview_content)
        except Exception as e:
            self.preview_text.setPlainText(f"Error generating preview: {str(e)}")
    
    def isComplete(self):
        """Always complete once we reach this page."""
        return True


class MainWindow(QMainWindow):
    """Main application window with integrated LLM functionality."""
    
    def __init__(self):
        super().__init__()
        self.session = SessionState()
        self.setup_ui()
        self.setup_menu_bar()
        self.setup_status_bar()
    
    def setup_ui(self):
        """Set up the main UI."""
        self.setWindowTitle("Budget Justification Automation Tool")
        self.setMinimumSize(1200, 800)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create wizard
        self.wizard = BudgetJustificationWizard(self.session, self)
        
        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.wizard)
        central_widget.setLayout(layout)
    
    def setup_menu_bar(self):
        """Set up the menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        new_action = QAction("New Session", self)
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(self.new_session)
        
        save_action = QAction("Save Session", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_session)
        
        load_action = QAction("Load Session", self)
        load_action.setShortcut("Ctrl+O")
        load_action.triggered.connect(self.load_session)
        
        file_menu.addAction(new_action)
        file_menu.addSeparator()
        file_menu.addAction(save_action)
        file_menu.addAction(load_action)
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # LLM menu
        llm_menu = menubar.addMenu("🤖 LLM")
        
        configure_action = QAction("Configure OpenAI API...", self)
        configure_action.setShortcut("Ctrl+L")
        configure_action.triggered.connect(self.configure_llm)
        
        analyze_action = QAction("Analyze Template", self)
        analyze_action.setShortcut("Ctrl+Shift+A")
        analyze_action.triggered.connect(self.quick_analyze)
        
        usage_action = QAction("View Usage & Costs", self)
        usage_action.triggered.connect(self.show_llm_usage)
        
        export_action = QAction("Export LLM Analysis...", self)
        export_action.triggered.connect(self.export_llm_analysis)
        
        llm_menu.addAction(configure_action)
        llm_menu.addSeparator()
        llm_menu.addAction(analyze_action)
        llm_menu.addAction(usage_action)
        llm_menu.addSeparator()
        llm_menu.addAction(export_action)
        
        # Store references for enabling/disabling
        self.llm_actions = {
            'analyze': analyze_action,
            'usage': usage_action,
            'export': export_action
        }
        
        # Initially disable LLM-dependent actions
        self.update_llm_menu_state(False)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def setup_status_bar(self):
        """Set up the status bar."""
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready")
    
    def new_session(self):
        """Start a new session."""
        try:
            if self.session.is_dirty:
                reply = QMessageBox.question(
                    self,
                    "Unsaved Changes",
                    "You have unsaved changes. Continue with new session?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                
                if reply == QMessageBox.StandardButton.No:
                    return
            
            self.session.reset()
            self.wizard.restart()
            self.status_bar.showMessage("New session started")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create new session: {str(e)}")
    
    def save_session(self):
        """Save the current session."""
        try:
            session_file = self.session.save_session()
            self.status_bar.showMessage(f"Session saved to {session_file.name}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save session:\n{str(e)}")
    
    def load_session(self):
        """Load a session from file."""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Load Session",
                str(self.session._session_dir),
                "Session Files (*.json)"
            )
            
            if file_path:
                if self.session.load_session(Path(file_path)):
                    self.wizard.restart()
                    self.status_bar.showMessage(f"Session loaded from {Path(file_path).name}")
                else:
                    QMessageBox.warning(self, "Load Error", "Failed to load session file.")
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load session:\n{str(e)}")
    
    def configure_llm(self):
        """Open LLM configuration dialog - FIXED VERSION."""
        try:
            dialog = LLMConfigDialog(
                session_state=self.session,  # FIXED: Pass session_state instead of llm_manager
                parent=self
            )
            dialog.llm_configured.connect(self.on_llm_configured)
            dialog.exec()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open LLM configuration: {str(e)}")
    
    def on_llm_configured(self, enabled: bool):
        """Handle LLM configuration changes."""
        try:
            self.update_llm_menu_state(enabled)
            if enabled:
                self.status_bar.showMessage("LLM integration enabled", 3000)
            else:
                self.status_bar.showMessage("LLM integration disabled", 3000)
        except Exception as e:
            logger.error(f"Error in LLM configuration handler: {e}")
    
    def update_llm_menu_state(self, enabled: bool):
        """Enable/disable LLM menu items."""
        try:
            for action in self.llm_actions.values():
                action.setEnabled(enabled)
        except Exception as e:
            logger.error(f"Error updating LLM menu state: {e}")
    
    def quick_analyze(self):
        """Quick LLM analysis of current template."""
        try:
            current_page = self.wizard.currentPage()
            
            if hasattr(current_page, 'run_llm_analysis'):
                current_page.run_llm_analysis()
            else:
                QMessageBox.information(
                    self,
                    "Analysis Not Available",
                    "LLM analysis is only available on the Field Mapping page."
                )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to run analysis: {str(e)}")
    
    def show_llm_usage(self):
        """Show LLM usage statistics."""
        try:
            if not hasattr(self.session, 'llm_enabled') or not self.session.llm_enabled:
                QMessageBox.warning(self, "No Data", "LLM is not enabled.")
                return
            
            if not hasattr(self.session, 'get_llm_usage_summary'):
                QMessageBox.warning(self, "Feature Not Available", "LLM usage tracking is not available.")
                return
            
            usage = self.session.get_llm_usage_summary()
            
            usage_text = f"""LLM Usage Summary

Total Cost: ${usage.get('total_cost', 0.0):.4f}
Remaining Budget: ${usage.get('budget_remaining', 0.0):.4f}
Budget Utilization: {usage.get('budget_utilization', 0.0):.1%}

Operations: {usage.get('total_operations', 0)}
Tokens Used: {usage.get('llm_tokens', 0):,}

Field Suggestions: {usage.get('field_suggestions', 0)}
Enhanced Mappings: {usage.get('enhanced_mappings', 0)}"""
            
            QMessageBox.information(self, "LLM Usage", usage_text)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to get usage data:\n{str(e)}")
    
    def export_llm_analysis(self):
        """Export LLM analysis to JSON file."""
        try:
            if not hasattr(self.session, 'llm_enabled') or not self.session.llm_enabled:
                QMessageBox.warning(self, "No Data", "LLM is not enabled.")
                return
            
            if not hasattr(self.session, 'export_llm_analysis'):
                QMessageBox.warning(self, "Feature Not Available", "LLM analysis export is not available.")
                return
            
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Export LLM Analysis",
                "llm_analysis.json",
                "JSON Files (*.json)"
            )
            
            if file_path:
                if self.session.export_llm_analysis(Path(file_path)):
                    QMessageBox.information(
                        self,
                        "Export Success",
                        f"LLM analysis exported to:\n{file_path}"
                    )
                else:
                    QMessageBox.warning(self, "Export Failed", "Failed to export LLM analysis.")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Export failed:\n{str(e)}")
    
    def show_about(self):
        """Show about dialog."""
        try:
            about_text = """
Budget Justification Automation Tool v1.0

Automates the creation of budget justifications by combining:
• Template analysis with placeholder detection
• Budget spreadsheet parsing and value extraction  
• AI-powered field mapping using OpenAI GPT models
• Document generation in Word and Markdown formats

Features:
• Multi-format template support (DOCX, MD, TXT, PDF)
• Excel/CSV budget file processing
• LLM-enhanced field detection and resolution
• Manual override capabilities
• Cost tracking and budget controls
• Comprehensive logging and error handling

Created with PyQt6, OpenAI API, and Python.
            """
            
            QMessageBox.about(self, "About", about_text.strip())
        except Exception as e:
            logger.error(f"Error showing about dialog: {e}")
    
    def closeEvent(self, event):
        """Handle application close."""
        try:
            if self.session.is_dirty:
                reply = QMessageBox.question(
                    self,
                    "Unsaved Changes",
                    "You have unsaved changes. Save before closing?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel
                )
                
                if reply == QMessageBox.StandardButton.Yes:
                    self.save_session()
                elif reply == QMessageBox.StandardButton.Cancel:
                    event.ignore()
                    return
            
            # Shutdown LLM
            if hasattr(self.session, 'shutdown_llm'):
                self.session.shutdown_llm()
            
            event.accept()
        except Exception as e:
            logger.error(f"Error during close: {e}")
            event.accept()  # Close anyway


def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    app.setApplicationName("Budget Justification Tool")
    app.setApplicationVersion("1.0")
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
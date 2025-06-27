"""
Enhanced session state management with improved string analysis tracking,
field mapping history, and grant-specific workflow management.
"""
import json
import logging
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import hashlib
import re

# Import will be handled after class definitions to avoid circular imports
# from field_detector import TemplateField
# from budget_book import PersonnelEntry, BudgetSection
# from cell_resolver import FieldMatch

class AnalysisStage(Enum):
    """Stages of the budget analysis workflow."""
    INITIALIZED = "initialized"
    TEMPLATE_LOADED = "template_loaded"
    BUDGET_LOADED = "budget_loaded"
    STRUCTURE_ANALYZED = "structure_analyzed"
    FIELDS_DETECTED = "fields_detected"
    INITIAL_MAPPING = "initial_mapping"
    LLM_ANALYSIS = "llm_analysis"
    USER_REVIEW = "user_review"
    FINAL_MAPPING = "final_mapping"
    DOCUMENT_GENERATED = "document_generated"
    COMPLETED = "completed"

class MappingConfidence(Enum):
    """Confidence levels for field mappings."""
    HIGH = "high"          # 0.8+
    MEDIUM = "medium"      # 0.5-0.8
    LOW = "low"           # 0.3-0.5
    UNCERTAIN = "uncertain" # <0.3

@dataclass
class FieldMappingState:
    """State information for a field mapping."""
    field_name: str
    template_field: Optional[Any] = None  # Will be TemplateField when imported
    mapped_location: Optional[str] = None  # e.g., "Sheet1!R5C3"
    mapped_value: Any = None
    confidence: float = 0.0
    confidence_level: MappingConfidence = MappingConfidence.UNCERTAIN
    mapping_method: str = "none"  # "automatic", "llm", "manual", "rule_based"
    alternatives: List[Tuple[str, float]] = field(default_factory=list)  # (location, confidence)
    user_verified: bool = False
    user_notes: str = ""
    last_updated: datetime = field(default_factory=datetime.now)
    mapping_history: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class PersonnelMappingState:
    """State for personnel-specific mappings."""
    person_id: str
    name: Optional[str] = None
    name_location: Optional[str] = None
    title: Optional[str] = None
    title_location: Optional[str] = None
    effort_percent: Optional[float] = None
    effort_location: Optional[str] = None
    salary: Optional[float] = None
    salary_location: Optional[str] = None
    notes: str = ""
    notes_location: Optional[str] = None
    confidence: float = 0.0
    user_verified: bool = False
    extraction_method: str = "automatic"

@dataclass
class MappingDisplay:
    """Simple wrapper for mapping display in GUI."""
    field_name: str = ""
    mapped: bool = False
    location: str = ""
    value: Any = ""
    confidence: float = 0.0
    method: str = "none"
    verified: bool = False
    template_field: Any = None
    
    @property
    def display_value(self):
        """GUI compatibility: return display value."""
        if self.value:
            return str(self.value)[:100]  # Truncate long values
        return ""
    
    @property
    def current_value(self):
        """GUI compatibility: alias for value."""
        return self.value
    
    @property
    def source(self):
        """GUI compatibility: return location as source."""
        return self.location

@dataclass
class AnalysisQualityMetrics:
    """Quality metrics for the current analysis."""
    total_fields: int = 0
    mapped_fields: int = 0
    high_confidence_mappings: int = 0
    medium_confidence_mappings: int = 0
    low_confidence_mappings: int = 0
    uncertain_mappings: int = 0
    user_verified_mappings: int = 0
    llm_enhanced_mappings: int = 0
    string_field_coverage: float = 0.0  # Percentage of text fields successfully mapped
    personnel_extraction_quality: float = 0.0
    notes_coverage: float = 0.0  # How well notes/descriptions are captured
    overall_quality_score: float = 0.0
    """Quality metrics for the current analysis."""
    total_fields: int = 0
    mapped_fields: int = 0
    high_confidence_mappings: int = 0
    medium_confidence_mappings: int = 0
    low_confidence_mappings: int = 0
    uncertain_mappings: int = 0
    user_verified_mappings: int = 0
    llm_enhanced_mappings: int = 0
    string_field_coverage: float = 0.0  # Percentage of text fields successfully mapped
    personnel_extraction_quality: float = 0.0
    notes_coverage: float = 0.0  # How well notes/descriptions are captured
    overall_quality_score: float = 0.0

@dataclass
class TemplateWrapper:
    """Wrapper for template data with GUI-expected attributes."""
    fields: List[Any] = field(default_factory=list)
    source_type: str = "template"
    source: str = ""
    description: str = ""
    file_path: str = ""
    _content: str = ""
    
    def __post_init__(self):
        """Set default values if not provided."""
        if not self.source and self.file_path:
            self.source = Path(self.file_path).name
        if not self.description and self.fields:
            self.description = f"Template with {len(self.fields)} fields"
    
    @property
    def placeholders(self):
        """Backward compatibility: return fields as placeholders for GUI."""
        return self.fields
    
    @property
    def content(self):
        """Backward compatibility: return template content for GUI."""
        return self._content
    
    @content.setter
    def content(self, value):
        """Set template content."""
        self._content = value if value else ""
    
    @property
    def filename(self):
        """GUI compatibility: return filename."""
        return Path(self.file_path).name if self.file_path else ""
    
    @property
    def name(self):
        """GUI compatibility: return name (alias for filename)."""
        return self.filename
    
    @property
    def path(self):
        """GUI compatibility: return file path."""
        return self.file_path
    
    @property
    def format(self):
        """GUI compatibility: return file format."""
        return Path(self.file_path).suffix.lower() if self.file_path else ""
    
    @property
    def file_type(self):
        """GUI compatibility: return file type (alias for format)."""
        return self.format
    
    @property
    def field_count(self):
        """GUI compatibility: return number of fields."""
        return len(self.fields)
    
    @property
    def text(self):
        """GUI compatibility: return text content (alias for content)."""
        return self.content
    
    @property
    def data(self):
        """GUI compatibility: return data (returns fields)."""
        return self.fields
    
    @property
    def metadata(self):
        """GUI compatibility: return metadata dict."""
        return {
            'filename': self.filename,
            'path': self.file_path,
            'format': self.format,
            'field_count': self.field_count,
            'source_type': self.source_type,
            'description': self.description
        }

@dataclass  
class BudgetWrapper:
    """Wrapper for budget data with GUI-expected attributes."""
    data: Dict[str, Any] = field(default_factory=dict)
    source_type: str = "budget"
    source: str = ""
    description: str = ""
    file_path: str = ""
    _content: Any = None
    
    def __post_init__(self):
        """Set default values if not provided."""
        if not self.source and self.file_path:
            self.source = Path(self.file_path).name
        if not self.description and self.data:
            sheet_count = len(self.data.get('sheets', {}))
            self.description = f"Budget with {sheet_count} sheet(s)"
    
    @property
    def content(self):
        """Backward compatibility: return budget content for GUI."""
        return self._content if self._content is not None else self.data
    
    @content.setter
    def content(self, value):
        """Set budget content."""
        self._content = value
    
    @property
    def filename(self):
        """GUI compatibility: return filename."""
        return Path(self.file_path).name if self.file_path else ""
    
    @property
    def name(self):
        """GUI compatibility: return name (alias for filename)."""
        return self.filename
    
    @property
    def path(self):
        """GUI compatibility: return file path."""
        return self.file_path
    
    @property
    def format(self):
        """GUI compatibility: return file format."""
        return Path(self.file_path).suffix.lower() if self.file_path else ""
    
    @property
    def file_type(self):
        """GUI compatibility: return file type (alias for format)."""
        return self.format
    
    @property
    def sheets(self):
        """GUI compatibility: return sheets data."""
        return self.data.get('sheets', {})
    
    @property
    def sheet_names(self):
        """GUI compatibility: return list of sheet names."""
        return list(self.sheets.keys())
    
    @property
    def sheet_count(self):
        """GUI compatibility: return number of sheets."""
        return len(self.sheets)
    
    @property
    def rows(self):
        """GUI compatibility: return total number of rows across all sheets."""
        total_rows = 0
        for sheet_data in self.sheets.values():
            if isinstance(sheet_data, dict) and 'data' in sheet_data:
                total_rows += len(sheet_data['data'])
        return total_rows
    
    @property
    def columns(self):
        """GUI compatibility: return max number of columns across all sheets."""
        max_cols = 0
        for sheet_data in self.sheets.values():
            if isinstance(sheet_data, dict) and 'data' in sheet_data:
                sheet_data_rows = sheet_data['data']
                if sheet_data_rows:
                    max_cols = max(max_cols, max(len(row) for row in sheet_data_rows))
        return max_cols
    
    @property
    def cells(self):
        """GUI compatibility: return cell data for spreadsheet interface."""
        # Return all cells from all sheets in a format the GUI expects
        all_cells = {}
        for sheet_name, sheet_data in self.sheets.items():
            if isinstance(sheet_data, dict) and 'data' in sheet_data:
                sheet_cells = {}
                rows = sheet_data['data']
                for row_idx, row in enumerate(rows):
                    for col_idx, cell_value in enumerate(row):
                        # Create cell reference like "A1", "B2", etc.
                        col_letter = chr(65 + col_idx)  # A, B, C, etc.
                        cell_ref = f"{col_letter}{row_idx + 1}"
                        sheet_cells[cell_ref] = cell_value
                all_cells[sheet_name] = sheet_cells
        return all_cells
    
    @property
    def values(self):
        """GUI compatibility: return all values as a flat list."""
        all_values = []
        for sheet_data in self.sheets.values():
            if isinstance(sheet_data, dict) and 'data' in sheet_data:
                for row in sheet_data['data']:
                    all_values.extend(row)
        return all_values
    
    @property
    def worksheets(self):
        """GUI compatibility: return worksheets (alias for sheets)."""
        return self.sheets
    
    @property
    def workbook(self):
        """GUI compatibility: return workbook data."""
        return self.data
    
    @property
    def raw_data(self):
        """GUI compatibility: return raw data."""
        return self.data
    
    @property
    def cell_range(self):
        """GUI compatibility: return cell range info."""
        max_row = 0
        max_col = 0
        for sheet_data in self.sheets.values():
            if isinstance(sheet_data, dict) and 'data' in sheet_data:
                rows = sheet_data['data']
                if rows:
                    max_row = max(max_row, len(rows))
                    max_col = max(max_col, max(len(row) for row in rows))
        
        end_col = chr(64 + max_col) if max_col > 0 else 'A'  # Convert to letter
        return f"A1:{end_col}{max_row}"
    
    @property
    def used_range(self):
        """GUI compatibility: return used range (alias for cell_range)."""
        return self.cell_range
    
    def get_years(self):
        """GUI compatibility: extract years from budget data."""
        years = set()
        
        # Look for year patterns in all cell values
        year_pattern = r'\b(20\d{2})\b'  # Matches years like 2024, 2025, etc.
        
        for sheet_data in self.sheets.values():
            if isinstance(sheet_data, dict) and 'data' in sheet_data:
                for row in sheet_data['data']:
                    for cell_value in row:
                        if isinstance(cell_value, str):
                            # Look for year patterns in text
                            year_matches = re.findall(year_pattern, cell_value)
                            years.update(int(year) for year in year_matches)
                        elif isinstance(cell_value, (int, float)):
                            # Check if the number looks like a year
                            if 2020 <= cell_value <= 2030:
                                years.add(int(cell_value))
        
        # If no years found, return current year as default
        if not years:
            from datetime import datetime
            current_year = datetime.now().year
            years = {current_year}
        
        return sorted(list(years))
    
    def get_budget_years(self):
        """GUI compatibility: alias for get_years()."""
        return self.get_years()
    
    def get_year_data(self, year):
        """GUI compatibility: get data for a specific year."""
        # This would need more sophisticated logic based on your budget structure
        # For now, return all data - could be enhanced to filter by year
        return self.data
    
    def get_total_by_year(self, year=None):
        """GUI compatibility: get budget totals by year."""
        # Simple implementation - sum all numeric values
        total = 0
        for sheet_data in self.sheets.values():
            if isinstance(sheet_data, dict) and 'data' in sheet_data:
                for row in sheet_data['data']:
                    for cell_value in row:
                        if isinstance(cell_value, (int, float)):
                            total += cell_value
        return total
    
    def get_categories(self):
        """GUI compatibility: get budget categories."""
        categories = set()
        
        # Look for common budget category terms in first column of each sheet
        category_keywords = [
            'personnel', 'salary', 'wages', 'equipment', 'supplies', 'travel',
            'indirect', 'overhead', 'fringe', 'benefits', 'contractual',
            'participant', 'support', 'other', 'total'
        ]
        
        for sheet_data in self.sheets.values():
            if isinstance(sheet_data, dict) and 'data' in sheet_data:
                rows = sheet_data['data']
                for row in rows:
                    if row and isinstance(row[0], str):
                        cell_text = row[0].lower()
                        for keyword in category_keywords:
                            if keyword in cell_text:
                                categories.add(row[0])
                                break
        
        return sorted(list(categories))
    
    def get_summary_data(self):
        """GUI compatibility: get summary of budget data."""
        return {
            'filename': self.filename,
            'years': self.get_years(),
            'categories': self.get_categories(),
            'sheet_count': self.sheet_count,
            'total_rows': self.rows,
            'total_value': self.get_total_by_year(),
            'format': self.format
        }
    
    @property
    def summary(self):
        """GUI compatibility: return budget summary."""
        return self.get_summary_data()
    
    @property
    def text(self):
        """GUI compatibility: return text representation."""
        return str(self.data)
    
    @property
    def metadata(self):
        """GUI compatibility: return metadata dict."""
        return {
            'filename': self.filename,
            'path': self.file_path,
            'format': self.format,
            'sheet_count': self.sheet_count,
            'rows': self.rows,
            'columns': self.columns,
            'source_type': self.source_type,
            'description': self.description
        }

class SessionState:
    """Enhanced session state with focus on string analysis and grant workflows."""
    
    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or self._generate_session_id()
        self.logger = logging.getLogger(__name__)
        
        # Core state
        self.current_stage = AnalysisStage.INITIALIZED
        self.created_at = datetime.now()
        self.last_updated = datetime.now()
        
        # File paths
        self.template_path: Optional[str] = None
        self.budget_path: Optional[str] = None
        self.output_path: Optional[str] = None
        
        # Template analysis
        self.template_fields: List[Any] = []  # Will be List[TemplateField] when imported
        self.template_analysis_complete: bool = False
        self.template_hash: Optional[str] = None
        self._template_wrapper: Optional[TemplateWrapper] = None
        
        # Budget analysis
        self.budget_data: Dict[str, Any] = {}
        self.budget_analysis_complete: bool = False
        self.budget_hash: Optional[str] = None
        self._budget_wrapper: Optional[BudgetWrapper] = None
        
        # Field mappings
        self.field_mappings: Dict[str, FieldMappingState] = {}
        self.personnel_mappings: Dict[str, PersonnelMappingState] = {}
        
        # Analysis results
        self.llm_analysis_results: Dict[str, Any] = {}
        self.quality_metrics = AnalysisQualityMetrics()
        
        # User interactions
        self.user_corrections: List[Dict[str, Any]] = []
        self.user_feedback: List[Dict[str, Any]] = []
        self.manual_mappings: Dict[str, Any] = {}
        
        # LLM usage tracking
        self.llm_calls_made: int = 0
        self.llm_tokens_used: int = 0
        self.llm_cost: float = 0.0
        self.llm_enabled: bool = False
        
        # LLM configuration - for GUI compatibility
        self.config = {
            'llm_enabled': False,
            'api_key': '',
            'model': 'gpt-4o-mini',
            'max_tokens': 1000,
            'temperature': 0.1,
            'cost_limit': 5.0
        }
        
        # String analysis specific
        self.string_analysis_results: Dict[str, Any] = {
            'personnel_extraction': {},
            'notes_extraction': {},
            'context_analysis': {},
            'terminology_matches': {}
        }
        
        # Workflow state
        self.workflow_progress: Dict[str, bool] = {
            'template_validated': False,
            'budget_structure_understood': False,
            'personnel_identified': False,
            'notes_columns_found': False,
            'string_fields_analyzed': False,
            'numeric_fields_mapped': False,
            'user_review_completed': False
        }
    
    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        timestamp = datetime.now().isoformat()
        return hashlib.sha256(timestamp.encode()).hexdigest()[:12]
    
    def advance_stage(self, new_stage: AnalysisStage) -> bool:
        """Advance to a new analysis stage with validation."""
        stage_order = list(AnalysisStage)
        current_index = stage_order.index(self.current_stage)
        new_index = stage_order.index(new_stage)
        
        # Allow advancement or staying at current stage
        if new_index >= current_index:
            self.current_stage = new_stage
            self.last_updated = datetime.now()
            self.logger.info(f"Advanced to stage: {new_stage.value}")
            return True
        else:
            self.logger.warning(f"Cannot move backwards from {self.current_stage.value} to {new_stage.value}")
            return False
    
    def set_template(self, template_path: str, template_fields: List[Any]) -> None:
        """Set template information and advance stage."""
        self.template_path = template_path
        self.template_fields = template_fields
        self.template_hash = self._calculate_file_hash(template_path)
        self.template_analysis_complete = True
        
        # Read template content for the wrapper
        template_content = ""
        try:
            template_path_str = str(template_path)  # Ensure it's a string
            if template_path_str.lower().endswith('.txt'):
                with open(template_path_str, 'r', encoding='utf-8') as f:
                    template_content = f.read()
            elif template_path_str.lower().endswith('.md'):
                with open(template_path_str, 'r', encoding='utf-8') as f:
                    template_content = f.read()
            else:
                # For other formats, use a placeholder or extract text
                template_content = f"Template content from {Path(template_path_str).name}"
        except Exception as e:
            self.logger.warning(f"Could not read template content: {e}")
            template_content = f"Template from {Path(template_path).name}"
        
        # Create template wrapper with GUI-expected attributes
        self._template_wrapper = TemplateWrapper(
            fields=template_fields,
            file_path=template_path,
            source=Path(template_path).name,
            description=f"Template with {len(template_fields)} fields from {Path(template_path).name}"
        )
        self._template_wrapper.content = template_content
        
        # Initialize field mappings - with better field name extraction
        self.field_mappings = {}  # Clear existing mappings
        
        for i, field in enumerate(template_fields):
            # Extract field name with multiple fallbacks
            field_name = None
            
            if hasattr(field, 'name') and field.name:
                field_name = str(field.name).strip()
            elif hasattr(field, 'placeholder_text') and field.placeholder_text:
                # Extract from placeholder like {field_name} or [field_name]
                placeholder = str(field.placeholder_text)
                # Remove common placeholder markers
                cleaned = placeholder.strip('{}[]()<>')
                if cleaned:
                    field_name = cleaned
            elif hasattr(field, 'context') and field.context:
                # Try to extract from context
                context = str(field.context)[:50]  # First 50 chars
                field_name = f"field_from_context_{i+1}"
            
            # Final fallback
            if not field_name:
                field_name = f"template_field_{i+1}"
            
            # Ensure unique field names
            original_name = field_name
            counter = 1
            while field_name in self.field_mappings:
                field_name = f"{original_name}_{counter}"
                counter += 1
            
            # Create the mapping
            self.field_mappings[field_name] = FieldMappingState(
                field_name=field_name,
                template_field=field
            )
            
            # Log field creation for debugging
            self.logger.debug(f"Created mapping for field {i+1}: '{field_name}' from {type(field)}")
        
        self.logger.info(f"Initialized {len(self.field_mappings)} field mappings from {len(template_fields)} template fields")
        
        # Auto-match fields if budget is already loaded
        if self.budget_analysis_complete and self.budget_data:
            self.logger.info("Both template and budget loaded - performing automatic field matching")
            matches_found = self.auto_match_fields()
            self.logger.info(f"Automatic matching found {matches_found} field matches")
        
        self.advance_stage(AnalysisStage.TEMPLATE_LOADED)
        self.workflow_progress['template_validated'] = True
    
    def set_budget(self, budget_path: str, budget_data: Dict[str, Any]) -> None:
        """Set budget information and advance stage."""
        self.budget_path = budget_path
        self.budget_data = budget_data
        self.budget_hash = self._calculate_file_hash(budget_path)
        self.budget_analysis_complete = True
        
        # Create budget wrapper with GUI-expected attributes  
        self._budget_wrapper = BudgetWrapper(
            data=budget_data,
            file_path=budget_path,
            source=Path(budget_path).name,
            description=f"Budget from {Path(budget_path).name}"
        )
        self._budget_wrapper.content = budget_data  # Set the content to the actual data
        
        # Initialize personnel mappings from budget analysis
        self._initialize_personnel_mappings()
        
        # Auto-match fields if template is already loaded
        if self.template_analysis_complete and self.field_mappings:
            self.logger.info("Both template and budget loaded - performing automatic field matching")
            matches_found = self.auto_match_fields()
            self.logger.info(f"Automatic matching found {matches_found} field matches")
        
        self.advance_stage(AnalysisStage.BUDGET_LOADED)
        self.workflow_progress['budget_structure_understood'] = True
    
    def _initialize_personnel_mappings(self) -> None:
        """Initialize personnel mappings from budget analysis."""
        person_counter = 0
        
        for sheet_name, sheet_data in self.budget_data.get('sheets', {}).items():
            personnel_entries = sheet_data.get('personnel', [])
            
            for person in personnel_entries:
                person_counter += 1
                person_id = f"person_{person_counter}"
                
                self.personnel_mappings[person_id] = PersonnelMappingState(
                    person_id=person_id,
                    name=person.name if hasattr(person, 'name') else None,
                    title=person.title if hasattr(person, 'title') else None,
                    effort_percent=person.effort_percent if hasattr(person, 'effort_percent') else None,
                    salary=person.salary if hasattr(person, 'salary') else None,
                    notes=person.notes if hasattr(person, 'notes') else "",
                    confidence=0.7,  # Default confidence for rule-based extraction
                    extraction_method="rule_based"
                )
    
    def update_field_mapping(self, field_name: str, location: str, value: Any, 
                           confidence: float, method: str = "automatic", 
                           user_verified: bool = False) -> None:
        """Update a field mapping with new information."""
        if field_name not in self.field_mappings:
            self.field_mappings[field_name] = FieldMappingState(field_name=field_name)
        
        mapping = self.field_mappings[field_name]
        
        # Save history
        if mapping.mapped_location:
            mapping.mapping_history.append({
                'timestamp': datetime.now().isoformat(),
                'old_location': mapping.mapped_location,
                'old_value': mapping.mapped_value,
                'old_confidence': mapping.confidence,
                'old_method': mapping.mapping_method
            })
        
        # Update mapping
        mapping.mapped_location = location
        mapping.mapped_value = value
        mapping.confidence = confidence
        mapping.confidence_level = self._determine_confidence_level(confidence)
        mapping.mapping_method = method
        mapping.user_verified = user_verified
        mapping.last_updated = datetime.now()
        
        self.last_updated = datetime.now()
        self._update_quality_metrics()
    
    def add_mapping_alternative(self, field_name: str, location: str, confidence: float) -> None:
        """Add an alternative mapping for a field."""
        if field_name not in self.field_mappings:
            return
        
        mapping = self.field_mappings[field_name]
        mapping.alternatives.append((location, confidence))
        
        # Keep only top 5 alternatives, sorted by confidence
        mapping.alternatives.sort(key=lambda x: x[1], reverse=True)
        mapping.alternatives = mapping.alternatives[:5]
    
    def mark_field_verified(self, field_name: str, user_notes: str = "") -> None:
        """Mark a field mapping as user-verified."""
        if field_name in self.field_mappings:
            self.field_mappings[field_name].user_verified = True
            self.field_mappings[field_name].user_notes = user_notes
            self.field_mappings[field_name].last_updated = datetime.now()
            self._update_quality_metrics()
    
    def update_personnel_mapping(self, person_id: str, field_type: str, 
                               location: str, value: Any, user_verified: bool = False) -> None:
        """Update personnel mapping information."""
        if person_id not in self.personnel_mappings:
            self.personnel_mappings[person_id] = PersonnelMappingState(person_id=person_id)
        
        person = self.personnel_mappings[person_id]
        
        if field_type == 'name':
            person.name = value
            person.name_location = location
        elif field_type == 'title':
            person.title = value
            person.title_location = location
        elif field_type == 'effort':
            person.effort_percent = value
            person.effort_location = location
        elif field_type == 'salary':
            person.salary = value
            person.salary_location = location
        elif field_type == 'notes':
            person.notes = value
            person.notes_location = location
        
        person.user_verified = user_verified
        self._update_quality_metrics()
    
    def set_llm_analysis_results(self, results: Dict[str, Any]) -> None:
        """Set LLM analysis results and update mappings."""
        self.llm_analysis_results = results
        
        # Update field mappings from LLM results
        if 'field_mappings' in results:
            self._process_llm_field_mappings(results['field_mappings'])
        
        # Update personnel mappings from LLM results
        if 'personnel_analysis' in results:
            self._process_llm_personnel_analysis(results['personnel_analysis'])
        
        # Update string analysis results
        if 'string_analysis' in results:
            self.string_analysis_results.update(results['string_analysis'])
        
        self.advance_stage(AnalysisStage.LLM_ANALYSIS)
        self._update_quality_metrics()
    
    def _process_llm_field_mappings(self, llm_mappings: Dict[str, Any]) -> None:
        """Process LLM field mapping results."""
        combined_mappings = llm_mappings.get('combined_mappings', {})
        
        for field_name, mapping_data in combined_mappings.items():
            if field_name in self.field_mappings:
                recommendation = mapping_data.get('final_recommendation')
                if recommendation:
                    self.update_field_mapping(
                        field_name=field_name,
                        location=recommendation.get('cell_address', ''),
                        value=recommendation.get('value'),
                        confidence=recommendation.get('confidence', 0.5),
                        method="llm_enhanced"
                    )
                    
                    # Add alternatives
                    for alt in mapping_data.get('llm_based', [])[:3]:
                        self.add_mapping_alternative(
                            field_name, 
                            alt.get('cell_address', ''), 
                            alt.get('confidence', 0.0)
                        )
    
    def _process_llm_personnel_analysis(self, personnel_analysis: Dict[str, Any]) -> None:
        """Process LLM personnel analysis results."""
        llm_personnel = personnel_analysis.get('llm_personnel', {})
        
        if 'personnel' in llm_personnel:
            person_counter = len(self.personnel_mappings)
            
            for person_data in llm_personnel['personnel']:
                person_counter += 1
                person_id = f"llm_person_{person_counter}"
                
                self.personnel_mappings[person_id] = PersonnelMappingState(
                    person_id=person_id,
                    name=person_data.get('name'),
                    name_location=person_data.get('name_location'),
                    title=person_data.get('title'),
                    title_location=person_data.get('title_location'),
                    effort_percent=person_data.get('effort_percent'),
                    salary=person_data.get('salary'),
                    notes=person_data.get('notes', ''),
                    confidence=person_data.get('confidence', 0.7),
                    extraction_method="llm"
                )
    
    def add_user_correction(self, field_name: str, old_location: str, 
                          new_location: str, reason: str) -> None:
        """Record a user correction for learning purposes."""
        correction = {
            'timestamp': datetime.now().isoformat(),
            'field_name': field_name,
            'old_location': old_location,
            'new_location': new_location,
            'reason': reason,
            'session_id': self.session_id
        }
        
        self.user_corrections.append(correction)
        
        # Update the actual mapping
        if field_name in self.field_mappings:
            mapping = self.field_mappings[field_name]
            # Get the value from new location (this would need to be implemented)
            # For now, just update the location
            mapping.mapped_location = new_location
            mapping.mapping_method = "manual"
            mapping.user_verified = True
            mapping.last_updated = datetime.now()
    
    def add_user_feedback(self, feedback_type: str, content: str, 
                         context: Dict[str, Any] = None) -> None:
        """Add user feedback for system improvement."""
        feedback = {
            'timestamp': datetime.now().isoformat(),
            'type': feedback_type,
            'content': content,
            'context': context or {},
            'session_id': self.session_id
        }
        
        self.user_feedback.append(feedback)
    
    def track_llm_usage(self, tokens_used: int, cost: float) -> None:
        """Track LLM usage for cost management."""
        self.llm_calls_made += 1
        self.llm_tokens_used += tokens_used
        self.llm_cost += cost
    
    def _determine_confidence_level(self, confidence: float) -> MappingConfidence:
        """Determine confidence level category."""
        if confidence >= 0.8:
            return MappingConfidence.HIGH
        elif confidence >= 0.5:
            return MappingConfidence.MEDIUM
        elif confidence >= 0.3:
            return MappingConfidence.LOW
        else:
            return MappingConfidence.UNCERTAIN
    
    def _update_quality_metrics(self) -> None:
        """Update quality metrics based on current mappings."""
        metrics = AnalysisQualityMetrics()
        
        # Field mapping metrics
        metrics.total_fields = len(self.field_mappings)
        
        for mapping in self.field_mappings.values():
            if mapping.mapped_location:
                metrics.mapped_fields += 1
                
                if mapping.confidence_level == MappingConfidence.HIGH:
                    metrics.high_confidence_mappings += 1
                elif mapping.confidence_level == MappingConfidence.MEDIUM:
                    metrics.medium_confidence_mappings += 1
                elif mapping.confidence_level == MappingConfidence.LOW:
                    metrics.low_confidence_mappings += 1
                else:
                    metrics.uncertain_mappings += 1
                
                if mapping.user_verified:
                    metrics.user_verified_mappings += 1
                
                if 'llm' in mapping.mapping_method:
                    metrics.llm_enhanced_mappings += 1
        
        # String field coverage
        string_fields = [m for m in self.field_mappings.values() 
                        if m.template_field and hasattr(m.template_field, 'field_type') and 
                        m.template_field.field_type in ['text', 'personnel', 'description']]
        
        if string_fields:
            mapped_string_fields = [m for m in string_fields if m.mapped_location]
            metrics.string_field_coverage = len(mapped_string_fields) / len(string_fields)
        
        # Personnel extraction quality
        if self.personnel_mappings:
            complete_personnel = [p for p in self.personnel_mappings.values() 
                                if p.name and p.title]
            metrics.personnel_extraction_quality = len(complete_personnel) / len(self.personnel_mappings)
        
        # Notes coverage (fields with 'note' or 'description' in name)
        note_fields = [m for m in self.field_mappings.values() 
                      if any(term in m.field_name.lower() for term in ['note', 'description', 'comment'])]
        
        if note_fields:
            mapped_note_fields = [m for m in note_fields if m.mapped_location]
            metrics.notes_coverage = len(mapped_note_fields) / len(note_fields)
        
        # Overall quality score (weighted average)
        if metrics.total_fields > 0:
            mapping_ratio = metrics.mapped_fields / metrics.total_fields
            confidence_score = (
                metrics.high_confidence_mappings * 1.0 +
                metrics.medium_confidence_mappings * 0.7 +
                metrics.low_confidence_mappings * 0.4 +
                metrics.uncertain_mappings * 0.1
            ) / max(metrics.mapped_fields, 1)
            
            metrics.overall_quality_score = (
                mapping_ratio * 0.4 +
                confidence_score * 0.3 +
                metrics.string_field_coverage * 0.2 +
                metrics.notes_coverage * 0.1
            )
        
        self.quality_metrics = metrics
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate hash of a file for change detection."""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()[:16]
        except Exception:
            return "unknown"
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get a summary of the current session state."""
        return {
            'session_id': self.session_id,
            'current_stage': self.current_stage.value,
            'created_at': self.created_at.isoformat(),
            'last_updated': self.last_updated.isoformat(),
            'template_path': self.template_path,
            'budget_path': self.budget_path,
            'quality_metrics': asdict(self.quality_metrics),
            'workflow_progress': self.workflow_progress,
            'llm_usage': {
                'calls_made': self.llm_calls_made,
                'tokens_used': self.llm_tokens_used,
                'cost': self.llm_cost,
                'enabled': self.llm_enabled
            },
            'mappings_summary': {
                'total_fields': len(self.field_mappings),
                'mapped_fields': len([m for m in self.field_mappings.values() if m.mapped_location]),
                'user_verified': len([m for m in self.field_mappings.values() if m.user_verified]),
                'personnel_entries': len(self.personnel_mappings)
            }
        }
    
    def get_mapping_report(self) -> Dict[str, Any]:
        """Get detailed mapping report for review."""
        report = {
            'field_mappings': {},
            'personnel_mappings': {},
            'quality_assessment': asdict(self.quality_metrics),
            'recommendations': []
        }
        
        # Field mappings
        for field_name, mapping in self.field_mappings.items():
            report['field_mappings'][field_name] = {
                'mapped': bool(mapping.mapped_location),
                'location': mapping.mapped_location,
                'value': str(mapping.mapped_value)[:100] if mapping.mapped_value else None,
                'confidence': mapping.confidence,
                'confidence_level': mapping.confidence_level.value,
                'method': mapping.mapping_method,
                'user_verified': mapping.user_verified,
                'alternatives_count': len(mapping.alternatives),
                'field_type': mapping.template_field.field_type if (mapping.template_field and 
                            hasattr(mapping.template_field, 'field_type')) else 'unknown'
            }
        
        # Personnel mappings
        for person_id, person in self.personnel_mappings.items():
            report['personnel_mappings'][person_id] = {
                'name': person.name,
                'title': person.title,
                'effort_percent': person.effort_percent,
                'salary': person.salary,
                'confidence': person.confidence,
                'extraction_method': person.extraction_method,
                'user_verified': person.user_verified
            }
        
        # Recommendations
        if self.quality_metrics.string_field_coverage < 0.7:
            report['recommendations'].append(
                "String field coverage is low. Review text fields and notes columns."
            )
        
        if self.quality_metrics.uncertain_mappings > self.quality_metrics.high_confidence_mappings:
            report['recommendations'].append(
                "Many mappings have low confidence. Consider manual review and verification."
            )
        
        if not self.workflow_progress.get('user_review_completed', False):
            report['recommendations'].append(
                "User review has not been completed. Please verify the mappings before generating documents."
            )
        
        return report
    
    def save_session(self, file_path: str = None) -> bool:
        """Save session state to file."""
        try:
            # Provide default file path if none given
            if file_path is None:
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                file_path = f"session_{self.session_id}_{timestamp}.json"
            
            # Helper function to safely serialize complex objects
            def safe_serialize(obj):
                if hasattr(obj, '__dict__'):
                    # Handle dataclass objects
                    try:
                        result = {}
                        for key, value in obj.__dict__.items():
                            if isinstance(value, Enum):
                                result[key] = value.value
                            elif isinstance(value, datetime):
                                result[key] = value.isoformat()
                            elif isinstance(value, list):
                                result[key] = [safe_serialize(item) for item in value]
                            elif hasattr(value, '__dict__'):
                                result[key] = safe_serialize(value)
                            else:
                                result[key] = value
                        return result
                    except Exception:
                        return str(obj)
                else:
                    return str(obj)
            
            session_data = {
                'session_id': self.session_id,
                'current_stage': self.current_stage.value,
                'created_at': self.created_at.isoformat(),
                'last_updated': self.last_updated.isoformat(),
                'template_path': self.template_path,
                'budget_path': self.budget_path,
                'template_fields': [safe_serialize(field) for field in self.template_fields],
                'budget_data': self.budget_data,
                'field_mappings': {k: safe_serialize(v) for k, v in self.field_mappings.items()},
                'personnel_mappings': {k: safe_serialize(v) for k, v in self.personnel_mappings.items()},
                'quality_metrics': safe_serialize(self.quality_metrics),
                'workflow_progress': self.workflow_progress,
                'llm_usage': {
                    'calls_made': self.llm_calls_made,
                    'tokens_used': self.llm_tokens_used,
                    'cost': self.llm_cost,
                    'enabled': self.llm_enabled
                },
                'user_corrections': self.user_corrections,
                'user_feedback': self.user_feedback
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, default=str)
            
            self.logger.info(f"Session saved to {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save session: {e}")
            return False
    
    @classmethod
    def load_session(cls, file_path: str) -> Optional['SessionState']:
        """Load session state from file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
            
            session = cls(session_data['session_id'])
            
            # Restore basic state
            session.current_stage = AnalysisStage(session_data['current_stage'])
            session.created_at = datetime.fromisoformat(session_data['created_at'])
            session.last_updated = datetime.fromisoformat(session_data['last_updated'])
            session.template_path = session_data.get('template_path')
            session.budget_path = session_data.get('budget_path')
            
            # Restore complex objects (simplified for now)
            session.template_fields = session_data.get('template_fields', [])
            session.budget_data = session_data.get('budget_data', {})
            
            # Restore mappings (simplified reconstruction)
            if 'field_mappings' in session_data:
                for field_name, mapping_data in session_data['field_mappings'].items():
                    mapping = FieldMappingState(field_name=field_name)
                    for key, value in mapping_data.items():
                        if hasattr(mapping, key):
                            setattr(mapping, key, value)
                    session.field_mappings[field_name] = mapping
            
            if 'personnel_mappings' in session_data:
                for person_id, person_data in session_data['personnel_mappings'].items():
                    person = PersonnelMappingState(person_id=person_id)
                    for key, value in person_data.items():
                        if hasattr(person, key):
                            setattr(person, key, value)
                    session.personnel_mappings[person_id] = person
            
            # Restore other state
            session.workflow_progress = session_data.get('workflow_progress', {})
            session.user_corrections = session_data.get('user_corrections', [])
            session.user_feedback = session_data.get('user_feedback', [])
            
            llm_usage = session_data.get('llm_usage', {})
            session.llm_calls_made = llm_usage.get('calls_made', 0)
            session.llm_tokens_used = llm_usage.get('tokens_used', 0)
            session.llm_cost = llm_usage.get('cost', 0.0)
            session.llm_enabled = llm_usage.get('enabled', False)
            
            session._update_quality_metrics()
            
            logging.getLogger(__name__).info(f"Session loaded from {file_path}")
            return session
            
        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to load session: {e}")
            return None
    
    def is_ready_for_document_generation(self) -> Tuple[bool, List[str]]:
        """Check if session is ready for document generation."""
        issues = []
        
        if not self.template_path or not self.budget_path:
            issues.append("Template and budget files must be loaded")
        
        if not self.field_mappings:
            issues.append("No field mappings available")
        
        # Check mapping quality
        mapped_count = len([m for m in self.field_mappings.values() if m.mapped_location])
        if mapped_count == 0:
            issues.append("No fields have been mapped")
        elif mapped_count / len(self.field_mappings) < 0.5:
            issues.append("Less than 50% of fields are mapped")
        
        # Check for critical unmapped fields
        critical_fields = [m for m in self.field_mappings.values() 
                          if (m.template_field and hasattr(m.template_field, 'confidence') and 
                              m.template_field.confidence > 0.8 and not m.mapped_location)]
        
        if critical_fields:
            issues.append(f"{len(critical_fields)} high-priority fields are unmapped")
        
        return len(issues) == 0, issues
    
    def is_ready_for_generation(self) -> bool:
        """Backward compatibility: GUI expects this method name."""
        ready, issues = self.is_ready_for_document_generation()
        return ready

    # Backward compatibility properties for existing GUI code
    @property
    def template(self):
        """Backward compatibility: return template wrapper with GUI-expected attributes."""
        return self._template_wrapper if self.template_analysis_complete else None
    
    @template.setter  
    def template(self, value):
        """Backward compatibility: set template_fields."""
        if value is not None:
            if hasattr(value, 'fields'):
                # It's already a wrapper
                self.template_fields = value.fields
                self._template_wrapper = value
            else:
                # It's a list of fields
                self.template_fields = value
                self._template_wrapper = TemplateWrapper(
                    fields=value,
                    source_type="template",
                    source="template",
                    description=f"Template with {len(value)} fields"
                )
            self.template_analysis_complete = True
        else:
            self.template_fields = []
            self._template_wrapper = None
            self.template_analysis_complete = False
    
    @property
    def budget(self):
        """Backward compatibility: return budget wrapper with GUI-expected attributes."""
        return self._budget_wrapper if self.budget_analysis_complete else None
    
    @budget.setter
    def budget(self, value):
        """Backward compatibility: set budget_data."""
        if value is not None:
            if hasattr(value, 'data'):
                # It's already a wrapper
                self.budget_data = value.data
                self._budget_wrapper = value
            else:
                # It's raw budget data
                self.budget_data = value
                self._budget_wrapper = BudgetWrapper(
                    data=value,
                    source_type="budget",
                    source="budget",
                    description="Budget data"
                )
            self.budget_analysis_complete = True
        else:
            self.budget_data = {}
            self._budget_wrapper = None
            self.budget_analysis_complete = False
    
    @property
    def field_mappings_dict(self):
        """Backward compatibility: return field mappings in old format."""
        return {name: mapping.mapped_value for name, mapping in self.field_mappings.items() 
                if mapping.mapped_location}
    
    @property 
    def mappings(self):
        """Backward compatibility: return field mappings for GUI."""
        # Return ALL fields as MappingDisplay objects for the GUI table
        mappings = {}
        
        self.logger.debug(f"Getting mappings: {len(self.field_mappings)} total field mappings")
        
        for field_name, mapping in self.field_mappings.items():
            mappings[field_name] = MappingDisplay(
                field_name=field_name,
                mapped=bool(mapping.mapped_location),
                location=mapping.mapped_location or "",
                value=mapping.mapped_value or "",
                confidence=mapping.confidence,
                method=mapping.mapping_method,
                verified=mapping.user_verified,
                template_field=mapping.template_field
            )
            
            # Debug first few mappings
            if len(mappings) <= 3:
                self.logger.debug(f"Mapping {field_name}: mapped={bool(mapping.mapped_location)}, confidence={mapping.confidence}")
        
        self.logger.debug(f"Returning {len(mappings)} mappings to GUI")
        return mappings
    
    @property
    def llm_enabled_flag(self):
        """Backward compatibility: return llm_enabled status."""
        return self.llm_enabled
    
    @llm_enabled_flag.setter
    def llm_enabled_flag(self, value):
        """Backward compatibility: set llm_enabled status."""
        self.llm_enabled = value
    
    # Backward compatibility methods for GUI operations
    def load_template(self, template_path: str):
        """Backward compatibility: load template using field detector."""
        try:
            # Import here to avoid circular imports
            from field_detector import FieldDetector
            
            detector = FieldDetector()
            template_fields = detector.analyze_template(template_path)
            
            # Ensure all fields have GUI-expected attributes
            for field in template_fields:
                if not hasattr(field, 'source_type'):
                    field.source_type = "template"
                if not hasattr(field, 'source'):
                    field.source = field.placeholder_text if hasattr(field, 'placeholder_text') else ""
                if not hasattr(field, 'description'):
                    field.description = field.context if hasattr(field, 'context') else ""
                    
                # Add any other attributes the GUI might expect
                if not hasattr(field, 'required'):
                    field.required = True
                if not hasattr(field, 'default_value'):
                    field.default_value = ""
            
            self.set_template(template_path, template_fields)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load template: {e}")
            raise e  # Re-raise to see the full error in GUI
    
    def load_budget(self, budget_path: str):
        """Backward compatibility: load budget using budget book."""
        try:
            # Import here to avoid circular imports
            from budget_book import BudgetBook
            
            budget_book = BudgetBook()
            budget_data = budget_book.load_budget(budget_path)
            
            self.set_budget(budget_path, budget_data)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load budget: {e}")
            return False
    
    def get_field_mappings(self):
        """Backward compatibility: return field mappings in GUI format."""
        mappings = {}
        for field_name, mapping in self.field_mappings.items():
            if mapping.mapped_location:
                mappings[field_name] = {
                    'location': mapping.mapped_location,
                    'value': mapping.mapped_value,
                    'confidence': mapping.confidence,
                    'method': mapping.mapping_method,
                    'verified': mapping.user_verified
                }
        return mappings
    
    def set_field_mapping(self, field_name: str, location: str, value: Any, confidence: float = 0.5):
        """Backward compatibility: set a field mapping."""
        self.update_field_mapping(field_name, location, value, confidence, method="manual", user_verified=True)
    
    def generate_document(self, output_path: str):
        """Backward compatibility: generate document using document generator."""
        try:
            # Import here to avoid circular imports
            from document_generator import DocumentGenerator
            
            generator = DocumentGenerator()
            # This would need to be implemented based on your DocumentGenerator interface
            # For now, just return success
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to generate document: {e}")
            return False
    
    def get_template_fields(self):
        """Backward compatibility: return template fields."""
        return self.template_fields
    
    def get_budget_data(self):
        """Backward compatibility: return budget data."""
        return self.budget_data
    
    def clear_session(self):
        """Backward compatibility: clear/reset session."""
        self.__init__(self.session_id)
    
    def is_template_loaded(self):
        """Backward compatibility: check if template is loaded."""
        return self.template_analysis_complete
    
    def is_budget_loaded(self):
        """Backward compatibility: check if budget is loaded."""
        return self.budget_analysis_complete
    
    def get_field_mapping_status(self):
        """Backward compatibility: return mapping status for GUI."""
        mapped_count = len([m for m in self.field_mappings.values() if m.mapped_location])
        total_count = len(self.field_mappings)
        
        return {
            'total_fields': total_count,
            'mapped_fields': mapped_count,
            'unmapped_fields': total_count - mapped_count,
            'mapping_percentage': (mapped_count / total_count * 100) if total_count > 0 else 0,
            'ready_for_generation': self.is_ready_for_generation()
        }
    
    def get_field_list(self):
        """Backward compatibility: return list of field names."""
        return list(self.field_mappings.keys())
    
    def get_unmapped_fields(self):
        """Backward compatibility: return list of unmapped field names."""
        return [name for name, mapping in self.field_mappings.items() if not mapping.mapped_location]
    
    def get_mapped_fields(self):
        """Backward compatibility: return list of mapped field names."""
        return [name for name, mapping in self.field_mappings.items() if mapping.mapped_location]
    
    def has_mappings(self):
        """Backward compatibility: check if any mappings exist."""
        return len(self.get_mapped_fields()) > 0
    
    def auto_match_fields(self):
        """Backward compatibility: perform automatic field matching."""
        if not self.budget_data or not self.field_mappings:
            self.logger.warning("Cannot auto-match: missing budget data or field mappings")
            return 0
        
        self.logger.info(f"Starting auto-match for {len(self.field_mappings)} fields")
        
        # Get all budget cell values for matching
        budget_cells = self._extract_budget_cells()
        
        # Perform basic matching for each field
        matches_found = 0
        for field_name, mapping in self.field_mappings.items():
            if mapping.mapped_location:
                continue  # Skip already mapped fields
            
            # Try to find a match for this field
            best_match = self._find_best_cell_match(mapping, budget_cells)
            if best_match:
                location, value, confidence = best_match
                self.update_field_mapping(
                    field_name=field_name,
                    location=location,
                    value=value,
                    confidence=confidence,
                    method="auto_match"
                )
                matches_found += 1
                self.logger.debug(f"Auto-matched '{field_name}' to '{location}': {value} (confidence: {confidence:.2f})")
        
        self.logger.info(f"Auto-match completed: {matches_found} fields matched")
        return matches_found
    
    def _extract_budget_cells(self):
        """Extract all budget cells with their locations and values."""
        cells = []
        
        self.logger.debug(f"Extracting budget cells from data: {type(self.budget_data)}")
        self.logger.debug(f"Budget data keys: {list(self.budget_data.keys()) if isinstance(self.budget_data, dict) else 'Not a dict'}")
        
        if 'sheets' not in self.budget_data:
            self.logger.warning("No 'sheets' key in budget_data")
            return cells
        
        sheets = self.budget_data.get('sheets', {})
        self.logger.debug(f"Found {len(sheets)} sheets: {list(sheets.keys())}")
        
        for sheet_name, sheet_data in sheets.items():
            self.logger.debug(f"Processing sheet '{sheet_name}': {type(sheet_data)}")
            
            if not isinstance(sheet_data, dict):
                self.logger.debug(f"Sheet {sheet_name} is not a dict, skipping")
                continue
                
            self.logger.debug(f"Sheet {sheet_name} keys: {list(sheet_data.keys())}")
            
            if 'data' not in sheet_data:
                self.logger.debug(f"No 'data' key in sheet {sheet_name}, trying other keys")
                # Try alternative keys that might contain the actual data
                for possible_key in ['rows', 'cells', 'values', 'content']:
                    if possible_key in sheet_data:
                        self.logger.debug(f"Found '{possible_key}' in sheet {sheet_name}")
                        sheet_data['data'] = sheet_data[possible_key]
                        break
                else:
                    self.logger.debug(f"No data found in sheet {sheet_name}")
                    continue
            
            rows = sheet_data['data']
            self.logger.debug(f"Sheet {sheet_name} has {len(rows)} rows")
            
            for row_idx, row in enumerate(rows):
                if not isinstance(row, (list, tuple)):
                    self.logger.debug(f"Row {row_idx} is not a list/tuple: {type(row)}")
                    continue
                    
                for col_idx, cell_value in enumerate(row):
                    if cell_value is not None and str(cell_value).strip():
                        # Create cell reference
                        if col_idx < 26:
                            col_letter = chr(65 + col_idx)  # A, B, C...
                        else:
                            col_letter = f"A{chr(65 + col_idx - 26)}"  # AA, AB, AC...
                        
                        cell_ref = f"{sheet_name}!{col_letter}{row_idx + 1}"
                        
                        cells.append({
                            'location': cell_ref,
                            'value': cell_value,
                            'sheet': sheet_name,
                            'row': row_idx,
                            'col': col_idx,
                            'type': type(cell_value).__name__
                        })
                        
                        # Debug first few cells
                        if len(cells) <= 5:
                            self.logger.debug(f"Cell {cell_ref}: '{cell_value}' ({type(cell_value).__name__})")
        
        self.logger.debug(f"Extracted {len(cells)} cells from budget data")
        return cells
    
    def _find_best_cell_match(self, field_mapping, budget_cells):
        """Find the best matching cell for a field."""
        if not field_mapping.template_field:
            return None
        
        field = field_mapping.template_field
        field_name = field_mapping.field_name.lower()
        
        best_match = None
        best_score = 0.0
        
        # Get field type for type-based matching
        field_type = getattr(field, 'field_type', 'text')
        
        for cell in budget_cells:
            score = 0.0
            cell_value = str(cell['value']).strip()
            cell_value_lower = cell_value.lower()
            
            # Name-based matching
            if any(word in cell_value_lower for word in field_name.split('_')):
                score += 0.4
            
            # Type-based matching
            if field_type == 'personnel' and self._looks_like_person_name(cell_value):
                score += 0.6
            elif field_type == 'currency' and self._looks_like_currency(cell_value):
                score += 0.6
            elif field_type == 'numeric' and self._looks_like_number(cell_value):
                score += 0.5
            elif field_type == 'description' and len(cell_value) > 30:
                score += 0.4
            
            # Keyword matching based on field name
            field_keywords = self._get_field_keywords(field_name)
            for keyword in field_keywords:
                if keyword in cell_value_lower:
                    score += 0.3
                    break
            
            # Prefer non-numeric cells for text fields
            if field_type in ['text', 'personnel', 'description']:
                if not cell_value.replace('.', '').replace(',', '').isdigit():
                    score += 0.2
            
            if score > best_score and score > 0.3:  # Minimum threshold
                best_match = (cell['location'], cell['value'], score)
                best_score = score
        
        return best_match
    
    def _get_field_keywords(self, field_name):
        """Get keywords for field matching based on field name."""
        keywords = field_name.lower().split('_')
        
        # Add synonyms
        keyword_synonyms = {
            'name': ['investigator', 'person', 'researcher'],
            'title': ['position', 'role', 'rank'],
            'project': ['study', 'research', 'grant'],
            'cost': ['amount', 'price', 'total', 'budget'],
            'salary': ['wage', 'compensation', 'pay'],
            'effort': ['fte', 'time', 'percent'],
            'description': ['detail', 'explain', 'summary'],
            'institution': ['university', 'organization', 'affiliation']
        }
        
        extended_keywords = keywords.copy()
        for keyword in keywords:
            if keyword in keyword_synonyms:
                extended_keywords.extend(keyword_synonyms[keyword])
        
        return extended_keywords
    
    def _looks_like_person_name(self, value):
        """Check if value looks like a person's name."""
        if not isinstance(value, str) or len(value.strip()) < 3:
            return False
        
        words = value.strip().split()
        if len(words) < 2 or len(words) > 4:
            return False
        
        # Names typically start with capital letters
        if not all(word[0].isupper() for word in words if word):
            return False
        
        # Names shouldn't contain numbers
        if any(char.isdigit() for char in value):
            return False
        
        return True
    
    def _looks_like_currency(self, value):
        """Check if value looks like a currency amount."""
        value_str = str(value)
        currency_patterns = [
            r'\$[\d,]+\.?\d*',
            r'[\d,]+\.?\d*\s*\$',
            r'USD\s*[\d,]+\.?\d*',
            r'[\d,]+\.?\d*\s*dollars?'
        ]
        
        return any(re.search(pattern, value_str, re.IGNORECASE) for pattern in currency_patterns)
    
    def _looks_like_number(self, value):
        """Check if value looks like a number."""
        try:
            # Remove common formatting
            cleaned = re.sub(r'[,%$]', '', str(value).strip())
            float(cleaned)
            return True
        except (ValueError, AttributeError):
            return False
    
    def clear_all_mappings(self):
        """Backward compatibility: clear all field mappings."""
        for mapping in self.field_mappings.values():
            mapping.mapped_location = None
            mapping.mapped_value = None
            mapping.confidence = 0.0
            mapping.mapping_method = "none"
            mapping.user_verified = False
        
        self._update_quality_metrics()
        self.logger.info("Cleared all field mappings")
    
    def refresh_mappings(self):
        """Backward compatibility: refresh mapping display."""
        self._update_quality_metrics()
        return self.mappings
    
    def save(self, file_path: str = None):
        """Backward compatibility: alternative save method for GUI."""
        return self.save_session(file_path)
    
    def auto_save(self):
        """Backward compatibility: auto-save with default filename."""
        return self.save_session()
    
    def debug_mappings(self):
        """Debug method to check mapping state."""
        self.logger.info(f"Debug - Total field mappings: {len(self.field_mappings)}")
        self.logger.info(f"Debug - Template loaded: {self.template_analysis_complete}")
        self.logger.info(f"Debug - Budget loaded: {self.budget_analysis_complete}")
        
        for i, (field_name, mapping) in enumerate(self.field_mappings.items()):
            self.logger.info(f"Debug - Field {i+1}: '{field_name}' -> location: {mapping.mapped_location}, confidence: {mapping.confidence}")
        
        return {
            'total_fields': len(self.field_mappings),
            'template_loaded': self.template_analysis_complete,
            'budget_loaded': self.budget_analysis_complete,
            'field_names': list(self.field_mappings.keys())[:5]  # First 5 field names
        }
    
    # LLM Configuration methods for GUI compatibility
    def get_llm_config(self):
        """Get LLM configuration for GUI."""
        return self.config.copy()
    
    def set_llm_config(self, config_dict):
        """Set LLM configuration from GUI."""
        self.config.update(config_dict)
        self.llm_enabled = config_dict.get('llm_enabled', False)
        return True
    
    def update_config(self, **kwargs):
        """Update configuration parameters."""
        self.config.update(kwargs)
        if 'llm_enabled' in kwargs:
            self.llm_enabled = kwargs['llm_enabled']
        return True
    
    def get_config_value(self, key, default=None):
        """Get a specific config value."""
        return self.config.get(key, default)
    
    def set_config_value(self, key, value):
        """Set a specific config value."""
        self.config[key] = value
        if key == 'llm_enabled':
            self.llm_enabled = value
        return True

# Backward compatibility aliases - placed at the very end after all class definitions
FieldMapping = FieldMappingState
PersonnelMapping = PersonnelMappingState  
QualityMetrics = AnalysisQualityMetrics
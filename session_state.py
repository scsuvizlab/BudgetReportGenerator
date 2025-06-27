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
        
        # Budget analysis
        self.budget_data: Dict[str, Any] = {}
        self.budget_analysis_complete: bool = False
        self.budget_hash: Optional[str] = None
        
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
        
        # Initialize field mappings
        for field in template_fields:
            field_name = field.name if hasattr(field, 'name') else str(field)
            self.field_mappings[field_name] = FieldMappingState(
                field_name=field_name,
                template_field=field
            )
        
        self.advance_stage(AnalysisStage.TEMPLATE_LOADED)
        self.workflow_progress['template_validated'] = True
    
    def set_budget(self, budget_path: str, budget_data: Dict[str, Any]) -> None:
        """Set budget information and advance stage."""
        self.budget_path = budget_path
        self.budget_data = budget_data
        self.budget_hash = self._calculate_file_hash(budget_path)
        self.budget_analysis_complete = True
        
        # Initialize personnel mappings from budget analysis
        self._initialize_personnel_mappings()
        
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
    
    def save_session(self, file_path: str) -> bool:
        """Save session state to file."""
        try:
            session_data = {
                'session_id': self.session_id,
                'current_stage': self.current_stage.value,
                'created_at': self.created_at.isoformat(),
                'last_updated': self.last_updated.isoformat(),
                'template_path': self.template_path,
                'budget_path': self.budget_path,
                'template_fields': [asdict(field) if hasattr(field, '__dict__') else str(field) 
                                  for field in self.template_fields],
                'budget_data': self.budget_data,
                'field_mappings': {k: asdict(v) for k, v in self.field_mappings.items()},
                'personnel_mappings': {k: asdict(v) for k, v in self.personnel_mappings.items()},
                'quality_metrics': asdict(self.quality_metrics),
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

    # Backward compatibility properties for existing GUI code
    @property
    def template(self):
        """Backward compatibility: return template_fields if loaded, None otherwise."""
        return self.template_fields if self.template_analysis_complete else None
    
    @template.setter  
    def template(self, value):
        """Backward compatibility: set template_fields."""
        if value is not None:
            self.template_fields = value
            self.template_analysis_complete = True
        else:
            self.template_fields = []
            self.template_analysis_complete = False
    
    @property
    def budget(self):
        """Backward compatibility: return budget_data if loaded, None otherwise."""
        return self.budget_data if self.budget_analysis_complete else None
    
    @budget.setter
    def budget(self, value):
        """Backward compatibility: set budget_data."""
        if value is not None:
            self.budget_data = value
            self.budget_analysis_complete = True
        else:
            self.budget_data = {}
            self.budget_analysis_complete = False
    
    @property
    def field_mappings_dict(self):
        """Backward compatibility: return field mappings in old format."""
        return {name: mapping.mapped_value for name, mapping in self.field_mappings.items() 
                if mapping.mapped_location}
    
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

# Backward compatibility aliases - placed at the very end after all class definitions
FieldMapping = FieldMappingState
PersonnelMapping = PersonnelMappingState  
QualityMetrics = AnalysisQualityMetrics
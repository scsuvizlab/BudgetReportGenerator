"""
Enhanced analysis coordinator that orchestrates the complete budget analysis workflow
with improved string analysis, contextual field mapping, and grant-specific processing.
"""
import logging
import time
from typing import Dict, List, Tuple, Optional, Any, Callable
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

from session_state import SessionState, AnalysisStage, MappingConfidence
from field_detector import FieldDetector, TemplateField
from budget_book import BudgetBook
from cell_resolver import CellResolver
from llm_integration_manager import LLMIntegrationManager, LLMAnalysisRequest

@dataclass
class AnalysisConfig:
    """Configuration for the analysis process."""
    enable_llm: bool = True
    llm_model: str = "gpt-4o-mini"
    max_llm_cost: float = 5.0
    enable_string_focus: bool = True
    enable_personnel_extraction: bool = True
    enable_notes_detection: bool = True
    enable_context_analysis: bool = True
    confidence_threshold: float = 0.5
    require_user_verification: bool = False
    save_intermediate_results: bool = True
    debug_mode: bool = False

class AnalysisCoordinator:
    """Enhanced coordinator for comprehensive budget analysis."""
    
    def __init__(self, config: AnalysisConfig, llm_client=None, progress_callback: Callable = None):
        self.config = config
        self.llm_client = llm_client
        self.progress_callback = progress_callback
        self.logger = logging.getLogger(__name__)
        
        # Initialize component managers
        self.field_detector = FieldDetector(enable_debug=config.debug_mode)
        self.budget_book = BudgetBook(enable_debug=config.debug_mode)
        self.cell_resolver = CellResolver(llm_client, enable_debug=config.debug_mode)
        
        if llm_client:
            self.llm_manager = LLMIntegrationManager(
                llm_client, 
                enable_debug=config.debug_mode
            )
        else:
            self.llm_manager = None
        
        # Analysis state
        self.current_session: Optional[SessionState] = None
        self.analysis_start_time: Optional[datetime] = None
        self.analysis_results: Dict[str, Any] = {}
    
    def analyze_budget_comprehensive(self, template_path: str, budget_path: str, 
                                   session_id: Optional[str] = None) -> SessionState:
        """Perform comprehensive budget analysis with enhanced string processing."""
        self.analysis_start_time = datetime.now()
        self.logger.info("Starting comprehensive budget analysis")
        
        # Initialize or load session
        if session_id:
            self.current_session = SessionState.load_session(f"{session_id}.json")
            if not self.current_session:
                self.logger.warning(f"Could not load session {session_id}, creating new session")
                self.current_session = SessionState()
        else:
            self.current_session = SessionState()
        
        try:
            # Step 1: Analyze Template
            self._report_progress("Analyzing template document...", 10)
            template_fields = self._analyze_template(template_path)
            
            # Step 2: Load and Analyze Budget
            self._report_progress("Loading and analyzing budget spreadsheet...", 20)
            budget_data = self._analyze_budget(budget_path)
            
            # Step 3: Enhanced Structure Analysis
            self._report_progress("Performing enhanced structure analysis...", 30)
            self._perform_structure_analysis()
            
            # Step 4: String-Focused Analysis
            if self.config.enable_string_focus:
                self._report_progress("Analyzing text fields and string content...", 40)
                self._perform_string_analysis()
            
            # Step 5: Personnel Extraction
            if self.config.enable_personnel_extraction:
                self._report_progress("Extracting personnel information...", 50)
                self._extract_personnel_information()
            
            # Step 6: Notes and Description Detection
            if self.config.enable_notes_detection:
                self._report_progress("Detecting notes and description columns...", 60)
                self._detect_notes_and_descriptions()
            
            # Step 7: Initial Field Mapping
            self._report_progress("Performing initial field mapping...", 70)
            self._perform_initial_field_mapping()
            
            # Step 8: LLM Enhancement (if enabled)
            if self.config.enable_llm and self.llm_manager:
                self._report_progress("Enhancing analysis with LLM...", 80)
                self._perform_llm_enhancement()
            
            # Step 9: Quality Assessment
            self._report_progress("Assessing mapping quality...", 90)
            self._assess_mapping_quality()
            
            # Step 10: Finalization
            self._report_progress("Finalizing analysis...", 100)
            self._finalize_analysis()
            
            self.current_session.advance_stage(AnalysisStage.COMPLETED)
            
            analysis_duration = datetime.now() - self.analysis_start_time
            self.logger.info(f"Analysis completed in {analysis_duration.total_seconds():.1f} seconds")
            
            return self.current_session
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            if self.current_session:
                self.current_session.add_user_feedback(
                    "error", 
                    f"Analysis failed: {str(e)}", 
                    {"stage": self.current_session.current_stage.value}
                )
            raise
    
    def _analyze_template(self, template_path: str) -> List[TemplateField]:
        """Analyze template document to identify fields."""
        try:
            template_fields = self.field_detector.analyze_template(template_path)
            self.current_session.set_template(template_path, template_fields)
            
            self.logger.info(f"Template analysis complete: {len(template_fields)} fields detected")
            
            # Log field types for debugging
            if self.config.debug_mode:
                field_types = {}
                for field in template_fields:
                    field_types[field.field_type] = field_types.get(field.field_type, 0) + 1
                self.logger.debug(f"Field types detected: {field_types}")
            
            return template_fields
            
        except Exception as e:
            self.logger.error(f"Template analysis failed: {e}")
            raise
    
    def _analyze_budget(self, budget_path: str) -> Dict[str, Any]:
        """Load and analyze budget spreadsheet."""
        try:
            budget_data = self.budget_book.load_budget(budget_path)
            self.current_session.set_budget(budget_path, budget_data)
            
            # Log budget structure
            sheet_count = len(budget_data['sheets'])
            total_personnel = sum(len(sheet_data.get('personnel', [])) 
                                for sheet_data in budget_data['sheets'].values())
            
            self.logger.info(f"Budget analysis complete: {sheet_count} sheets, {total_personnel} personnel entries")
            
            return budget_data
            
        except Exception as e:
            self.logger.error(f"Budget analysis failed: {e}")
            raise
    
    def _perform_structure_analysis(self) -> None:
        """Perform enhanced structure analysis of the budget."""
        try:
            self.current_session.advance_stage(AnalysisStage.STRUCTURE_ANALYZED)
            
            structure_analysis = {}
            
            for sheet_name, sheet_data in self.current_session.budget_data['sheets'].items():
                analysis = sheet_data.get('analysis', {})
                
                # Enhanced structure analysis
                structure_info = {
                    'sections_found': len(analysis.get('sections', {})),
                    'text_cells': len(analysis.get('text_cells', [])),
                    'currency_cells': len(analysis.get('currency_cells', [])),
                    'potential_note_columns': len(analysis.get('potential_note_columns', [])),
                    'personnel_sections': len(analysis.get('personnel_sections', [])),
                    'column_types': analysis.get('column_types', {})
                }
                
                structure_analysis[sheet_name] = structure_info
            
            # Update workflow progress
            self.current_session.workflow_progress['budget_structure_understood'] = True
            
            if self.config.debug_mode:
                self.logger.debug(f"Structure analysis: {structure_analysis}")
                
        except Exception as e:
            self.logger.error(f"Structure analysis failed: {e}")
            raise
    
    def _perform_string_analysis(self) -> None:
        """Perform focused analysis of string content."""
        try:
            string_results = {
                'text_field_locations': {},
                'potential_names': [],
                'potential_titles': [],
                'long_text_fields': [],
                'terminology_matches': {}
            }
            
            for sheet_name, sheet_data in self.current_session.budget_data['sheets'].items():
                cells = sheet_data.get('cells', [])
                
                for cell in cells:
                    if cell.data_type == 'text' and isinstance(cell.value, str):
                        value = cell.value.strip()
                        location = f"{sheet_name}!R{cell.row}C{cell.col}"
                        
                        # Categorize text content
                        if self._looks_like_person_name(value):
                            string_results['potential_names'].append({
                                'location': location,
                                'value': value,
                                'confidence': 0.8
                            })
                        
                        if self._looks_like_job_title(value):
                            string_results['potential_titles'].append({
                                'location': location,
                                'value': value,
                                'confidence': 0.7
                            })
                        
                        if len(value) > 50:  # Long text likely descriptions
                            string_results['long_text_fields'].append({
                                'location': location,
                                'value': value[:100] + "..." if len(value) > 100 else value,
                                'length': len(value)
                            })
                        
                        # Check for grant terminology
                        terminology_matches = self._find_grant_terminology(value)
                        if terminology_matches:
                            string_results['terminology_matches'][location] = terminology_matches
            
            self.current_session.string_analysis_results.update(string_results)
            self.current_session.workflow_progress['string_fields_analyzed'] = True
            
            self.logger.info(f"String analysis complete: {len(string_results['potential_names'])} names, "
                           f"{len(string_results['potential_titles'])} titles, "
                           f"{len(string_results['long_text_fields'])} long text fields")
            
        except Exception as e:
            self.logger.error(f"String analysis failed: {e}")
            raise
    
    def _extract_personnel_information(self) -> None:
        """Enhanced personnel information extraction."""
        try:
            # Personnel information is already extracted in budget_book
            # Here we enhance it with additional analysis
            
            total_personnel = 0
            complete_personnel = 0
            
            for person_id, person_mapping in self.current_session.personnel_mappings.items():
                total_personnel += 1
                
                # Check completeness
                if person_mapping.name and person_mapping.title:
                    complete_personnel += 1
                
                # Try to enhance with string analysis results
                if not person_mapping.title:
                    # Look for titles near the person's name location
                    potential_title = self._find_nearby_title(person_mapping.name_location)
                    if potential_title:
                        person_mapping.title = potential_title
                        person_mapping.extraction_method = "enhanced"
            
            self.current_session.workflow_progress['personnel_identified'] = True
            
            personnel_quality = complete_personnel / total_personnel if total_personnel > 0 else 0
            self.logger.info(f"Personnel extraction complete: {complete_personnel}/{total_personnel} complete entries "
                           f"(quality: {personnel_quality:.1%})")
            
        except Exception as e:
            self.logger.error(f"Personnel extraction failed: {e}")
            raise
    
    def _detect_notes_and_descriptions(self) -> None:
        """Detect notes and description columns/fields."""
        try:
            notes_analysis = {
                'note_columns': [],
                'description_fields': [],
                'rightmost_text_columns': []
            }
            
            for sheet_name, sheet_data in self.current_session.budget_data['sheets'].items():
                analysis = sheet_data.get('analysis', {})
                df = sheet_data['dataframe']
                
                # Find potential note columns
                potential_note_columns = analysis.get('potential_note_columns', [])
                for col_idx, col_name in potential_note_columns:
                    notes_analysis['note_columns'].append({
                        'sheet': sheet_name,
                        'column': col_idx,
                        'header': col_name,
                        'sample_content': self._get_column_sample(df, col_idx)
                    })
                
                # Find rightmost text columns (often contain notes)
                if len(df.columns) > 0:
                    rightmost_cols = list(range(max(0, len(df.columns) - 3), len(df.columns)))
                    for col_idx in rightmost_cols:
                        if col_idx < len(df.columns):
                            col_data = df.iloc[:, col_idx].dropna()
                            text_ratio = sum(isinstance(val, str) for val in col_data) / len(col_data) if len(col_data) > 0 else 0
                            
                            if text_ratio > 0.7:  # Mostly text
                                notes_analysis['rightmost_text_columns'].append({
                                    'sheet': sheet_name,
                                    'column': col_idx,
                                    'text_ratio': text_ratio,
                                    'sample_content': self._get_column_sample(df, col_idx)
                                })
            
            self.current_session.string_analysis_results['notes_analysis'] = notes_analysis
            self.current_session.workflow_progress['notes_columns_found'] = len(notes_analysis['note_columns']) > 0
            
            total_note_sources = (len(notes_analysis['note_columns']) + 
                                len(notes_analysis['rightmost_text_columns']))
            self.logger.info(f"Notes detection complete: {total_note_sources} potential note sources found")
            
        except Exception as e:
            self.logger.error(f"Notes detection failed: {e}")
            raise
    
    def _perform_initial_field_mapping(self) -> None:
        """Perform initial field mapping using rule-based approaches."""
        try:
            self.current_session.advance_stage(AnalysisStage.INITIAL_MAPPING)
            
            mapped_count = 0
            high_confidence_count = 0
            
            for field in self.current_session.template_fields:
                # Use the enhanced cell resolver
                best_matches = []
                
                for sheet_name, sheet_data in self.current_session.budget_data['sheets'].items():
                    df = sheet_data['dataframe']
                    
                    try:
                        matches = self.cell_resolver.resolve_field_mappings(df, [field.name])
                        for match in matches:
                            location = f"{sheet_name}!{match.cell_address}"
                            best_matches.append((location, match.confidence, match.value, match.notes))
                    except Exception as e:
                        self.logger.warning(f"Field mapping failed for {field.name} in {sheet_name}: {e}")
                
                # Sort by confidence and take the best match
                if best_matches:
                    best_matches.sort(key=lambda x: x[1], reverse=True)
                    location, confidence, value, notes = best_matches[0]
                    
                    self.current_session.update_field_mapping(
                        field_name=field.name,
                        location=location,
                        value=value,
                        confidence=confidence,
                        method="rule_based"
                    )
                    
                    # Add alternatives
                    for alt_location, alt_confidence, alt_value, alt_notes in best_matches[1:3]:
                        self.current_session.add_mapping_alternative(
                            field.name, alt_location, alt_confidence
                        )
                    
                    mapped_count += 1
                    if confidence >= 0.8:
                        high_confidence_count += 1
            
            self.logger.info(f"Initial mapping complete: {mapped_count}/{len(self.current_session.template_fields)} fields mapped, "
                           f"{high_confidence_count} high confidence")
            
        except Exception as e:
            self.logger.error(f"Initial field mapping failed: {e}")
            raise
    
    def _perform_llm_enhancement(self) -> None:
        """Enhance analysis using LLM capabilities."""
        if not self.llm_manager:
            self.logger.warning("LLM enhancement requested but LLM manager not available")
            return
        
        try:
            self.current_session.advance_stage(AnalysisStage.LLM_ANALYSIS)
            
            # Perform comprehensive LLM analysis
            llm_results = self.llm_manager.analyze_budget_with_enhanced_llm(
                self.current_session.template_path,
                self.current_session.budget_path
            )
            
            # Update session with LLM results
            self.current_session.set_llm_analysis_results(llm_results)
            
            # Track LLM usage (this would come from the actual LLM client)
            self.current_session.track_llm_usage(tokens_used=1000, cost=0.05)  # Example values
            
            self.logger.info("LLM enhancement complete")
            
        except Exception as e:
            self.logger.error(f"LLM enhancement failed: {e}")
            # Don't raise - continue with rule-based results
    
    def _assess_mapping_quality(self) -> None:
        """Assess the quality of current field mappings."""
        try:
            # Quality metrics are automatically updated in session_state
            # Here we can add additional quality checks
            
            metrics = self.current_session.quality_metrics
            
            # Check for critical issues
            issues = []
            
            if metrics.string_field_coverage < 0.5:
                issues.append("Low string field coverage - many text fields may be unmapped")
            
            if metrics.personnel_extraction_quality < 0.7:
                issues.append("Personnel extraction quality is low")
            
            if metrics.notes_coverage < 0.3:
                issues.append("Poor notes/description coverage")
            
            if metrics.uncertain_mappings > metrics.high_confidence_mappings:
                issues.append("More uncertain mappings than high-confidence mappings")
            
            # Log quality assessment
            if issues:
                self.logger.warning(f"Quality issues detected: {'; '.join(issues)}")
                for issue in issues:
                    self.current_session.add_user_feedback("quality_issue", issue)
            
            self.logger.info(f"Quality assessment: {metrics.overall_quality_score:.1%} overall quality score")
            
        except Exception as e:
            self.logger.error(f"Quality assessment failed: {e}")
            raise
    
    def _finalize_analysis(self) -> None:
        """Finalize analysis and prepare for document generation."""
        try:
            # Save intermediate results if configured
            if self.config.save_intermediate_results:
                session_file = f"session_{self.current_session.session_id}.json"
                self.current_session.save_session(session_file)
                self.logger.info(f"Session saved to {session_file}")
            
            # Generate final analysis report
            analysis_report = self.current_session.get_mapping_report()
            self.analysis_results = analysis_report
            
            # Update workflow completion
            ready_for_generation, issues = self.current_session.is_ready_for_document_generation()
            
            if ready_for_generation:
                self.current_session.workflow_progress['user_review_completed'] = True
                self.logger.info("Analysis complete and ready for document generation")
            else:
                self.logger.warning(f"Analysis complete but issues remain: {'; '.join(issues)}")
                for issue in issues:
                    self.current_session.add_user_feedback("generation_issue", issue)
            
        except Exception as e:
            self.logger.error(f"Analysis finalization failed: {e}")
            raise
    
    def _report_progress(self, message: str, percentage: int) -> None:
        """Report progress to callback if available."""
        self.logger.info(f"Progress {percentage}%: {message}")
        
        if self.progress_callback:
            try:
                self.progress_callback(percentage, message)
            except Exception as e:
                self.logger.warning(f"Progress callback failed: {e}")
    
    def _looks_like_person_name(self, text: str) -> bool:
        """Check if text looks like a person's name."""
        if not isinstance(text, str) or len(text.strip()) < 3:
            return False
        
        text = text.strip()
        words = text.split()
        
        if len(words) < 2 or len(words) > 4:
            return False
        
        # Names typically start with capital letters
        if not all(word[0].isupper() for word in words if word):
            return False
        
        # Names shouldn't contain numbers or special characters
        if any(char.isdigit() for char in text):
            return False
        
        return True
    
    def _looks_like_job_title(self, text: str) -> bool:
        """Check if text looks like a job title."""
        if not isinstance(text, str) or len(text.strip()) < 3:
            return False
        
        text_lower = text.lower().strip()
        
        title_indicators = [
            'investigator', 'professor', 'director', 'manager', 'coordinator',
            'scientist', 'researcher', 'assistant', 'associate', 'postdoc',
            'student', 'technician', 'specialist', 'analyst', 'engineer'
        ]
        
        return any(indicator in text_lower for indicator in title_indicators)
    
    def _find_grant_terminology(self, text: str) -> List[str]:
        """Find grant-specific terminology in text."""
        text_lower = text.lower()
        matches = []
        
        grant_terms = [
            'principal investigator', 'co-investigator', 'key personnel',
            'direct costs', 'indirect costs', 'overhead', 'fringe benefits',
            'equipment', 'supplies', 'travel', 'participant support',
            'person months', 'calendar months', 'academic months',
            'release time', 'course release', 'effort'
        ]
        
        for term in grant_terms:
            if term in text_lower:
                matches.append(term)
        
        return matches
    
    def _find_nearby_title(self, name_location: Optional[str]) -> Optional[str]:
        """Find a job title near a person's name location."""
        if not name_location:
            return None
        
        # Parse location (e.g., "Sheet1!R5C3")
        # This would need proper implementation based on location format
        # For now, return None
        return None
    
    def _get_column_sample(self, df, col_idx: int, max_samples: int = 3) -> List[str]:
        """Get sample content from a column."""
        if col_idx >= len(df.columns):
            return []
        
        col_data = df.iloc[:, col_idx].dropna()
        samples = []
        
        for value in col_data[:max_samples]:
            if isinstance(value, str) and value.strip():
                sample = value.strip()[:100]  # Truncate long values
                if len(value) > 100:
                    sample += "..."
                samples.append(sample)
        
        return samples
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get a summary of the analysis results."""
        if not self.current_session:
            return {"error": "No analysis session available"}
        
        summary = self.current_session.get_session_summary()
        
        # Add analysis-specific information
        if self.analysis_start_time:
            analysis_duration = datetime.now() - self.analysis_start_time
            summary['analysis_duration_seconds'] = analysis_duration.total_seconds()
        
        summary['analysis_results'] = self.analysis_results
        summary['config'] = {
            'llm_enabled': self.config.enable_llm,
            'string_focus': self.config.enable_string_focus,
            'personnel_extraction': self.config.enable_personnel_extraction,
            'notes_detection': self.config.enable_notes_detection
        }
        
        return summary
    
    def export_mapping_results(self, output_path: str) -> bool:
        """Export mapping results to a file for review."""
        if not self.current_session:
            return False
        
        try:
            mapping_report = self.current_session.get_mapping_report()
            
            # Add additional export information
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'session_id': self.current_session.session_id,
                'template_path': self.current_session.template_path,
                'budget_path': self.current_session.budget_path,
                'analysis_config': {
                    'llm_enabled': self.config.enable_llm,
                    'string_focus': self.config.enable_string_focus,
                    'personnel_extraction': self.config.enable_personnel_extraction,
                    'notes_detection': self.config.enable_notes_detection
                },
                'mapping_report': mapping_report
            }
            
            import json
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            self.logger.info(f"Mapping results exported to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Export failed: {e}")
            return False
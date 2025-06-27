"""
String analysis utilities for enhanced budget processing.
Provides specialized functions for analyzing text content in grant budgets,
including name recognition, title detection, and contextual pattern matching.
"""
import re
import logging
from typing import Dict, List, Tuple, Optional, Set, Any, Union
from dataclasses import dataclass
from enum import Enum
import string
from rapidfuzz import fuzz, process

class TextType(Enum):
    """Types of text content that can be identified."""
    PERSON_NAME = "person_name"
    JOB_TITLE = "job_title"
    INSTITUTION = "institution"
    DESCRIPTION = "description"
    NOTE = "note"
    CURRENCY_DESCRIPTION = "currency_description"
    EQUIPMENT_ITEM = "equipment_item"
    TRAVEL_ITEM = "travel_item"
    GRANT_TERMINOLOGY = "grant_terminology"
    GENERIC_TEXT = "generic_text"

@dataclass
class TextAnalysisResult:
    """Result of text analysis for a single piece of text."""
    original_text: str
    text_type: TextType
    confidence: float
    features: Dict[str, Any]
    suggestions: List[str] = None

class GrantTerminologyMatcher:
    """Specialized matcher for grant and research terminology."""
    
    def __init__(self):
        self.personnel_terms = {
            'pi_terms': [
                'principal investigator', 'pi', 'p.i.', 'chief investigator',
                'lead investigator', 'primary investigator'
            ],
            'copi_terms': [
                'co-principal investigator', 'co-pi', 'co-investigator',
                'co investigator', 'coinvestigator', 'co-p.i.'
            ],
            'academic_ranks': [
                'professor', 'assistant professor', 'associate professor',
                'professor emeritus', 'distinguished professor', 'endowed professor',
                'lecturer', 'senior lecturer', 'instructor', 'adjunct professor'
            ],
            'research_positions': [
                'research scientist', 'senior scientist', 'staff scientist',
                'research associate', 'senior research associate',
                'research assistant', 'research specialist', 'research coordinator'
            ],
            'postdoc_terms': [
                'postdoc', 'post-doc', 'postdoctoral', 'postdoctoral fellow',
                'postdoctoral researcher', 'postdoctoral associate'
            ],
            'student_terms': [
                'graduate student', 'grad student', 'graduate research assistant',
                'doctoral student', 'phd student', 'masters student',
                'undergraduate', 'undergraduate student', 'undergraduate researcher'
            ],
            'technical_staff': [
                'technician', 'laboratory technician', 'research technician',
                'technical specialist', 'laboratory manager', 'facility manager'
            ]
        }
        
        self.budget_categories = {
            'personnel_costs': [
                'salary', 'wages', 'stipend', 'compensation', 'personnel costs',
                'labor', 'human resources', 'staff costs'
            ],
            'fringe_benefits': [
                'fringe', 'fringe benefits', 'benefits', 'insurance', 'retirement',
                'fica', 'social security', 'medicare', 'unemployment', 'workers comp'
            ],
            'equipment': [
                'equipment', 'instruments', 'instrumentation', 'apparatus',
                'hardware', 'computers', 'software', 'laboratory equipment'
            ],
            'supplies': [
                'supplies', 'materials', 'consumables', 'reagents', 'chemicals',
                'laboratory supplies', 'office supplies', 'research materials'
            ],
            'travel': [
                'travel', 'transportation', 'airfare', 'flights', 'lodging',
                'accommodation', 'per diem', 'meals', 'conference', 'conference travel'
            ],
            'contractual': [
                'contractual', 'subcontract', 'consultant', 'consulting',
                'professional services', 'external services'
            ],
            'other_costs': [
                'other direct costs', 'odc', 'miscellaneous', 'publication costs',
                'communication', 'telephone', 'internet', 'utilities'
            ]
        }
        
        self.time_effort_terms = [
            'effort', 'percent effort', '% effort', 'time commitment',
            'fte', 'full-time equivalent', 'person months', 'person-months',
            'calendar months', 'academic months', 'summer months',
            'academic year', 'summer support', 'release time', 'course release'
        ]
        
        # Compile regex patterns for better performance
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for efficient matching."""
        self.name_patterns = [
            re.compile(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b'),  # First Last
            re.compile(r'\b[A-Z][a-z]+,\s*[A-Z][a-z]+\b'),  # Last, First
            re.compile(r'\b[A-Z][a-z]+\s+[A-Z]\.\s*[A-Z][a-z]+\b'),  # First M. Last
            re.compile(r'\b[A-Z]\.\s*[A-Z][a-z]+\b'),  # F. Last
            re.compile(r'\bDr\.\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'),  # Dr. Name
            re.compile(r'\bProf\.\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b')  # Prof. Name
        ]
        
        self.currency_patterns = [
            re.compile(r'\$\s*[\d,]+\.?\d*'),
            re.compile(r'[\d,]+\.?\d*\s*\$'),
            re.compile(r'USD\s*[\d,]+\.?\d*'),
            re.compile(r'[\d,]+\.?\d*\s*dollars?', re.IGNORECASE)
        ]
        
        self.percentage_patterns = [
            re.compile(r'[\d.]+\s*%'),
            re.compile(r'[\d.]+\s*percent', re.IGNORECASE),
            re.compile(r'[\d.]+\s*fte', re.IGNORECASE)
        ]

class StringAnalyzer:
    """Main string analysis class with comprehensive text classification."""
    
    def __init__(self, enable_debug: bool = False):
        self.enable_debug = enable_debug
        self.logger = logging.getLogger(__name__)
        self.terminology_matcher = GrantTerminologyMatcher()
        
        # Common patterns for different text types
        self.institution_indicators = [
            'university', 'college', 'institute', 'laboratory', 'center',
            'school', 'department', 'division', 'foundation', 'corporation'
        ]
        
        self.description_indicators = [
            'description', 'purpose', 'objective', 'goal', 'aim', 'rationale',
            'justification', 'explanation', 'details', 'background', 'overview'
        ]
        
        self.note_indicators = [
            'note', 'notes', 'comment', 'comments', 'remark', 'remarks',
            'additional', 'misc', 'miscellaneous', 'other', 'special'
        ]
    
    def analyze_text(self, text: str, context: Dict[str, Any] = None) -> TextAnalysisResult:
        """Comprehensive analysis of a text string."""
        if not isinstance(text, str) or not text.strip():
            return TextAnalysisResult(
                original_text=text,
                text_type=TextType.GENERIC_TEXT,
                confidence=0.0,
                features={}
            )
        
        text = text.strip()
        context = context or {}
        
        # Run all analysis methods
        analysis_results = [
            self._analyze_person_name(text, context),
            self._analyze_job_title(text, context),
            self._analyze_institution(text, context),
            self._analyze_description(text, context),
            self._analyze_note(text, context),
            self._analyze_currency_description(text, context),
            self._analyze_equipment_item(text, context),
            self._analyze_travel_item(text, context),
            self._analyze_grant_terminology(text, context)
        ]
        
        # Filter out None results and sort by confidence
        valid_results = [r for r in analysis_results if r is not None]
        if not valid_results:
            return TextAnalysisResult(
                original_text=text,
                text_type=TextType.GENERIC_TEXT,
                confidence=0.1,
                features={'length': len(text), 'word_count': len(text.split())}
            )
        
        # Return the highest confidence result
        best_result = max(valid_results, key=lambda x: x.confidence)
        
        if self.enable_debug:
            self.logger.debug(f"Text analysis for '{text[:50]}...': {best_result.text_type.value} "
                            f"(confidence: {best_result.confidence:.2f})")
        
        return best_result
    
    def _analyze_person_name(self, text: str, context: Dict[str, Any]) -> Optional[TextAnalysisResult]:
        """Analyze if text is a person's name."""
        features = {}
        confidence = 0.0
        
        # Basic length and character checks
        if len(text) < 2 or len(text) > 100:
            return None
        
        # Check for numbers (names shouldn't have them)
        if any(char.isdigit() for char in text):
            return None
        
        # Check against name patterns
        pattern_matches = 0
        for pattern in self.terminology_matcher.name_patterns:
            if pattern.search(text):
                pattern_matches += 1
                confidence += 0.3
        
        features['pattern_matches'] = pattern_matches
        
        # Word analysis
        words = text.split()
        if len(words) < 2 or len(words) > 4:
            if len(words) == 1 and len(text) > 10:  # Could be a single long name
                confidence -= 0.2
            else:
                return None
        
        # Check capitalization (names should be capitalized)
        proper_capitalization = all(word[0].isupper() and word[1:].islower() 
                                  for word in words if word and len(word) > 1)
        if proper_capitalization:
            confidence += 0.4
            features['proper_capitalization'] = True
        
        # Check for title prefixes/suffixes
        title_prefixes = ['dr', 'prof', 'mr', 'mrs', 'ms']
        title_suffixes = ['phd', 'md', 'jr', 'sr', 'ii', 'iii']
        
        has_title = any(word.lower().rstrip('.') in title_prefixes for word in words) or \
                   any(word.lower().rstrip('.') in title_suffixes for word in words)
        
        if has_title:
            confidence += 0.2
            features['has_title'] = True
        
        # Context clues
        if context:
            nearby_text = context.get('nearby_text', [])
            for nearby in nearby_text:
                if any(term in nearby.lower() for term in ['investigator', 'researcher', 'professor']):
                    confidence += 0.1
                    features['context_support'] = True
                    break
        
        # Avoid common false positives
        common_non_names = ['total', 'amount', 'cost', 'budget', 'year', 'month', 'percent']
        if any(word.lower() in common_non_names for word in words):
            confidence -= 0.3
        
        if confidence > 0.3:
            return TextAnalysisResult(
                original_text=text,
                text_type=TextType.PERSON_NAME,
                confidence=min(confidence, 1.0),
                features=features
            )
        
        return None
    
    def _analyze_job_title(self, text: str, context: Dict[str, Any]) -> Optional[TextAnalysisResult]:
        """Analyze if text is a job title."""
        features = {}
        confidence = 0.0
        text_lower = text.lower()
        
        # Check against all personnel term categories
        for category, terms in self.terminology_matcher.personnel_terms.items():
            for term in terms:
                if term in text_lower:
                    confidence += 0.6
                    features[f'matches_{category}'] = True
                    break
        
        # Additional title indicators
        title_words = [
            'director', 'manager', 'coordinator', 'administrator', 'supervisor',
            'head', 'chief', 'lead', 'senior', 'junior', 'assistant', 'associate'
        ]
        
        for word in title_words:
            if word in text_lower:
                confidence += 0.3
                features['title_indicator'] = word
                break
        
        # Check for academic/research context
        academic_words = ['department', 'laboratory', 'research', 'academic', 'faculty']
        for word in academic_words:
            if word in text_lower:
                confidence += 0.2
                features['academic_context'] = True
                break
        
        # Length and structure checks
        word_count = len(text.split())
        if 2 <= word_count <= 6:  # Reasonable title length
            confidence += 0.1
            features['reasonable_length'] = True
        
        if confidence > 0.4:
            return TextAnalysisResult(
                original_text=text,
                text_type=TextType.JOB_TITLE,
                confidence=min(confidence, 1.0),
                features=features
            )
        
        return None
    
    def _analyze_institution(self, text: str, context: Dict[str, Any]) -> Optional[TextAnalysisResult]:
        """Analyze if text is an institution name."""
        features = {}
        confidence = 0.0
        text_lower = text.lower()
        
        # Check for institution indicators
        for indicator in self.institution_indicators:
            if indicator in text_lower:
                confidence += 0.5
                features['institution_type'] = indicator
                break
        
        # Check for common institution patterns
        if 'university of' in text_lower or text_lower.endswith(' university'):
            confidence += 0.4
            features['university_pattern'] = True
        
        if any(word in text_lower for word in ['institute', 'center', 'laboratory']):
            confidence += 0.3
            features['research_institution'] = True
        
        # Length check (institutions usually have longer names)
        if len(text) > 10 and len(text.split()) >= 2:
            confidence += 0.1
            features['appropriate_length'] = True
        
        if confidence > 0.4:
            return TextAnalysisResult(
                original_text=text,
                text_type=TextType.INSTITUTION,
                confidence=min(confidence, 1.0),
                features=features
            )
        
        return None
    
    def _analyze_description(self, text: str, context: Dict[str, Any]) -> Optional[TextAnalysisResult]:
        """Analyze if text is a description or explanation."""
        features = {}
        confidence = 0.0
        
        # Length is a key indicator for descriptions
        text_length = len(text)
        word_count = len(text.split())
        
        if text_length > 100:  # Long text is likely descriptive
            confidence += 0.6
            features['long_text'] = True
        elif text_length > 50:
            confidence += 0.3
            features['medium_text'] = True
        
        # Check for description indicators
        text_lower = text.lower()
        for indicator in self.description_indicators:
            if indicator in text_lower:
                confidence += 0.4
                features['description_indicator'] = indicator
                break
        
        # Check for explanatory sentence structures
        if any(phrase in text_lower for phrase in ['this is', 'will be', 'used for', 'necessary for']):
            confidence += 0.2
            features['explanatory_structure'] = True
        
        # Sentence structure (descriptions often have complete sentences)
        sentence_count = text.count('.') + text.count('!') + text.count('?')
        if sentence_count > 0:
            confidence += 0.2
            features['has_sentences'] = True
        
        # Word variety (descriptions tend to have diverse vocabulary)
        unique_words = len(set(text.lower().split()))
        if word_count > 0 and unique_words / word_count > 0.7:
            confidence += 0.1
            features['diverse_vocabulary'] = True
        
        if confidence > 0.4:
            return TextAnalysisResult(
                original_text=text,
                text_type=TextType.DESCRIPTION,
                confidence=min(confidence, 1.0),
                features=features
            )
        
        return None
    
    def _analyze_note(self, text: str, context: Dict[str, Any]) -> Optional[TextAnalysisResult]:
        """Analyze if text is a note or comment."""
        features = {}
        confidence = 0.0
        text_lower = text.lower()
        
        # Check for note indicators
        for indicator in self.note_indicators:
            if indicator in text_lower:
                confidence += 0.5
                features['note_indicator'] = indicator
                break
        
        # Context clues (note columns, headers)
        if context:
            column_header = context.get('column_header', '')
            if isinstance(column_header, str):
                header_lower = column_header.lower()
                for indicator in self.note_indicators:
                    if indicator in header_lower:
                        confidence += 0.4
                        features['header_support'] = True
                        break
        
            # Check if this is in a rightmost column (notes often appear there)
            if context.get('is_rightmost_column', False):
                confidence += 0.2
                features['rightmost_column'] = True
        
        # Check for informal language patterns (common in notes)
        informal_patterns = ['see ', 'note:', 'nb:', 'also ', 'additional ', 'extra ']
        for pattern in informal_patterns:
            if pattern in text_lower:
                confidence += 0.3
                features['informal_language'] = True
                break
        
        # Length considerations (notes can be short or long)
        if 10 <= len(text) <= 200:  # Typical note length
            confidence += 0.1
            features['typical_note_length'] = True
        
        if confidence > 0.4:
            return TextAnalysisResult(
                original_text=text,
                text_type=TextType.NOTE,
                confidence=min(confidence, 1.0),
                features=features
            )
        
        return None
    
    def _analyze_currency_description(self, text: str, context: Dict[str, Any]) -> Optional[TextAnalysisResult]:
        """Analyze if text describes a currency amount or financial item."""
        features = {}
        confidence = 0.0
        text_lower = text.lower()
        
        # Check for currency patterns
        currency_matches = 0
        for pattern in self.terminology_matcher.currency_patterns:
            if pattern.search(text):
                currency_matches += 1
                confidence += 0.4
        
        features['currency_matches'] = currency_matches
        
        # Check for budget-related terms
        budget_terms = ['cost', 'price', 'amount', 'budget', 'expense', 'fee', 'payment']
        for term in budget_terms:
            if term in text_lower:
                confidence += 0.3
                features['budget_term'] = term
                break
        
        # Check for percentage patterns (effort, overhead rates)
        percentage_matches = 0
        for pattern in self.terminology_matcher.percentage_patterns:
            if pattern.search(text):
                percentage_matches += 1
                confidence += 0.2
        
        features['percentage_matches'] = percentage_matches
        
        if confidence > 0.4:
            return TextAnalysisResult(
                original_text=text,
                text_type=TextType.CURRENCY_DESCRIPTION,
                confidence=min(confidence, 1.0),
                features=features
            )
        
        return None
    
    def _analyze_equipment_item(self, text: str, context: Dict[str, Any]) -> Optional[TextAnalysisResult]:
        """Analyze if text describes equipment or instruments."""
        features = {}
        confidence = 0.0
        text_lower = text.lower()
        
        # Check against equipment terms
        equipment_terms = self.terminology_matcher.budget_categories['equipment']
        for term in equipment_terms:
            if term in text_lower:
                confidence += 0.5
                features['equipment_term'] = term
                break
        
        # Check for specific equipment types
        specific_equipment = [
            'computer', 'laptop', 'server', 'printer', 'scanner',
            'microscope', 'spectrometer', 'centrifuge', 'incubator',
            'analyzer', 'detector', 'sensor', 'camera', 'monitor'
        ]
        
        for item in specific_equipment:
            if item in text_lower:
                confidence += 0.4
                features['specific_equipment'] = item
                break
        
        # Check for technical specifications (often found in equipment descriptions)
        if any(unit in text_lower for unit in ['mhz', 'ghz', 'gb', 'tb', 'inch', 'cm', 'mm']):
            confidence += 0.2
            features['technical_specs'] = True
        
        if confidence > 0.4:
            return TextAnalysisResult(
                original_text=text,
                text_type=TextType.EQUIPMENT_ITEM,
                confidence=min(confidence, 1.0),
                features=features
            )
        
        return None
    
    def _analyze_travel_item(self, text: str, context: Dict[str, Any]) -> Optional[TextAnalysisResult]:
        """Analyze if text describes travel-related expenses."""
        features = {}
        confidence = 0.0
        text_lower = text.lower()
        
        # Check against travel terms
        travel_terms = self.terminology_matcher.budget_categories['travel']
        for term in travel_terms:
            if term in text_lower:
                confidence += 0.5
                features['travel_term'] = term
                break
        
        # Check for specific travel items
        travel_items = [
            'flight', 'hotel', 'rental car', 'taxi', 'uber', 'registration',
            'conference fee', 'workshop', 'meeting', 'symposium'
        ]
        
        for item in travel_items:
            if item in text_lower:
                confidence += 0.4
                features['travel_item'] = item
                break
        
        # Check for location names (cities, states, countries)
        # This is a simplified check - a full implementation would use a location database
        if any(word.istitle() and len(word) > 3 for word in text.split()):
            confidence += 0.1
            features['potential_location'] = True
        
        if confidence > 0.4:
            return TextAnalysisResult(
                original_text=text,
                text_type=TextType.TRAVEL_ITEM,
                confidence=min(confidence, 1.0),
                features=features
            )
        
        return None
    
    def _analyze_grant_terminology(self, text: str, context: Dict[str, Any]) -> Optional[TextAnalysisResult]:
        """Analyze if text contains specific grant/research terminology."""
        features = {}
        confidence = 0.0
        text_lower = text.lower()
        
        # Check time/effort terms
        for term in self.terminology_matcher.time_effort_terms:
            if term in text_lower:
                confidence += 0.6
                features['effort_term'] = term
                break
        
        # Check for grant-specific phrases
        grant_phrases = [
            'indirect cost', 'facilities and administrative', 'f&a',
            'cost share', 'matching funds', 'base salary', 'academic year salary'
        ]
        
        for phrase in grant_phrases:
            if phrase in text_lower:
                confidence += 0.5
                features['grant_phrase'] = phrase
                break
        
        if confidence > 0.5:
            return TextAnalysisResult(
                original_text=text,
                text_type=TextType.GRANT_TERMINOLOGY,
                confidence=min(confidence, 1.0),
                features=features
            )
        
        return None
    
    def batch_analyze_texts(self, texts: List[str], contexts: List[Dict[str, Any]] = None) -> List[TextAnalysisResult]:
        """Analyze multiple texts efficiently."""
        if contexts is None:
            contexts = [{}] * len(texts)
        
        results = []
        for text, context in zip(texts, contexts):
            result = self.analyze_text(text, context)
            results.append(result)
        
        return results
    
    def find_personnel_in_texts(self, texts: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
        """Find personnel information in a list of (location, text) tuples."""
        personnel_candidates = []
        
        for location, text in texts:
            result = self.analyze_text(text)
            
            if result.text_type == TextType.PERSON_NAME and result.confidence > 0.5:
                personnel_candidates.append({
                    'location': location,
                    'name': text,
                    'confidence': result.confidence,
                    'features': result.features
                })
        
        return personnel_candidates
    
    def find_notes_in_texts(self, texts: List[Tuple[str, str]], contexts: List[Dict] = None) -> List[Dict[str, Any]]:
        """Find notes and descriptions in a list of (location, text) tuples."""
        if contexts is None:
            contexts = [{}] * len(texts)
        
        notes_candidates = []
        
        for (location, text), context in zip(texts, contexts):
            result = self.analyze_text(text, context)
            
            if result.text_type in [TextType.NOTE, TextType.DESCRIPTION] and result.confidence > 0.4:
                notes_candidates.append({
                    'location': location,
                    'text': text,
                    'type': result.text_type.value,
                    'confidence': result.confidence,
                    'features': result.features
                })
        
        return notes_candidates
    
    def suggest_field_mappings(self, template_field: str, candidate_texts: List[Tuple[str, str]]) -> List[Tuple[str, float, str]]:
        """Suggest mappings for a template field based on text analysis."""
        suggestions = []
        field_lower = template_field.lower()
        
        # Determine what type of content this field is looking for
        field_type = self._classify_template_field(template_field)
        
        for location, text in candidate_texts:
            result = self.analyze_text(text)
            
            # Calculate match score based on field type and text analysis
            match_score = self._calculate_field_match_score(field_type, field_lower, result)
            
            if match_score > 0.3:
                reasoning = self._generate_match_reasoning(field_type, result, match_score)
                suggestions.append((location, match_score, reasoning))
        
        # Sort by match score
        suggestions.sort(key=lambda x: x[1], reverse=True)
        return suggestions[:10]  # Return top 10 suggestions
    
    def _classify_template_field(self, field_name: str) -> TextType:
        """Classify what type of content a template field is looking for."""
        field_lower = field_name.lower()
        
        if any(term in field_lower for term in ['name', 'investigator', 'pi', 'researcher']):
            return TextType.PERSON_NAME
        elif any(term in field_lower for term in ['title', 'position', 'role', 'rank']):
            return TextType.JOB_TITLE
        elif any(term in field_lower for term in ['institution', 'university', 'organization']):
            return TextType.INSTITUTION
        elif any(term in field_lower for term in ['description', 'explain', 'detail', 'summary']):
            return TextType.DESCRIPTION
        elif any(term in field_lower for term in ['note', 'comment', 'remark']):
            return TextType.NOTE
        elif any(term in field_lower for term in ['cost', 'amount', 'budget', 'salary', 'price']):
            return TextType.CURRENCY_DESCRIPTION
        elif any(term in field_lower for term in ['equipment', 'instrument', 'computer', 'software']):
            return TextType.EQUIPMENT_ITEM
        elif any(term in field_lower for term in ['travel', 'conference', 'trip']):
            return TextType.TRAVEL_ITEM
        else:
            return TextType.GENERIC_TEXT
    
    def _calculate_field_match_score(self, field_type: TextType, field_name: str, 
                                   text_result: TextAnalysisResult) -> float:
        """Calculate how well a text analysis result matches a field type."""
        base_score = 0.0
        
        # Direct type match
        if text_result.text_type == field_type:
            base_score = text_result.confidence * 0.8
        
        # Partial matches for related types
        elif field_type == TextType.PERSON_NAME and text_result.text_type == TextType.JOB_TITLE:
            base_score = text_result.confidence * 0.3  # Titles sometimes contain names
        elif field_type == TextType.DESCRIPTION and text_result.text_type == TextType.NOTE:
            base_score = text_result.confidence * 0.6  # Notes are often descriptions
        elif field_type == TextType.CURRENCY_DESCRIPTION and text_result.text_type == TextType.EQUIPMENT_ITEM:
            base_score = text_result.confidence * 0.4  # Equipment items often have costs
        
        # Add fuzzy string matching for field name
        field_name_words = set(field_name.lower().split())
        text_words = set(text_result.original_text.lower().split())
        
        if field_name_words.intersection(text_words):
            base_score += 0.2
        
        # Fuzzy similarity bonus
        similarity = fuzz.partial_ratio(field_name, text_result.original_text.lower()) / 100.0
        if similarity > 0.5:
            base_score += similarity * 0.2
        
        return min(base_score, 1.0)
    
    def _generate_match_reasoning(self, field_type: TextType, text_result: TextAnalysisResult, 
                                match_score: float) -> str:
        """Generate human-readable reasoning for a field match."""
        reasons = []
        
        if text_result.text_type == field_type:
            reasons.append(f"Text type matches field requirement ({field_type.value})")
        
        if text_result.confidence > 0.8:
            reasons.append("High confidence in text classification")
        elif text_result.confidence > 0.5:
            reasons.append("Medium confidence in text classification")
        
        # Add specific feature explanations
        if 'pattern_matches' in text_result.features:
            reasons.append("Matches expected text patterns")
        
        if 'context_support' in text_result.features:
            reasons.append("Supported by surrounding context")
        
        if not reasons:
            reasons.append(f"Partial match with {match_score:.1%} confidence")
        
        return "; ".join(reasons)

# Utility functions for external use
def quick_analyze_text_type(text: str) -> TextType:
    """Quick analysis to determine text type without full analysis."""
    analyzer = StringAnalyzer()
    result = analyzer.analyze_text(text)
    return result.text_type

def extract_names_from_texts(texts: List[str]) -> List[str]:
    """Extract likely person names from a list of texts."""
    analyzer = StringAnalyzer()
    names = []
    
    for text in texts:
        result = analyzer.analyze_text(text)
        if result.text_type == TextType.PERSON_NAME and result.confidence > 0.5:
            names.append(text)
    
    return names

def find_description_fields(texts: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Find texts that are likely descriptions or notes."""
    analyzer = StringAnalyzer()
    descriptions = []
    
    for location, text in texts:
        result = analyzer.analyze_text(text)
        if result.text_type in [TextType.DESCRIPTION, TextType.NOTE] and result.confidence > 0.4:
            descriptions.append((location, text))
    
    return descriptions
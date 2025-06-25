"""
API Key Manager - FIXED VERSION

Fixed to allow underscores in API keys (required for OpenAI project-based keys).
"""
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import keyring
from PyQt6.QtCore import QSettings

logger = logging.getLogger(__name__)


class APIKeyManager:
    """
    Manages OpenAI API keys with multiple storage options.
    
    Responsibilities:
    - Secure key storage using system keyring
    - Key validation
    - JSON import/export
    - Settings persistence
    """
    
    SERVICE_NAME = "BudgetJustificationTool"
    KEY_USERNAME = "openai_api_key"
    
    def __init__(self):
        """Initialize API key manager."""
        self.settings = QSettings("BudgetTool", "BudgetJustification")
        self._cached_key: Optional[str] = None
        self._key_validated = False
    
    def set_api_key(self, api_key: str, save_to_keyring: bool = True) -> bool:
        """
        Set and optionally store the API key.
        
        Args:
            api_key: The OpenAI API key
            save_to_keyring: Whether to save to system keyring
            
        Returns:
            True if key was set successfully
        """
        if not api_key or not api_key.strip():
            logger.error("Empty API key provided")
            return False
        
        # Clean the key
        cleaned_key = api_key.strip()
        
        # Basic format validation
        if not self._is_valid_key_format(cleaned_key):
            logger.error("API key format appears invalid")
            return False
        
        # Store in cache
        self._cached_key = cleaned_key
        self._key_validated = False
        
        # Store in keyring if requested
        if save_to_keyring:
            try:
                keyring.set_password(self.SERVICE_NAME, self.KEY_USERNAME, cleaned_key)
                logger.info("API key stored in system keyring")
            except Exception as e:
                logger.warning(f"Failed to store key in keyring: {e}")
                # Continue anyway - we still have the cached key
        
        # Store preference in settings
        self.settings.setValue("api_key_stored_in_keyring", save_to_keyring)
        
        return True
    
    def get_api_key(self) -> Optional[str]:
        """
        Retrieve the API key from cache or keyring.
        
        Returns:
            API key if available, None otherwise
        """
        # Return cached key if available
        if self._cached_key:
            return self._cached_key
        
        # Try to load from keyring
        try:
            stored_key = keyring.get_password(self.SERVICE_NAME, self.KEY_USERNAME)
            if stored_key:
                self._cached_key = stored_key
                logger.info("API key loaded from system keyring")
                return stored_key
        except Exception as e:
            logger.warning(f"Failed to load key from keyring: {e}")
        
        return None
    
    def validate_current_key(self) -> bool:
        """
        Validate the current API key with a test call.
        
        Returns:
            True if key is valid
        """
        api_key = self.get_api_key()
        if not api_key:
            return False
        
        # Use the LLM client for validation
        try:
            from llm_client import LLMClient
            
            client = LLMClient(api_key, "gpt-4o-mini")
            is_valid = client.validate_api_key()
            
            self._key_validated = is_valid
            
            if is_valid:
                logger.info("API key validation successful")
            else:
                logger.warning("API key validation failed")
            
            return is_valid
            
        except Exception as e:
            logger.error(f"API key validation error: {e}")
            return False
    
    def clear_api_key(self) -> None:
        """Clear the API key from all storage locations."""
        # Clear cache
        self._cached_key = None
        self._key_validated = False
        
        # Clear from keyring
        try:
            keyring.delete_password(self.SERVICE_NAME, self.KEY_USERNAME)
            logger.info("API key cleared from system keyring")
        except Exception as e:
            logger.warning(f"Failed to clear key from keyring: {e}")
        
        # Clear settings
        self.settings.remove("api_key_stored_in_keyring")
    
    def import_from_json(self, json_path: Path) -> bool:
        """
        Import API key and settings from JSON file.
        
        Args:
            json_path: Path to JSON configuration file
            
        Returns:
            True if import was successful
        """
        try:
            with open(json_path, 'r') as f:
                config = json.load(f)
            
            # Extract API key
            api_key = config.get('openai_api_key', '').strip()
            if not api_key:
                logger.error("No API key found in JSON file")
                return False
            
            # Set the API key
            success = self.set_api_key(api_key, save_to_keyring=True)
            
            if success:
                # Import other settings if available
                if 'default_model' in config:
                    self.settings.setValue("default_model", config['default_model'])
                
                if 'cost_limit_usd' in config:
                    self.settings.setValue("cost_limit", float(config['cost_limit_usd']))
                
                logger.info(f"Configuration imported from {json_path}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to import from JSON: {e}")
            return False
    
    def export_to_json(self, json_path: Path, include_api_key: bool = False) -> bool:
        """
        Export configuration to JSON file.
        
        Args:
            json_path: Path where to save the JSON file
            include_api_key: Whether to include the API key in export
            
        Returns:
            True if export was successful
        """
        try:
            config = {
                'default_model': self.settings.value("default_model", "gpt-4o-mini"),
                'cost_limit_usd': self.settings.value("cost_limit", 5.0, type=float),
                'export_timestamp': json.dumps({"timestamp": "generated"})  # Add timestamp
            }
            
            # Include API key if requested
            if include_api_key:
                api_key = self.get_api_key()
                if api_key:
                    config['openai_api_key'] = api_key
                else:
                    logger.warning("No API key available for export")
            else:
                config['openai_api_key'] = "[REDACTED - Import your key separately]"
            
            with open(json_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Configuration exported to {json_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export to JSON: {e}")
            return False
    
    def _is_valid_key_format(self, api_key: str) -> bool:
        """
        Basic validation of API key format.
        
        Args:
            api_key: The API key to validate
            
        Returns:
            True if format appears valid
        """
        # OpenAI keys typically start with "sk-" and are 51+ characters
        if not api_key.startswith("sk-"):
            return False
        
        if len(api_key) < 40:  # Reasonable minimum length
            return False
        
        # FIXED: Should contain alphanumeric characters, hyphens, AND UNDERSCORES
        # OpenAI project-based keys (sk-proj-*) contain underscores
        allowed_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
        if not set(api_key).issubset(allowed_chars):
            return False
        
        return True
    
    def get_key_status(self) -> Dict[str, Any]:
        """
        Get status information about the API key.
        
        Returns:
            Dictionary with key status information
        """
        api_key = self.get_api_key()
        
        return {
            'has_key': api_key is not None,
            'key_length': len(api_key) if api_key else 0,
            'key_format_valid': self._is_valid_key_format(api_key) if api_key else False,
            'key_validated': self._key_validated,
            'stored_in_keyring': self.settings.value("api_key_stored_in_keyring", False, type=bool),
            'masked_key': self._mask_key(api_key) if api_key else None
        }
    
    def _mask_key(self, api_key: str) -> str:
        """
        Create a masked version of the API key for display.
        
        Args:
            api_key: The API key to mask
            
        Returns:
            Masked key string
        """
        if len(api_key) <= 10:
            return "*" * len(api_key)
        
        # Show first 6 and last 4 characters
        return api_key[:6] + "*" * (len(api_key) - 10) + api_key[-4:]
    
    def has_valid_key(self) -> bool:
        """
        Check if we have a valid API key available.
        
        Returns:
            True if we have a validated API key
        """
        return self.get_api_key() is not None and self._key_validated


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    manager = APIKeyManager()
    
    print("API Key Manager Example:")
    
    # Check current status
    status = manager.get_key_status()
    print(f"Current status: {status}")
    
    # Example of setting a test key (don't use a real key here)
    test_key = "sk-proj-test1234567890abcdefghijklmnopqrstuvwxyz_1234567890"
    if manager.set_api_key(test_key, save_to_keyring=False):
        print(f"Test key set successfully")
        print(f"Masked key: {manager._mask_key(test_key)}")
    
    # Example JSON structure
    print("\nExample JSON configuration file structure:")
    example_config = {
        "openai_api_key": "sk-proj-your-actual-key-here",
        "default_model": "gpt-4o-mini",
        "cost_limit_usd": 5.0
    }
    print(json.dumps(example_config, indent=2))
    
    # Clean up test
    manager.clear_api_key()
    print("\nTest key cleared")
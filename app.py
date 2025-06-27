#!/usr/bin/env python3
"""
Budget Justification Automation Tool - Complete Application Launcher with LLM Integration

This is the main entry point for the budget justification automation tool
with full LLM integration capabilities.
"""
import sys
import os
import logging
from pathlib import Path
import traceback

from PyQt6.QtGui import QIcon

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Set up module-level logger
logger = logging.getLogger(__name__)


def check_dependencies():
    """Check if all required dependencies are available."""
    missing_deps = []
    optional_missing = []
    
    # Core dependencies
    try:
        from PyQt6.QtCore import qVersion
        print(f"✓ PyQt6 {qVersion()}")
    except ImportError:
        missing_deps.append("PyQt6")
    
    try:
        import docx
        print(f"✓ python-docx")
    except ImportError:
        missing_deps.append("python-docx")
    
    try:
        import openpyxl
        version = getattr(openpyxl, '__version__', 'unknown')
        print(f"✓ openpyxl {version}")
    except ImportError:
        missing_deps.append("openpyxl")
    
    try:
        import pandas as pd
        print(f"✓ pandas {pd.__version__}")
    except ImportError:
        missing_deps.append("pandas")
    
    try:
        import rapidfuzz
        version = getattr(rapidfuzz, '__version__', 'unknown')
        print(f"✓ rapidfuzz {version}")
    except ImportError:
        missing_deps.append("rapidfuzz")
    
    # LLM dependencies
    try:
        import openai
        print(f"✓ openai {openai.__version__}")
    except ImportError:
        missing_deps.append("openai")
    
    try:
        import tiktoken
        version = getattr(tiktoken, '__version__', 'unknown')
        print(f"✓ tiktoken {version}")
    except ImportError:
        missing_deps.append("tiktoken")
    
    try:
        import keyring
        version = getattr(keyring, '__version__', 'unknown')
        print(f"✓ keyring {version}")
    except ImportError:
        missing_deps.append("keyring")
    
    try:
        import markdown
        version = getattr(markdown, '__version__', 'unknown')
        print(f"✓ markdown {version}")
    except ImportError:
        missing_deps.append("markdown")
    
    # Optional dependencies
    try:
        import PyPDF2
        version = getattr(PyPDF2, '__version__', 'unknown')
        print(f"✓ PyPDF2 {version} (PDF support)")
    except ImportError:
        optional_missing.append("PyPDF2")
    
    try:
        import pdfplumber
        version = getattr(pdfplumber, '__version__', 'unknown')
        print(f"✓ pdfplumber {version} (PDF support)")
    except ImportError:
        optional_missing.append("pdfplumber")
    
    try:
        from PyQt6.QtWebEngineWidgets import QWebEngineView
        print(f"✓ PyQt6-WebEngine (HTML preview)")
    except ImportError:
        optional_missing.append("PyQt6-WebEngine")
    
    if missing_deps:
        print("\n❌ Missing required dependencies:")
        for dep in missing_deps:
            print(f"  - {dep}")
        print("\nInstall missing dependencies with:")
        print("pip install " + " ".join(missing_deps))
        return False
    
    if optional_missing:
        print("\n⚠️  Missing optional dependencies:")
        for dep in optional_missing:
            print(f"  - {dep}")
        print("\nThese are optional but recommended for full functionality.")
        print("Install with: pip install " + " ".join(optional_missing))
    
    return True


def setup_logging():
    """Set up application logging."""
    # Create logs directory
    log_dir = Path.home() / ".budget_tool" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging configuration
    log_file = log_dir / "app.log"
    
    # Configure logging with both file and console output
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("Budget Justification Tool with LLM Integration starting...")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Log file: {log_file}")
    
    return logger


def check_environment():
    """Check environment and system requirements."""
    print("Environment Check:")
    print(f"Python version: {sys.version}")
    print(f"Platform: {sys.platform}")
    print(f"Working directory: {os.getcwd()}")
    
    # Check for environment variables
    if "OPENAI_API_KEY" in os.environ:
        print("✓ OPENAI_API_KEY environment variable found")
    else:
        print("ℹ  No OPENAI_API_KEY environment variable (will use GUI configuration)")
    
    # Check write permissions
    try:
        test_dir = Path.home() / ".budget_tool"
        test_dir.mkdir(exist_ok=True)
        test_file = test_dir / "test_write.tmp"
        test_file.write_text("test")
        test_file.unlink()
        print("✓ Write permissions OK")
    except Exception as e:
        print(f"⚠️  Write permission issue: {e}")
    
    print()


def check_required_files():
    """Check if all required Python modules are present."""
    required_files = [
        "session_state.py",
        "template_document.py", 
        "budget_book.py",
        "llm_client.py",
        "llm_integration_manager.py",
        "field_detector.py",
        "cell_resolver.py",
        "cost_guard.py",
        "api_key_manager.py",
        "llm_config_dialog.py",
        "document_generator.py",
        "main_window.py"
    ]
    
    missing_files = []
    
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
        else:
            print(f"✓ {file}")
    
    if missing_files:
        print(f"\n❌ Missing required files:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nPlease ensure all Python modules are in the same directory.")
        return False
    
    return True


def show_startup_info():
    """Show application startup information."""
    print("=" * 60)
    print("Budget Justification Automation Tool v1.0")
    print("With OpenAI LLM Integration")
    print("=" * 60)
    print()
    print("Features:")
    print("• Multi-format template support (DOCX, MD, TXT, PDF)")
    print("• Excel/CSV budget file processing")
    print("• AI-powered field mapping using OpenAI GPT models")
    print("• Manual override capabilities")
    print("• Real-time cost tracking and budget controls")
    print("• Secure API key storage")
    print("• Document generation in Word and Markdown formats")
    print()


def main():
    """Main application entry point."""
    show_startup_info()
    
    print("Checking environment...")
    check_environment()
    
    print("Checking dependencies...")
    if not check_dependencies():
        print("\n❌ Cannot start due to missing dependencies.")
        return 1
    
    print("\nChecking required files...")
    if not check_required_files():
        print("\n❌ Cannot start due to missing files.")
        return 1
    
    print("\nSetting up logging...")
    logger = setup_logging()
    
    try:
        print("\nStarting GUI application...")
        
        # Import the main window (after dependency checks)
        from main_window import MainWindow, QApplication
        
        # Create QApplication
        app = QApplication(sys.argv)
        app.setApplicationName("Budget Justification Tool")
        app.setApplicationVersion("1.0")
        app.setOrganizationName("BudgetTool")
        app.setOrganizationDomain("budgettool.local")
        
        # Set application icon if available
        icon_path = Path("icon.png")
        if icon_path.exists():
            app.setWindowIcon(QIcon(str(icon_path)))
        
        # Create and show main window
        logger.info("Creating main window...")
        window = MainWindow()
        
        # Center window on screen
        screen = app.primaryScreen().geometry()
        window.move(
            (screen.width() - window.width()) // 2,
            (screen.height() - window.height()) // 2
        )
        
        window.show()
        
        logger.info("Application started successfully")
        print("✓ Application started successfully!")
        print("\nGUI is now running. Check the application window.")
        print("To enable LLM features, go to LLM → Configure OpenAI API...")
        
        # Run the application
        return app.exec()
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        print(f"\n❌ Import error: {e}")
        print("Make sure all Python files are in the same directory.")
        return 1
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())
        print(f"\n❌ Unexpected error: {e}")
        print("Check the log file for details.")
        return 1


def run_tests():
    """Run basic tests to verify installation."""
    print("Running basic tests...")
    
    # Set up logger for testing
    logger = logging.getLogger(__name__)
    
    try:
        # Test session state
        from session_state import SessionState
        session = SessionState()
        print("✓ SessionState creation: OK")
        
        # Test template parsing
        from template_document import TemplateParser
        parser = TemplateParser()
        print("✓ TemplateParser creation: OK")
        
        # Test budget parsing
        from budget_book import BudgetParser
        budget_parser = BudgetParser()
        print("✓ BudgetParser creation: OK")
        
        # Test LLM integration (without API key)
        from llm_integration_manager import LLMIntegrationManager
        llm_manager = LLMIntegrationManager()
        print("✓ LLMIntegrationManager creation: OK")
        
        # Test document generation
        from document_generator import DocumentGenerator
        doc_gen = DocumentGenerator()
        print("✓ DocumentGenerator creation: OK")
        
        print("\n✓ All core components working correctly!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        logger.error(f"Test failed: {e}")
        return False


if __name__ == "__main__":
    # Check for test mode
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        print("Test Mode")
        print("=" * 40)
        setup_logging()
        
        if check_dependencies() and check_required_files():
            if run_tests():
                print("\n✅ All tests passed! Ready to run the application.")
                sys.exit(0)
            else:
                print("\n❌ Tests failed.")
                sys.exit(1)
        else:
            print("\n❌ Dependency or file checks failed.")
            sys.exit(1)
    
    # Normal application startup
    exit_code = main()
    
    # Clean shutdown message
    if exit_code == 0:
        print("\nApplication closed normally.")
    else:
        print(f"\nApplication exited with code {exit_code}")
    
    sys.exit(exit_code)
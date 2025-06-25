"""
LLM Client

Core OpenAI API client for budget justification tool.
Handles API calls, response parsing, and usage tracking.
"""
import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from openai import OpenAI
import tiktoken

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Structured response from LLM API calls."""
    content: str
    usage: Dict[str, int]
    cost_usd: float
    model: str
    latency_ms: int
    success: bool = True
    error_message: str = ""


class LLMClient:
    """
    OpenAI API client with cost tracking and error handling.
    
    Responsibilities:
    - Make API calls to OpenAI
    - Track token usage and costs
    - Handle errors and retries
    - Estimate costs before calls
    """
    
    # Token pricing (as of June 2025 - update as needed)
    PRICING = {
        'gpt-4o-mini': {
            'input': 0.000150 / 1000,   # $0.15 per 1M tokens
            'output': 0.000600 / 1000   # $0.60 per 1M tokens
        },
        'gpt-4o': {
            'input': 0.005 / 1000,      # $5.00 per 1M tokens  
            'output': 0.015 / 1000      # $15.00 per 1M tokens
        }
    }
    
    def __init__(self, api_key: str, default_model: str = "gpt-4o-mini"):
        """
        Initialize LLM client.
        
        Args:
            api_key: OpenAI API key
            default_model: Default model to use for calls
        """
        if not api_key or not api_key.strip():
            raise ValueError("API key is required")
        
        self.client = OpenAI(api_key=api_key.strip())
        self.default_model = default_model
        self.total_tokens_used = 0
        self.total_cost_usd = 0.0
        self.call_count = 0
        
        # Initialize tokenizer for cost estimation
        try:
            self.tokenizer = tiktoken.encoding_for_model(default_model)
        except KeyError:
            logger.warning(f"Unknown model {default_model}, using fallback tokenizer")
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def estimate_cost(self, prompt: str, max_tokens: int = 500, model: str = None) -> float:
        """
        Estimate the cost of an LLM call before making it.
        
        Args:
            prompt: The input prompt text
            max_tokens: Maximum tokens to generate
            model: Model to use (defaults to client's default)
            
        Returns:
            Estimated cost in USD
        """
        model = model or self.default_model
        
        if model not in self.PRICING:
            logger.warning(f"Unknown model {model}, using gpt-4o-mini pricing")
            model = "gpt-4o-mini"
        
        input_tokens = len(self.tokenizer.encode(prompt))
        input_cost = input_tokens * self.PRICING[model]['input']
        output_cost = max_tokens * self.PRICING[model]['output']
        
        return input_cost + output_cost
    
    def call(self, 
             messages: List[Dict[str, str]], 
             model: str = None,
             max_tokens: int = 500, 
             temperature: float = 0.1,
             retries: int = 3) -> LLMResponse:
        """
        Make an LLM API call with error handling and cost tracking.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model to use
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            retries: Number of retry attempts on failure
            
        Returns:
            LLMResponse with content and metadata
        """
        model = model or self.default_model
        start_time = time.time()
        
        for attempt in range(retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
                latency_ms = int((time.time() - start_time) * 1000)
                
                # Calculate actual cost
                usage = response.usage
                input_cost = usage.prompt_tokens * self.PRICING.get(model, self.PRICING['gpt-4o-mini'])['input']
                output_cost = usage.completion_tokens * self.PRICING.get(model, self.PRICING['gpt-4o-mini'])['output']
                total_cost = input_cost + output_cost
                
                # Update tracking
                self.total_tokens_used += usage.total_tokens
                self.total_cost_usd += total_cost
                self.call_count += 1
                
                logger.info(f"LLM call successful: {model}, {usage.total_tokens} tokens, ${total_cost:.4f}, {latency_ms}ms")
                
                return LLMResponse(
                    content=response.choices[0].message.content,
                    usage={
                        'prompt_tokens': usage.prompt_tokens,
                        'completion_tokens': usage.completion_tokens,
                        'total_tokens': usage.total_tokens
                    },
                    cost_usd=total_cost,
                    model=model,
                    latency_ms=latency_ms,
                    success=True
                )
                
            except Exception as e:
                logger.warning(f"LLM call attempt {attempt + 1} failed: {e}")
                
                if attempt == retries:
                    # Final attempt failed
                    logger.error(f"LLM call failed after {retries + 1} attempts: {e}")
                    return LLMResponse(
                        content="",
                        usage={'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0},
                        cost_usd=0.0,
                        model=model,
                        latency_ms=int((time.time() - start_time) * 1000),
                        success=False,
                        error_message=str(e)
                    )
                else:
                    # Wait before retry
                    time.sleep(2 ** attempt)  # Exponential backoff
    
    def validate_api_key(self) -> bool:
        """
        Validate the API key with a minimal test call.
        
        Returns:
            True if API key is valid, False otherwise
        """
        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Reply with just the word 'test'"}
            ]
            
            response = self.call(messages, max_tokens=10)
            return response.success and "test" in response.content.lower()
            
        except Exception as e:
            logger.error(f"API key validation failed: {e}")
            return False
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get current usage statistics.
        
        Returns:
            Dictionary with usage metrics
        """
        return {
            'total_tokens': self.total_tokens_used,
            'total_cost_usd': self.total_cost_usd,
            'call_count': self.call_count,
            'avg_cost_per_call': self.total_cost_usd / max(1, self.call_count),
            'supported_models': list(self.PRICING.keys())
        }
    
    def reset_usage(self) -> None:
        """Reset usage tracking."""
        self.total_tokens_used = 0
        self.total_cost_usd = 0.0
        self.call_count = 0
        logger.info("LLM usage statistics reset")


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example of how to use the client (don't run without real API key)
    print("LLM Client implementation complete.")
    print("Usage example:")
    print("""
    client = LLMClient("your-api-key-here")
    
    # Estimate cost first
    cost = client.estimate_cost("Analyze this budget template")
    print(f"Estimated cost: ${cost:.4f}")
    
    # Make the call
    messages = [
        {"role": "system", "content": "You are a budget analysis expert."},
        {"role": "user", "content": "Analyze this budget template"}
    ]
    response = client.call(messages)
    
    if response.success:
        print(f"Response: {response.content}")
        print(f"Cost: ${response.cost_usd:.4f}")
    else:
        print(f"Error: {response.error_message}")
    """)

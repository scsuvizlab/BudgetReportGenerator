"""
Cost Guard

Monitors and enforces LLM usage costs to prevent budget overruns.
"""
import time
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CostRecord:
    """Record of a single LLM operation cost."""
    timestamp: float
    cost_usd: float
    tokens: int
    model: str
    operation: str
    session_id: str


class CostGuard:
    """
    Monitors and enforces LLM usage costs.
    
    Responsibilities:
    - Track cumulative costs
    - Enforce budget limits
    - Provide cost projections
    - Log cost history
    """
    
    def __init__(self, 
                 budget_limit_usd: float = 5.0,
                 warning_threshold: float = 0.8,
                 session_id: str = "default"):
        """
        Initialize cost guard.
        
        Args:
            budget_limit_usd: Maximum allowed spending
            warning_threshold: Fraction of budget at which to warn (0.0-1.0)
            session_id: Session identifier for tracking
        """
        self.budget_limit_usd = budget_limit_usd
        self.warning_threshold = warning_threshold
        self.session_id = session_id
        
        self.accumulated_cost = 0.0
        self.cost_history: List[CostRecord] = []
        self.warnings_issued = 0
        self.is_budget_exceeded = False
    
    def check_affordability(self, estimated_cost: float) -> bool:
        """
        Check if an operation can be afforded within budget.
        
        Args:
            estimated_cost: Estimated cost of the operation
            
        Returns:
            True if operation is affordable, False otherwise
        """
        projected_total = self.accumulated_cost + estimated_cost
        
        if projected_total > self.budget_limit_usd:
            logger.warning(f"Operation would exceed budget: ${projected_total:.4f} > ${self.budget_limit_usd:.4f}")
            return False
        
        # Check warning threshold
        warning_amount = self.budget_limit_usd * self.warning_threshold
        if projected_total > warning_amount and self.accumulated_cost <= warning_amount:
            self.warnings_issued += 1
            logger.warning(f"Approaching budget limit: ${projected_total:.4f} / ${self.budget_limit_usd:.4f}")
        
        return True
    
    def record_cost(self, 
                   cost_usd: float, 
                   tokens: int, 
                   model: str, 
                   operation: str = "llm_call") -> None:
        """
        Record actual cost of an operation.
        
        Args:
            cost_usd: Actual cost incurred
            tokens: Number of tokens used
            model: Model that was used
            operation: Type of operation performed
        """
        # Create cost record
        record = CostRecord(
            timestamp=time.time(),
            cost_usd=cost_usd,
            tokens=tokens,
            model=model,
            operation=operation,
            session_id=self.session_id
        )
        
        self.cost_history.append(record)
        self.accumulated_cost += cost_usd
        
        # Check if budget is now exceeded
        if self.accumulated_cost > self.budget_limit_usd:
            if not self.is_budget_exceeded:
                self.is_budget_exceeded = True
                logger.error(f"Budget exceeded! ${self.accumulated_cost:.4f} > ${self.budget_limit_usd:.4f}")
        
        logger.info(f"Cost recorded: ${cost_usd:.4f} for {operation} using {model}")
    
    def get_remaining_budget(self) -> float:
        """
        Get remaining budget amount.
        
        Returns:
            Remaining budget in USD
        """
        return max(0.0, self.budget_limit_usd - self.accumulated_cost)
    
    def get_budget_utilization(self) -> float:
        """
        Get budget utilization as a fraction.
        
        Returns:
            Fraction of budget used (0.0 to 1.0+)
        """
        if self.budget_limit_usd <= 0:
            return 0.0
        return self.accumulated_cost / self.budget_limit_usd
    
    def project_remaining_operations(self, avg_cost_per_operation: float) -> int:
        """
        Project how many more operations can be afforded.
        
        Args:
            avg_cost_per_operation: Average cost per operation
            
        Returns:
            Estimated number of remaining operations
        """
        if avg_cost_per_operation <= 0:
            return 0
        
        remaining_budget = self.get_remaining_budget()
        return int(remaining_budget / avg_cost_per_operation)
    
    def get_cost_breakdown(self) -> Dict[str, Any]:
        """
        Get detailed cost breakdown by operation and model.
        
        Returns:
            Dictionary with cost analysis
        """
        if not self.cost_history:
            return {
                'total_operations': 0,
                'total_cost': 0.0,
                'by_operation': {},
                'by_model': {},
                'recent_activity': []
            }
        
        # Analyze by operation type
        by_operation = {}
        for record in self.cost_history:
            if record.operation not in by_operation:
                by_operation[record.operation] = {'count': 0, 'cost': 0.0, 'tokens': 0}
            by_operation[record.operation]['count'] += 1
            by_operation[record.operation]['cost'] += record.cost_usd
            by_operation[record.operation]['tokens'] += record.tokens
        
        # Analyze by model
        by_model = {}
        for record in self.cost_history:
            if record.model not in by_model:
                by_model[record.model] = {'count': 0, 'cost': 0.0, 'tokens': 0}
            by_model[record.model]['count'] += 1
            by_model[record.model]['cost'] += record.cost_usd
            by_model[record.model]['tokens'] += record.tokens
        
        # Recent activity (last 10 operations)
        recent_activity = []
        for record in self.cost_history[-10:]:
            recent_activity.append({
                'timestamp': record.timestamp,
                'operation': record.operation,
                'model': record.model,
                'cost': record.cost_usd,
                'tokens': record.tokens
            })
        
        return {
            'total_operations': len(self.cost_history),
            'total_cost': self.accumulated_cost,
            'budget_limit': self.budget_limit_usd,
            'budget_remaining': self.get_remaining_budget(),
            'budget_utilization': self.get_budget_utilization(),
            'by_operation': by_operation,
            'by_model': by_model,
            'recent_activity': recent_activity,
            'warnings_issued': self.warnings_issued,
            'budget_exceeded': self.is_budget_exceeded
        }
    
    def update_budget_limit(self, new_limit: float) -> None:
        """
        Update the budget limit.
        
        Args:
            new_limit: New budget limit in USD
        """
        old_limit = self.budget_limit_usd
        self.budget_limit_usd = new_limit
        
        # Recheck budget status
        if self.accumulated_cost > new_limit:
            self.is_budget_exceeded = True
        else:
            self.is_budget_exceeded = False
        
        logger.info(f"Budget limit updated: ${old_limit:.2f} → ${new_limit:.2f}")
    
    def reset(self) -> None:
        """Reset all cost tracking."""
        old_cost = self.accumulated_cost
        old_count = len(self.cost_history)
        
        self.accumulated_cost = 0.0
        self.cost_history.clear()
        self.warnings_issued = 0
        self.is_budget_exceeded = False
        
        logger.info(f"Cost guard reset: was ${old_cost:.4f} with {old_count} operations")
    
    def export_history(self) -> List[Dict[str, Any]]:
        """
        Export cost history for external analysis.
        
        Returns:
            List of cost records as dictionaries
        """
        return [
            {
                'timestamp': record.timestamp,
                'cost_usd': record.cost_usd,
                'tokens': record.tokens,
                'model': record.model,
                'operation': record.operation,
                'session_id': record.session_id
            }
            for record in self.cost_history
        ]


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    guard = CostGuard(budget_limit_usd=1.0, warning_threshold=0.8)
    
    print("Cost Guard Example:")
    print(f"Budget limit: ${guard.budget_limit_usd:.2f}")
    print(f"Remaining: ${guard.get_remaining_budget():.2f}")
    
    # Simulate some operations
    print("\nSimulating operations:")
    
    # Check affordability
    estimated_cost = 0.25
    if guard.check_affordability(estimated_cost):
        guard.record_cost(0.25, 1000, "gpt-4o-mini", "template_analysis")
        print(f"Operation 1: ${0.25:.2f} - Approved")
    
    if guard.check_affordability(0.30):
        guard.record_cost(0.30, 1200, "gpt-4o-mini", "field_resolution")
        print(f"Operation 2: ${0.30:.2f} - Approved")
    
    # This should trigger warning
    if guard.check_affordability(0.50):
        guard.record_cost(0.50, 2000, "gpt-4o", "complex_analysis")
        print(f"Operation 3: ${0.50:.2f} - Approved (should warn)")
    
    # This should be rejected
    if guard.check_affordability(0.50):
        print(f"Operation 4: ${0.50:.2f} - Approved")
    else:
        print(f"Operation 4: ${0.50:.2f} - Rejected (would exceed budget)")
    
    # Show breakdown
    breakdown = guard.get_cost_breakdown()
    print(f"\nFinal state:")
    print(f"Total cost: ${breakdown['total_cost']:.4f}")
    print(f"Operations: {breakdown['total_operations']}")
    print(f"Budget utilization: {breakdown['budget_utilization']:.1%}")
    print(f"Warnings issued: {breakdown['warnings_issued']}")

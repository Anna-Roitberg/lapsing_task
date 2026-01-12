from dataclasses import dataclass
from typing import List, Optional, Dict

@dataclass
class CustomerContext:
    """
    Compact feature snapshot for the customer at risk.
    """
    policy_id: str
    month: str
    policy_age: int
    premium_amount: float
    payment_status: str # 'Paid', 'Late', 'Missed'
    customer_calls: int
    claim_count: int
    
    # Model output
    p_lapse: float  # Probability of lapse (0.0 to 1.0)
    
    # Derived segments could go here
    risk_tier: str # 'Low', 'Medium', 'High'

    def to_retrieval_query(self) -> str:
        """
        Builds a short textual query from the customer snapshot for retrieval.
        """
        parts = []
        
        # Payment Status context
        if self.payment_status == 'Late':
            parts.append("customer is past due")
            parts.append("payment late")
        elif self.payment_status == 'Missed':
            parts.append("customer missed payment")
            parts.append("failed payment attempts")
        else:
            parts.append("payments up to date")

        # Renewal context (heuristic based on policy age)
        # Assuming annual policies, renewal at 12, 24, 36 months
        months_to_renewal = 12 - (self.policy_age % 12)
        if months_to_renewal <= 2:
            parts.append(f"renewal in {months_to_renewal} months")
            parts.append("upcoming renewal intervention")
        
        # Engagement/Calls
        if self.customer_calls > 1:
            parts.append("high call volume")
            parts.append("customer calling in")
        elif self.customer_calls == 0:
            parts.append("low engagement")

        # Claims
        if self.claim_count > 0:
            parts.append("recent claim")
            parts.append("high claim volume")

        # Risk Score
        parts.append(f"predicted lapse probability {self.p_lapse:.2f}")
        
        # Explicit needs based on risk tier
        if self.p_lapse >= 0.75:
            parts.append("need immediate retention intervention")
            parts.append("high risk rescue")
        elif self.p_lapse >= 0.40:
            parts.append("need renewal or pricing intervention")
        
        return ", ".join(parts)

@dataclass
class RecommendedAction:
    """
    A single recommended action.
    """
    action_name: str
    reasoning: str # Why this fits this customer
    expected_impact: str # e.g. "Low", "High"
    cost_effort: str # e.g. "Low", "Medium"
    messaging_draft: Optional[str] = None

@dataclass
class StrategyOutput:
    """
    Structured strategy object returned by the generator.
    """
    policy_id: str
    top_actions: List[RecommendedAction]
    do_not_do_warnings: List[str]
    
    def to_dict(self):
        return {
            'policy_id': self.policy_id,
            'actions': [
                {
                    'action': a.action_name,
                    'reasoning': a.reasoning,
                    'impact': a.expected_impact,
                    'effort': a.cost_effort,
                    'message': a.messaging_draft
                } for a in self.top_actions
            ],
            'warnings': self.do_not_do_warnings
        }

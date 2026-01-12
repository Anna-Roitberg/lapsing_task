from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ConversionContext:
    """
    Context for generating a conversion/retention plan.
    """
    policy_id: str
    age: int
    region: str
    channel: str      # e.g. 'Email', 'Phone', 'In-Person'
    needs: str        # e.g. 'Family Protection', 'Budget'
    objections: str   # e.g. 'Price', 'Competitor'
    premium: float
    
    def to_retrieval_query(self) -> str:
        """
        Creates a search query focused on the customer's specific objection and needs.
        """
        parts = []
        
        # Primary focus on objection handling
        if self.objections:
            parts.append(f"handle objection {self.objections}")
            parts.append(f"customer says {self.objections}")
            
        # Secondary focus on needs matching
        if self.needs:
            parts.append(f"sell to {self.needs} need")
            
        # Channel specific advice
        if self.channel:
            parts.append(f"{self.channel} communication tips")
            
        return ", ".join(parts)

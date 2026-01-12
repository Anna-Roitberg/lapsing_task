from strategy_contract import CustomerContext
import json

class StrategyPromptBuilder:
    SYSTEM_PROMPT = """You are a Retention Strategist for an insurance company. 
Your goal is to generate a personalized retention strategy for a customer at risk of lapsing.

Rules:
1. Only use the provided Customer Context and retrieved Playbook Snippets.
2. If critical information is missing, state your assumptions clearly.
3. Your output must be Valid JSON.
4. Adhere to the following Risk Tier guidelines:
    - p_lapse >= 0.75 (Critical): Immediate escalation, friction removal, direct call.
    - 0.40 <= p_lapse < 0.75 (Watchlist): Targeted nudges, offer plan adjustment, explain price changes.
    - p_lapse < 0.40 (Stable): Light-touch retention, value reminders, avoid aggressive discounts.
"""

    USER_TEMPLATE = """
### Customer Context
- **Policy ID**: {policy_id}
- **p_lapse**: {p_lapse:.2f} (Risk Tier: {risk_tier})
- **Policy Age**: {policy_age} months
- **Premium**: ${premium_amount}
- **Payment Status**: {payment_status}
- **Calls to Support**: {customer_calls}
- **Recent Claims**: {claim_count}

### Retrieved Playbook Snippets
{rag_snippets}

### Instructions
Analyze the customer context and the playbook advice. 
Return a JSON object with the following structure:
{{
  "risk_tier": "Critical | Watchlist | Stable",
  "primary_driver": "Main reason for risk (e.g. Payment Friction, Price Shock)",
  "actions": [
    {{
      "action_name": "Name of the action",
      "reasoning": "Why this fits this customer",
      "expected_impact": "High | Medium | Low",
      "cost_effort": "High | Medium | Low"
    }}
  ],
  "message_templates": [
    {{
      "channel": "SMS | Email | Call Script",
      "content": "Draft of the message"
    }}
  ],
  "metrics_to_track": ["Metric 1", "Metric 2"],
  "assumptions": ["Assumption 1"]
}}
"""

    @staticmethod
    def determine_risk_tier(p_lapse):
        if p_lapse >= 0.75:
            return "Critical"
        elif p_lapse >= 0.40:
            return "Watchlist"
        else:
            return "Stable"

    @staticmethod
    def build_messages(context: CustomerContext, rag_snippets: list[dict]) -> list[dict]:
        """
        Constructs the messages for the LLM chat completion API.
        """
        risk_tier = StrategyPromptBuilder.determine_risk_tier(context.p_lapse)
        
        # Format snippets
        formatted_snippets = []
        for i, snippet in enumerate(rag_snippets, 1):
            formatted_snippets.append(f"Snippet {i} (Source: {snippet['source']}):\n{snippet['chunk']}")
        
        rag_text = "\n\n".join(formatted_snippets)
        
        user_content = StrategyPromptBuilder.USER_TEMPLATE.format(
            policy_id=context.policy_id,
            p_lapse=context.p_lapse,
            risk_tier=risk_tier,
            policy_age=context.policy_age,
            premium_amount=context.premium_amount,
            payment_status=context.payment_status,
            customer_calls=context.customer_calls,
            claim_count=context.claim_count,
            rag_snippets=rag_text
        )
        
        return [
            {"role": "system", "content": StrategyPromptBuilder.SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]

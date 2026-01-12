from conversion_contract import ConversionContext

class ConversionPromptBuilder:
    SYSTEM_PROMPT = """You are a Sales Conversion Expert for an insurance company.
Your goal is to create a personalized 3-step conversion plan to close a lead who is at risk of not buying or lapsing.

Rules:
1. Address the specific 'Objection' and 'Need' provided in the context.
2. Adapt the tone to the 'Channel' (e.g., Email = concise/subject line; Phone = conversational script).
3. Use the provided 'Playbook Snippets' to cite your advice using [Doc#] or [Source].
4. Output must be Valid JSON.
"""

    USER_TEMPLATE = """
### Customer Context
- **Policy/Lead ID**: {policy_id}
- **Demographics**: Age {age}, Region {region}
- **Channel**: {channel}
- **Identified Need**: {needs}
- **Stated Objection**: {objections}
- **Premium**: ${premium}

### Playbook Snippets
{rag_snippets}

### Instructions
Generate a JSON object with this structure:
{{
  "conversion_strategy_name": "Short title",
  "steps": [
    {{
      "step": 1,
      "action": "What to do",
      "script_or_content": "Draft message or talking point",
      "rationale": "Why this works (cite source if applicable)"
    }},
    {{
      "step": 2,
      "action": "...",
        ...
    }},
    {{
      "step": 3,
      "action": "...",
        ...
    }}
  ],
  "citations": ["List of sources used"]
}}
"""

    @staticmethod
    def build_messages(context: ConversionContext, rag_snippets: list[dict]) -> list[dict]:
        # Format snippets
        formatted_snippets = []
        for i, snippet in enumerate(rag_snippets, 1):
            formatted_snippets.append(f"Snippet {i} (Source: {snippet['source']}):\n{snippet['chunk']}")
        
        rag_text = "\n\n".join(formatted_snippets)
        
        user_content = ConversionPromptBuilder.USER_TEMPLATE.format(
            policy_id=context.policy_id,
            age=context.age,
            region=context.region,
            channel=context.channel,
            needs=context.needs,
            objections=context.objections,
            premium=context.premium,
            rag_snippets=rag_text
        )
        
        return [
            {"role": "system", "content": ConversionPromptBuilder.SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]

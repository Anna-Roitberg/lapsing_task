import pandas as pd
import json
import os
import random

# Import our components
from retrieval_system import MinimalRAG
from conversion_contract import ConversionContext
from conversion_prompt import ConversionPromptBuilder

def infer_context(row):
    """
    Simulates missing fields (Channel, Needs, Objections) based on available data.
    """
    age = int(row.get('age', 35))
    has_agent = int(row.get('has_agent', 0))
    premium = float(row.get('premium', 0))
    
    # 1. Infer Channel
    if has_agent:
        channel = 'Phone' # Agents call
    else:
        channel = 'Email' # Direct business usually email
        
    # 2. Infer Needs based on Age
    if age < 30:
        needs = "Budget Friendly Coverage"
    elif age < 50:
        needs = "Family Protection"
    else:
        needs = "Retirement Security"
        
    # 3. Infer Objection based on Premium and Randomness
    objection_pool = ['Competitor Offer', 'Not Interested', 'Trust Issues']
    if premium > 150:
        objection_pool = ['Price too high', 'Value for money'] + objection_pool
        
    objection = random.choice(objection_pool)
    
    return channel, needs, objection

def main():
    # Use the requested file
    leads_file = 'data/three_lead_profiles_small.csv'
    
    if not os.path.exists(leads_file):
        print(f"Error: {leads_file} not found.")
        return

    print("Initializing RAG System (Leads Knowledge Base)...")
    rag = MinimalRAG(docs_dir='rag_docs/leads')
    
    print(f"Loading leads from {leads_file}...")
    df = pd.read_csv(leads_file)
    
    print("\n" + "="*80)
    print("GENERATING CONVERSION PLANS")
    print("="*80)
    
    for _, row in df.iterrows():
        policy_id = row['policy_id']
        age = int(row.get('age', 30))
        region = row.get('region', 'Unknown')
        premium = float(row.get('premium', 0.0))
        
        # Infer the missing context
        channel, needs, objection = infer_context(row)
        
        context = ConversionContext(
            policy_id=policy_id,
            age=age,
            region=region,
            channel=channel,
            needs=needs,
            objections=objection,
            premium=premium
        )
        
        print(f"\nProcessing {policy_id} ({age}yo, {region})")
        print(f"  Context -> Channel: {channel} | Need: {needs} | Obj: {objection}")
        
        # Retrieve
        query = context.to_retrieval_query()
        print(f"  Query: \"{query}\"")
        results = rag.retrieve(query, k=2)
        
        # Build Prompt
        messages = ConversionPromptBuilder.build_messages(context, results)
        
        # Mock LLM Response (Static for demo, but dynamic fields)
        mock_plan = {
            "conversion_strategy_name": f"Overcoming {objection} for {needs}",
            "steps": [
                {
                    "step": 1,
                    "action": f"Acknowledge {objection}",
                    "script_or_content": f"I hear you on {objection}. Many of our clients in {region} felt the same initially.",
                    "rationale": "Empathy builds trust (Sales 101)."
                },
                {
                    "step": 2,
                    "action": f"Pivot to {needs}",
                    "script_or_content": f"However, considering your age ({age}), {needs} is critical...",
                    "rationale": f"Aligns with their life stage."
                },
                {
                    "step": 3,
                    "action": "Close with Value",
                    "script_or_content": "Let's review the benefits.",
                    "rationale": "Reinforce value proposition."
                }
            ],
            "citations": [r['chunk'] for r in results]
        }
        
        print("--- Generated 3-Step Plan ---")
        print(json.dumps(mock_plan, indent=2))
        print("-" * 40)

if __name__ == "__main__":
    main()

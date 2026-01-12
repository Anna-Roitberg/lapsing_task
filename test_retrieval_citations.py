from retrieval_system import MinimalRAG

def test_citations():
    rag = MinimalRAG()
    
    # Query relevant to "Grace Period" (Doc1 in lapse)
    query_1 = "customer missed payment needs extension"
    print(f"\nQuery: {query_1}")
    res_1 = rag.retrieve(query_1, k=1)
    for r in res_1:
        print(f"Source: {r['source']}")
        print(f"Chunk Preview: {r['chunk'][:100]}...")
        
    # Query relevant to "Loyalty" (Doc4 in lapse)
    query_2 = "long tenure customer discount"
    print(f"\nQuery: {query_2}")
    res_2 = rag.retrieve(query_2, k=1)
    for r in res_2:
        print(f"Source: {r['source']}")
        print(f"Chunk Preview: {r['chunk'][:100]}...")

if __name__ == "__main__":
    test_citations()

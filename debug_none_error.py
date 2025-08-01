#!/usr/bin/env python3
"""
Test to reproduce and debug the AttributeError issue.
"""

def test_evaluation_scenarios():
    """Test different evaluation_history scenarios that could cause the error."""
    print("🧪 Testing various evaluation_history scenarios...")
    
    # Scenario 1: Normal working case
    print("\n1️⃣ Testing normal working case:")
    good_history = [
        {
            'query': 'Test query',
            'evaluation': {
                'winner': 'A',
                'response_a_scores': {'overall': 8.5},
                'response_b_scores': {'overall': 7.2}
            }
        }
    ]
    
    try:
        valid_evaluations = []
        for eval in good_history:
            if eval is not None and isinstance(eval, dict) and 'evaluation' in eval:
                valid_evaluations.append(eval)
        
        rag_wins = sum(1 for eval in valid_evaluations 
                      if eval.get('evaluation', {}) is not None and 
                         eval.get('evaluation', {}).get('winner') == 'A')
        print(f"   ✅ Normal case works: {rag_wins} RAG wins")
    except Exception as e:
        print(f"   ❌ Normal case failed: {e}")
    
    # Scenario 2: History with None values
    print("\n2️⃣ Testing with None values:")
    none_history = [
        {
            'query': 'Test query 1',
            'evaluation': {
                'winner': 'A'
            }
        },
        None,  # This is handled
        {
            'query': 'Test query 2',
            'evaluation': {
                'winner': 'B'
            }
        }
    ]
    
    try:
        valid_evaluations = []
        for eval in none_history:
            if eval is not None and isinstance(eval, dict) and 'evaluation' in eval:
                valid_evaluations.append(eval)
        
        rag_wins = sum(1 for eval in valid_evaluations 
                      if eval.get('evaluation', {}) is not None and 
                         eval.get('evaluation', {}).get('winner') == 'A')
        print(f"   ✅ None values handled: {rag_wins} RAG wins from {len(valid_evaluations)} valid")
    except Exception as e:
        print(f"   ❌ None values case failed: {e}")
    
    # Scenario 3: History with evaluation = None
    print("\n3️⃣ Testing with evaluation = None:")
    eval_none_history = [
        {
            'query': 'Test query 1',
            'evaluation': None  # This could cause the error!
        },
        {
            'query': 'Test query 2',
            'evaluation': {
                'winner': 'A'
            }
        }
    ]
    
    try:
        valid_evaluations = []
        for eval in eval_none_history:
            if eval is not None and isinstance(eval, dict) and 'evaluation' in eval:
                valid_evaluations.append(eval)
        
        rag_wins = sum(1 for eval in valid_evaluations 
                      if eval.get('evaluation', {}) is not None and 
                         eval.get('evaluation', {}).get('winner') == 'A')
        print(f"   ✅ Eval=None handled: {rag_wins} RAG wins from {len(valid_evaluations)} valid")
    except Exception as e:
        print(f"   ❌ Eval=None case failed: {e}")
    
    # Scenario 4: The problematic case - evaluation exists but is None
    print("\n4️⃣ Testing problematic case - evaluation key exists but value is None:")
    problematic_history = [
        {
            'query': 'Test query 1',
            'evaluation': {
                'winner': 'A'
            }
        },
        {
            'query': 'Test query 2',
            'evaluation': None  # This passes the 'evaluation' in eval check but is None!
        }
    ]
    
    try:
        # This mimics the OLD logic that would fail
        rag_wins = sum(1 for eval in problematic_history 
                      if eval is not None and eval.get('evaluation', {}).get('winner') == 'A')
        print(f"   This should fail with old logic: {rag_wins}")
    except AttributeError as e:
        print(f"   ✅ Expected error with old logic: {e}")
    
    try:
        # This mimics the NEW logic that should work
        valid_evaluations = []
        for eval in problematic_history:
            if eval is not None and isinstance(eval, dict) and 'evaluation' in eval:
                # Additional check: ensure evaluation is not None
                if eval['evaluation'] is not None:
                    valid_evaluations.append(eval)
        
        rag_wins = sum(1 for eval in valid_evaluations 
                      if eval.get('evaluation', {}) is not None and 
                         eval.get('evaluation', {}).get('winner') == 'A')
        print(f"   ✅ Fixed logic works: {rag_wins} RAG wins from {len(valid_evaluations)} valid")
    except Exception as e:
        print(f"   ❌ Fixed logic still failed: {e}")

def main():
    """Main test function."""
    print("=" * 70)
    print("🚀 Debugging AttributeError: 'NoneType' object has no attribute 'get'")
    print("=" * 70)
    
    test_evaluation_scenarios()
    
    print("\n" + "=" * 70)
    print("💡 SOLUTION: Add additional check for eval['evaluation'] is not None")
    print("=" * 70)

if __name__ == "__main__":
    main()

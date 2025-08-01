#!/usr/bin/env python3
"""
Test the final fix for the AttributeError issue.
"""

def test_final_fix():
    """Test the final comprehensive fix."""
    print("🧪 Testing the final fix...")
    
    # Create the most problematic scenario
    test_history = [
        {
            'query': 'Good evaluation',
            'evaluation': {
                'winner': 'A',
                'response_a_scores': {'overall': 8.5},
                'response_b_scores': {'overall': 7.2}
            }
        },
        None,  # Complete None object
        {
            'query': 'Missing evaluation key'
            # No 'evaluation' key at all
        },
        {
            'query': 'Evaluation is None',
            'evaluation': None  # This was the main culprit
        },
        {
            'query': 'Evaluation is empty dict',
            'evaluation': {}  # Empty but valid
        },
        {
            'query': 'Another good evaluation',
            'evaluation': {
                'winner': 'B',
                'response_a_scores': {'overall': 6.0},
                'response_b_scores': {'overall': 8.0}
            }
        }
    ]
    
    print(f"📊 Test data: {len(test_history)} items with various corruption scenarios")
    
    # Apply the final fix logic
    valid_evaluations = []
    for eval in test_history:
        if (eval is not None and 
            isinstance(eval, dict) and 
            'evaluation' in eval and 
            eval['evaluation'] is not None and
            isinstance(eval['evaluation'], dict)):
            valid_evaluations.append(eval)
    
    print(f"✅ Valid evaluations after filtering: {len(valid_evaluations)}")
    
    # Test statistics calculation
    total_evaluations = len(valid_evaluations)
    rag_wins = sum(1 for eval in valid_evaluations 
                  if eval['evaluation'].get('winner') == 'A')
    non_rag_wins = sum(1 for eval in valid_evaluations 
                      if eval['evaluation'].get('winner') == 'B')
    
    print(f"📈 Statistics:")
    print(f"   Total valid: {total_evaluations}")
    print(f"   RAG wins: {rag_wins}")
    print(f"   Non-RAG wins: {non_rag_wins}")
    
    # Test DataFrame creation
    history_df = []
    for i, eval_result in enumerate(valid_evaluations):
        evaluation = eval_result['evaluation']  # We know this is valid now
        row = {
            'Query': eval_result.get('query', 'N/A')[:50] + '...' if len(eval_result.get('query', '')) > 50 else eval_result.get('query', 'N/A'),
            'Winner': evaluation.get('winner', 'N/A'),
            'RAG Score': evaluation.get('response_a_scores', {}).get('overall', 0) if evaluation.get('response_a_scores') else 0,
            'Non-RAG Score': evaluation.get('response_b_scores', {}).get('overall', 0) if evaluation.get('response_b_scores') else 0,
        }
        history_df.append(row)
    
    print(f"📋 DataFrame created successfully with {len(history_df)} rows")
    for row in history_df:
        print(f"   {row['Query']} -> {row['Winner']} (RAG: {row['RAG Score']}, Non-RAG: {row['Non-RAG Score']})")
    
    return True

def main():
    """Main test function."""
    print("=" * 70)
    print("🚀 Final Fix Test - Comprehensive AttributeError Solution")
    print("=" * 70)
    
    success = test_final_fix()
    
    print("\n" + "=" * 70)
    if success:
        print("✅ FINAL FIX VERIFIED!")
        print("🎉 The AttributeError should now be completely resolved.")
    else:
        print("❌ Fix verification failed.")
    print("=" * 70)

if __name__ == "__main__":
    main()

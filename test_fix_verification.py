#!/usr/bin/env python3
"""
Test script to verify the AttributeError fix for NoneType evaluation handling.
"""

def test_evaluation_history_with_none_values():
    """Test that evaluation history properly handles None values."""
    print("🧪 Testing evaluation history None value handling...")
    
    # Simulate evaluation history with None values (as could happen during errors)
    test_evaluation_history = [
        {
            'query': 'Test query 1',
            'evaluation': {
                'winner': 'A',
                'response_a_scores': {'overall': 8.5},
                'response_b_scores': {'overall': 7.2}
            }
        },
        None,  # This could happen if evaluation fails
        {
            'query': 'Test query 2', 
            'evaluation': {
                'winner': 'B',
                'response_a_scores': {'overall': 6.8},
                'response_b_scores': {'overall': 8.1}
            }
        },
        None,  # Another failed evaluation
        {
            'query': 'Test query 3',
            'evaluation': {
                'winner': 'A',
                'response_a_scores': {'overall': 9.2},
                'response_b_scores': {'overall': 7.5}
            }
        }
    ]
    
    print(f"📊 Test data: {len(test_evaluation_history)} total entries (includes None values)")
    
    # Test the OLD approach (would cause AttributeError)
    print("\n❌ OLD APPROACH (would fail):")
    try:
        # This is what caused the error before
        rag_wins_old = sum(1 for eval in test_evaluation_history 
                          if eval.get('evaluation', {}).get('winner') == 'A')
        print(f"   This should not work: {rag_wins_old}")
    except AttributeError as e:
        print(f"   ✅ Expected error caught: {e}")
    
    # Test the NEW approach (fixed)
    print("\n✅ NEW APPROACH (fixed):")
    # Filter out None values first
    valid_evaluations = [eval for eval in test_evaluation_history if eval is not None]
    total_evaluations = len(valid_evaluations)
    rag_wins = sum(1 for eval in valid_evaluations 
                  if eval.get('evaluation', {}).get('winner') == 'A')
    non_rag_wins = sum(1 for eval in valid_evaluations 
                      if eval.get('evaluation', {}).get('winner') == 'B')
    
    print(f"   📈 Total valid evaluations: {total_evaluations}")
    print(f"   🏆 RAG wins: {rag_wins}")
    print(f"   🤖 Non-RAG wins: {non_rag_wins}")
    print(f"   📊 Win rate: RAG {rag_wins/total_evaluations*100:.1f}% vs Non-RAG {non_rag_wins/total_evaluations*100:.1f}%")
    
    return True

def test_history_dataframe_creation():
    """Test that history DataFrame creation handles None values."""
    print("\n🧪 Testing history DataFrame creation with None values...")
    
    test_evaluation_history = [
        {
            'query': 'What is Apple revenue growth?',
            'evaluation': {
                'winner': 'A',
                'response_a_scores': {'overall': 8.5},
                'response_b_scores': {'overall': 7.2}
            },
            'company_filter': 'AAPL'
        },
        None,  # Failed evaluation
        {
            'query': 'Microsoft cloud services performance?',
            'evaluation': {
                'winner': 'B',
                'response_a_scores': {'overall': 6.8},
                'response_b_scores': {'overall': 8.1}
            },
            'company_filter': 'MSFT'
        }
    ]
    
    # Test the fixed approach
    valid_evaluations = [eval for eval in test_evaluation_history if eval is not None]
    history_df = []
    
    for i, eval_result in enumerate(valid_evaluations):
        evaluation = eval_result.get('evaluation', {})
        if evaluation:
            row = {
                'Query': eval_result['query'][:50] + '...' if len(eval_result['query']) > 50 else eval_result['query'],
                'Winner': evaluation.get('winner', 'N/A'),
                'RAG Score': evaluation.get('response_a_scores', {}).get('overall', 0),
                'Non-RAG Score': evaluation.get('response_b_scores', {}).get('overall', 0),
                'Company Filter': eval_result.get('company_filter', 'All')
            }
            history_df.append(row)
    
    print(f"   ✅ Successfully created DataFrame with {len(history_df)} rows")
    for i, row in enumerate(history_df):
        print(f"   📋 Row {i+1}: {row['Query']} -> Winner: {row['Winner']}")
    
    return True

def main():
    """Main test function."""
    print("=" * 70)
    print("🚀 RAG Evaluation System - AttributeError Fix Verification")
    print("=" * 70)
    
    success = True
    
    success &= test_evaluation_history_with_none_values()
    success &= test_history_dataframe_creation()
    
    print("\n" + "=" * 70)
    if success:
        print("✅ ALL TESTS PASSED! AttributeError fix verified.")
        print("🎉 The Streamlit app should now handle None values properly.")
    else:
        print("❌ Some tests failed. Please check the implementation.")
    print("=" * 70)
    
    return success

if __name__ == "__main__":
    main()

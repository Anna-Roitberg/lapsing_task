import train_model
import generate_strategy
import generate_conversion_plan
import time

def main():
    print("="*80)
    print("STARTING FULL WORKFLOW")
    print("="*80)
    
    # 1. Train Model
    print("\n[STEP 1] Training Model...")
    start_train = time.time()
    train_model.train_xgboost_optuna()
    print(f"Training completed in {time.time() - start_train:.2f}s")
    
    # 2. Generate Retention Strategy (Lapse/Turn-around)
    print("\n[STEP 2] Generating Retention Strategies (using new model)...")
    try:
        generate_strategy.main()
    except Exception as e:
        print(f"Error in Strategy Generation: {e}")
        
    # 3. Generate Conversion Plans (New Leads)
    print("\n[STEP 3] Generating Conversion Plans...")
    try:
        generate_conversion_plan.main()
    except Exception as e:
        print(f"Error in Conversion Planning: {e}")
        
    print("\n" + "="*80)
    print("WORKFLOW COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()

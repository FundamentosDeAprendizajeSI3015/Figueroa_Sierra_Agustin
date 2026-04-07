import pickle
import os
import sys

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    artifacts_path = os.path.join(base_dir, 'data', 'preprocessing_artifacts.pkl')
    
    if not os.path.exists(artifacts_path):
        print(f"Error: {artifacts_path} not found. Please run scripts/13-preprocess.py first.")
        return

    with open(artifacts_path, 'rb') as f:
        artifacts = pickle.load(f)
    
    classes = artifacts.get('target_classes', [])
    
    if not classes:
        print("Error: No target classes found in the artifacts file.")
        return

    print("=" * 40)
    print("WeatherChart: Genre Class Lookup")
    print("=" * 40)
    print(f"Total classes found: {len(classes)}")
    
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:]).strip().lower()
        
        if query.isdigit():
            idx = int(query)
            if 0 <= idx < len(classes):
                print(f"\n[Index {idx}] -> {classes[idx]}")
            else:
                print(f"\nIndex {idx} is out of range.")
        else:
            found = False
            for i, name in enumerate(classes):
                if query in name.lower():
                    print(f"[Index {i:2}] -> {name}")
                    found = True
            if not found:
                print(f"\nNo genre matches found for query: '{query}'")
    else:
        print("\nFull List of Classes:")
        for i, name in enumerate(classes):
            print(f"{i:2}: {name}")

if __name__ == "__main__":
    main()

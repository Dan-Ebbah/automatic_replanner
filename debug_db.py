from app.data.extract_data import get_place
import json

print("Testing MongoDB connection...")
result = get_place()
if result:
    print(f"\nSuccess! Got {len(result['days'])} days of data")
    print("\nFull data structure:")
    print(json.dumps(result, indent=2))
else:
    print("Failed! get_place() returned None")

from app.data.extract_data import get_place

print("Testing MongoDB connection...")
result = get_place()
if result:
    print(f"Success! Got {len(result['days'])} days of data")
    print(f"First day: {result['days'][0]}")
else:
    print("Failed! get_place() returned None")

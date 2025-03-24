import pandera as pa

print("Available attributes in pandera:")
print(dir(pa))

print("\nChecking specific data types:")
print("Integer type:", pa.Int)
print("String type:", pa.String)
print("Float type:", pa.Float)
try:
    print("Boolean type:", pa.Boolean)
except AttributeError:
    print("Boolean type not found")

try:
    print("Bool type:", pa.Bool)
except AttributeError:
    print("Bool type not found")

try:
    print("BOOL type:", pa.BOOL)
except AttributeError:
    print("BOOL type not found")

print("Date type:", pa.Date if hasattr(pa, "Date") else "Not found")
print("DateTime type:", pa.DateTime if hasattr(pa, "DateTime") else "Not found")
print("Decimal type:", pa.Decimal if hasattr(pa, "Decimal") else "Not found")

# Check schema validation for different DataFrame types
print("\nChecking schema validation support:")
print("Has validate method:", hasattr(pa.DataFrameSchema, "validate")) 
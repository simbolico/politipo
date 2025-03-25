import pydantic
from pydantic import BaseModel, Field

class Test(BaseModel):
    code: str = Field(pattern=r'^[A-Z]{3}\d{3}$')

print(f"Pydantic version: {pydantic.VERSION}")
print(f"Code field: {Test.model_fields['code']}")
print(f"Code field metadata: {Test.model_fields['code'].metadata}")
print(f"Code field metadata type: {type(Test.model_fields['code'].metadata)}")

if Test.model_fields['code'].metadata:
    for item in Test.model_fields['code'].metadata:
        print(f"Metadata item: {item}, type: {type(item)}")
        print(f"Item dict: {item.__dict__ if hasattr(item, '__dict__') else 'No dict'}")
        print(f"Item class: {item.__class__.__name__ if hasattr(item, '__class__') else 'No class'}")
        
        # Print all attributes
        for attr in dir(item):
            if not attr.startswith('__'):
                try:
                    value = getattr(item, attr)
                    print(f"  Attr {attr}: {value}")
                except:
                    print(f"  Attr {attr}: error getting value") 
from pydantic import BaseModel

class SalesData(BaseModel):
    store_id: int
    product_id: int
    product_name: str
    month: int
    day: int
    year: int
    weekday: int
    sales: float
    quantity: int
    discount: float
    region: str
    segment: str
    category: str
    shipping_mode: str
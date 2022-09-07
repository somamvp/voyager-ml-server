from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

fake_db = {
    "foo": {"id": "foo", "title": "Foo", "description": "There goes my hero"},
    "bar": {"id": "bar", "title": "Bar", "description": "The bartenders"},
}


class Item(BaseModel):
    id: str
    title: str
    description: Optional[str] = None


@app.get("/")
async def root():
    return {"msg": "Hello World"}


@app.get("/items/{item_id}", response_model=Item)
async def read_item(item_id: str):
    return fake_db.get(item_id, None)


@app.post("/items/", response_model=Item)
async def create_item(item: Item):
    if item.id in fake_db:
        raise HTTPException(status_code=400, detail="Item already exists")

    fake_db[item.id] = item
    return item
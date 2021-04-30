import sys

from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel
from fun import *

version = f"{sys.version_info.major}.{sys.version_info.minor}"

app = FastAPI()


class Item(BaseModel):
    name: str
    price: float
    is_offer: Optional[bool] = None
@app.get("/")
async def read_root():
    message = f"Hello world! 访问/docs查看接口使用. Using Python {version}"
    return {"message": message}

@app.post("/ner/")
def read_item( q: Optional[str] = None):
    wd=None
    if q!=None:
        wd=model.decode(q)
    return { "q": q,"data":wd[-1]}
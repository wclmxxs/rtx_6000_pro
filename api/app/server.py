from .business import router as business_router
from .main import app


app.include_router(business_router)

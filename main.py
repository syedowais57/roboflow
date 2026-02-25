import os
import shutil
import zipfile
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from roboflow import Roboflow

# =============================
# INITIALIZE
# =============================

load_dotenv()

API_KEY = os.getenv("ROBOFLOW_PRIVATE_API_KEY")
WORKSPACE = os.getenv("ROBOFLOW_WORKSPACE")
PROJECT = os.getenv("ROBOFLOW_PROJECT")

if not all([API_KEY, WORKSPACE, PROJECT]):
    raise RuntimeError("Missing required environment variables")

app = FastAPI(title="Roboflow Upload Service")

# =============================
# HEALTH CHECK
# =============================

@app.get("/health")
def health():
    return {"status": "ok"}

# =============================
# DATASET UPLOAD ENDPOINT
# =============================

@app.post("/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):

    if not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only ZIP files allowed")

    request_id = str(uuid.uuid4())
    temp_zip_path = f"temp_{request_id}.zip"
    extract_path = f"temp_dataset_{request_id}"

    try:
        # Save ZIP
        with open(temp_zip_path, "wb") as buffer:
            buffer.write(await file.read())

        # Extract ZIP
        os.makedirs(extract_path, exist_ok=True)
        with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

        # Validate data.yaml exists
        if not os.path.exists(os.path.join(extract_path, "data.yaml")):
            raise HTTPException(
                status_code=400,
                detail="data.yaml not found in dataset root"
            )

        # Connect to Roboflow
        rf = Roboflow(api_key=API_KEY)
        workspace = rf.workspace(WORKSPACE)

        # Upload dataset
        workspace.upload_dataset(
            dataset_path=extract_path,
            project_name=PROJECT,
            project_type="object-detection",
            num_workers=8
        )

        return JSONResponse({
            "status": "success",
            "message": "Dataset uploaded successfully",
            "request_id": request_id
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e),
                "request_id": request_id
            }
        )

    finally:
        # Cleanup
        if os.path.exists(temp_zip_path):
            os.remove(temp_zip_path)
        if os.path.exists(extract_path):
            shutil.rmtree(extract_path)
import os
import shutil
import zipfile
import uuid
import base64
import requests
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
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
MODEL_VERSION = os.getenv("ROBOFLOW_MODEL_VERSION", "1")

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
# DATASET UPLOAD ENDPOINT (SDK)
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
            num_workers=1
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

# =============================
# REST API UPLOAD ENDPOINT
# =============================

@app.post("/upload-dataset-rest")
async def upload_dataset_rest(file: UploadFile = File(...)):
    """
    Uploads a dataset via Roboflow REST API directly.
    """
    if not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only ZIP files allowed")

    request_id = str(uuid.uuid4())
    temp_zip_path = f"temp_rest_{request_id}.zip"
    extract_path = f"temp_dataset_rest_{request_id}"

    try:
        # Save ZIP
        with open(temp_zip_path, "wb") as buffer:
            buffer.write(await file.read())

        # Extract ZIP
        os.makedirs(extract_path, exist_ok=True)
        with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

        # Upload images to Roboflow via REST API
        upload_results = []
        image_extensions = (".jpg", ".jpeg", ".png", ".bmp")
        
        # We'll recursively find all images in the extracted folder
        for root, dirs, files in os.walk(extract_path):
            for filename in files:
                if filename.lower().endswith(image_extensions):
                    image_path = os.path.join(root, filename)
                    
                    with open(image_path, "rb") as image_file:
                        image_data = base64.b64encode(image_file.read()).decode("utf-8")
                    
                    # Construct REST API URL
                    upload_url = f"https://api.roboflow.com/dataset/{PROJECT}/upload"
                    
                    params = {
                        "api_key": API_KEY,
                        "name": filename,
                        "split": "train" # Default split
                    }
                    
                    response = requests.post(upload_url, params=params, data=image_data, headers={
                        "Content-Type": "application/x-www-form-urlencoded"
                    })
                    
                    upload_results.append({
                        "file": filename,
                        "status": response.status_code,
                        "response": response.json() if response.status_code == 200 else response.text
                    })

        return JSONResponse({
            "status": "success",
            "message": f"Processed {len(upload_results)} images via REST API",
            "request_id": request_id,
            "results": upload_results
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

# =============================
# PREDICTION ENDPOINT
# =============================

@app.post("/predict")
async def predict(
    image_url: str = Query(..., description="Publicly accessible URL of the image to analyze"),
    confidence: int = Query(40, description="Minimum confidence threshold (0-100). Default: 40"),
    overlap: int = Query(30, description="Maximum overlap threshold for NMS (0-100). Default: 30")
):
    """
    Gets predictions for an image URL using the hosted Roboflow model.
    """
    try:
        # Strip whitespace from URL
        image_url = image_url.strip()
        
        # Construct Inference API URL
        # Format: https://detect.roboflow.com/dataset/version
        inference_url = f"https://detect.roboflow.com/{PROJECT}/{MODEL_VERSION}"
        
        params = {
            "api_key": API_KEY,
            "image": image_url,
            "confidence": confidence,
            "overlap": overlap
        }
        
        response = requests.post(inference_url, params=params)
        
        if response.status_code != 200:
            return JSONResponse(
                status_code=response.status_code,
                content={
                    "status": "error",
                    "message": "Inference failed",
                    "roboflow_response": response.text
                }
            )
            
        return JSONResponse({
            "status": "success",
            "predictions": response.json()
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )
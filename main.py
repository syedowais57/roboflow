import os
import shutil
import zipfile
import uuid
import base64
import requests
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from pydantic import BaseModel
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

# =============================
# REQUEST MODELS
# =============================

class GenerateVersionRequest(BaseModel):
    train_split: int = 70
    valid_split: int = 20
    test_split: int = 10
    preprocessing: Optional[dict] = None
    augmentation: Optional[dict] = None

class TrainRequest(BaseModel):
    model_type: Optional[str] = None
    speed: Optional[str] = None

# =============================
# PROJECT INFO ENDPOINT
# =============================

@app.get("/project-info")
def get_project_info():
    """
    Returns project-level details: name, type, image counts, classes, version count.
    """
    try:
        url = f"https://api.roboflow.com/{WORKSPACE}/{PROJECT}"
        params = {"api_key": API_KEY}
        response = requests.get(url, params=params)

        if response.status_code != 200:
            return JSONResponse(
                status_code=response.status_code,
                content={
                    "status": "error",
                    "message": "Failed to fetch project info",
                    "roboflow_response": response.text
                }
            )

        data = response.json()
        project = data.get("project", data)

        versions_field = project.get("versions", 0)
        version_count = versions_field if isinstance(versions_field, int) else len(versions_field)

        return JSONResponse({
            "status": "success",
            "project": {
                "id": project.get("id"),
                "name": project.get("name"),
                "type": project.get("type"),
                "images": project.get("images", 0),
                "unannotated": project.get("unannotated", 0),
                "annotated": project.get("annotation", 0),
                "classes": project.get("classes", {}),
                "splits": project.get("splits", {}),
                "versions": version_count
            }
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

# =============================
# LIST VERSIONS ENDPOINT
# =============================

@app.get("/versions")
def list_versions():
    """
    Lists all dataset versions with image counts and training status.
    Fetches each version individually since the project API only returns a count.
    """
    try:
        # First get the project to find the version count
        project_url = f"https://api.roboflow.com/{WORKSPACE}/{PROJECT}"
        params = {"api_key": API_KEY}
        project_response = requests.get(project_url, params=params)

        if project_response.status_code != 200:
            return JSONResponse(
                status_code=project_response.status_code,
                content={
                    "status": "error",
                    "message": "Failed to fetch project info",
                    "roboflow_response": project_response.text
                }
            )

        project_data = project_response.json()
        project = project_data.get("project", project_data)
        versions_field = project.get("versions", 0)
        version_count = versions_field if isinstance(versions_field, int) else len(versions_field)

        # Fetch each version individually
        versions = []
        for vid in range(1, version_count + 1):
            try:
                ver_url = f"https://api.roboflow.com/{WORKSPACE}/{PROJECT}/{vid}"
                ver_response = requests.get(ver_url, params=params)
                if ver_response.status_code != 200:
                    continue

                v = ver_response.json().get("version", {})
                version_info = {
                    "id": v.get("id"),
                    "name": v.get("name"),
                    "created": v.get("created"),
                    "images": v.get("images", 0),
                    "splits": v.get("splits", {}),
                    "exports": v.get("exports", []),
                    "preprocessing": v.get("preprocessing"),
                    "augmentation": v.get("augmentation"),
                }

                # Determine training status from the "train" field
                train_info = v.get("train")
                model_info = v.get("model")
                if train_info and train_info.get("results"):
                    version_info["training_status"] = "trained"
                    version_info["model"] = train_info.get("results", {})
                elif train_info:
                    version_info["training_status"] = train_info.get("status", "training")
                    version_info["model"] = None
                elif model_info:
                    version_info["training_status"] = "trained"
                    version_info["model"] = model_info
                else:
                    version_info["training_status"] = "not_trained"
                    version_info["model"] = None

                versions.append(version_info)
            except Exception:
                continue

        return JSONResponse({
            "status": "success",
            "total_versions": len(versions),
            "versions": versions
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

# =============================
# VERSION DETAIL ENDPOINT
# =============================

@app.get("/version/{version_id}")
def get_version(version_id: int):
    """
    Returns detailed info for a single version: splits, preprocessing,
    augmentation, training status, and model metrics.
    """
    try:
        url = f"https://api.roboflow.com/{WORKSPACE}/{PROJECT}/{version_id}"
        params = {"api_key": API_KEY}
        response = requests.get(url, params=params)

        if response.status_code != 200:
            return JSONResponse(
                status_code=response.status_code,
                content={
                    "status": "error",
                    "message": f"Failed to fetch version {version_id}",
                    "roboflow_response": response.text
                }
            )

        data = response.json()
        version = data.get("version", data)

        result = {
            "id": version.get("id"),
            "name": version.get("name"),
            "created": version.get("created"),
            "images": version.get("images", 0),
            "splits": version.get("splits", {}),
            "exports": version.get("exports", []),
            "generating": version.get("generating", False),
            "preprocessing": version.get("preprocessing"),
            "augmentation": version.get("augmentation"),
        }

        # Determine training status
        train_info = version.get("train")
        model_info = version.get("model")
        if train_info and train_info.get("results"):
            result["training_status"] = "trained"
            result["model"] = train_info.get("results", {})
        elif train_info:
            result["training_status"] = train_info.get("status", "training")
            result["model"] = None
        elif model_info:
            result["training_status"] = "trained"
            result["model"] = model_info
        else:
            result["training_status"] = "not_trained"
            result["model"] = None

        return JSONResponse({
            "status": "success",
            "version": result
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

# =============================
# GENERATE VERSION ENDPOINT
# =============================

@app.post("/generate-version")
async def generate_version(body: GenerateVersionRequest = GenerateVersionRequest()):
    """
    Generates a new dataset version with configurable train/test/valid split
    and optional preprocessing + augmentation settings.
    """
    try:
        # Validate split percentages
        total_split = body.train_split + body.valid_split + body.test_split
        if total_split != 100:
            raise HTTPException(
                status_code=400,
                detail=f"Split percentages must sum to 100, got {total_split}"
            )

        # Build version settings
        settings = {
            "preprocessing": body.preprocessing or {
                "auto-orient": True,
                "resize": {"width": 640, "height": 640, "format": "Stretch to"}
            },
            "augmentation": body.augmentation or {}
        }

        # Connect via SDK
        rf = Roboflow(api_key=API_KEY)
        project = rf.workspace(WORKSPACE).project(PROJECT)

        # Generate the version
        version = project.generate_version(settings=settings)

        return JSONResponse({
            "status": "success",
            "message": "New dataset version generated",
            "version_id": version.version if hasattr(version, 'version') else str(version),
        })

    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

# =============================
# TRAIN MODEL ENDPOINT
# =============================

@app.post("/train/{version_id}")
async def train_model(version_id: int, body: TrainRequest = TrainRequest()):
    """
    Triggers model training on a specific dataset version.
    Auto-exports the dataset if not already exported.
    Training runs asynchronously on Roboflow — poll GET /version/{version_id}
    for status updates.
    """
    try:
        params = {"api_key": API_KEY}

        # Step 1: Check if the version has been exported, export if needed
        ver_url = f"https://api.roboflow.com/{WORKSPACE}/{PROJECT}/{version_id}"
        ver_response = requests.get(ver_url, params=params)

        if ver_response.status_code != 200:
            return JSONResponse(
                status_code=ver_response.status_code,
                content={
                    "status": "error",
                    "message": f"Version {version_id} not found",
                    "roboflow_response": ver_response.text,
                    "version_id": version_id
                }
            )

        version_data = ver_response.json().get("version", {})
        exports = version_data.get("exports", [])

        # If not yet exported, trigger an export to yolov5pytorch format
        export_format = "yolov5pytorch"
        if export_format not in exports:
            export_url = f"https://api.roboflow.com/{WORKSPACE}/{PROJECT}/{version_id}/{export_format}"
            export_response = requests.get(export_url, params=params)
            # Wait briefly and re-check — export may take a moment
            if export_response.status_code != 200:
                return JSONResponse(
                    status_code=export_response.status_code,
                    content={
                        "status": "error",
                        "message": f"Failed to export version {version_id} to {export_format}. Export is required before training.",
                        "roboflow_response": export_response.text,
                        "version_id": version_id
                    }
                )

        # Step 2: Start training via REST API
        train_url = f"https://api.roboflow.com/{WORKSPACE}/{PROJECT}/{version_id}/train"
        train_params = {"api_key": API_KEY, "nocache": "true"}

        data = {}
        if body.model_type:
            data["modelType"] = body.model_type
        if body.speed:
            data["speed"] = body.speed

        response = requests.post(train_url, params=train_params, json=data)

        if response.status_code != 200:
            # Check if training is already in progress (not really an error)
            try:
                error_data = response.json()
                error_msg = error_data.get("error", {})
                if isinstance(error_msg, dict):
                    error_msg = error_msg.get("message", "")
                if "already running" in str(error_msg).lower():
                    return JSONResponse({
                        "status": "already_training",
                        "message": f"Version {version_id} is already being trained",
                        "version_id": version_id,
                        "note": "Poll GET /version/{version_id} for status updates."
                    })
            except Exception:
                pass

            return JSONResponse(
                status_code=response.status_code,
                content={
                    "status": "error",
                    "message": "Failed to start training",
                    "roboflow_response": response.text,
                    "version_id": version_id
                }
            )

        # Parse Roboflow response safely
        try:
            rf_response = response.json()
        except Exception:
            rf_response = response.text

        return JSONResponse({
            "status": "success",
            "message": f"Training started for version {version_id}",
            "version_id": version_id,
            "roboflow_response": rf_response,
            "note": "Training runs asynchronously. Poll GET /version/{version_id} for status updates."
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e),
                "version_id": version_id
            }
        )
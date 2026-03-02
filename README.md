# Roboflow Upload & Training Service

A FastAPI server for managing Roboflow datasets, model training, and predictions.

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env with your Roboflow credentials

# Run locally
uvicorn main:app --reload --port 8000
```

### Environment Variables

| Variable | Required | Description |
|---|---|---|
| `ROBOFLOW_PRIVATE_API_KEY` | ✅ | Your Roboflow private API key |
| `ROBOFLOW_WORKSPACE` | ✅ | Workspace slug (e.g. `site-sync`) |
| `ROBOFLOW_PROJECT` | ✅ | Project slug (e.g. `entity-detection-ykdwv`) |
| `ROBOFLOW_MODEL_VERSION` | ❌ | Model version for predictions (default: `1`) |

---

## API Endpoints

### Health Check

#### `GET /health`
Simple liveness check.

**Response:**
```json
{ "status": "ok" }
```

---

### Dataset Upload

#### `POST /upload-dataset`
Upload a dataset ZIP via the Roboflow SDK.

- **Content-Type:** `multipart/form-data`
- **Body:** `file` — ZIP containing a dataset with `data.yaml` at the root

**Response:**
```json
{
  "status": "success",
  "message": "Dataset uploaded successfully",
  "request_id": "uuid"
}
```

#### `POST /upload-dataset-rest`
Upload images via the Roboflow REST API directly.

- **Content-Type:** `multipart/form-data`
- **Body:** `file` — ZIP containing images (`.jpg`, `.jpeg`, `.png`, `.bmp`)

**Response:**
```json
{
  "status": "success",
  "message": "Processed 10 images via REST API",
  "request_id": "uuid",
  "results": [{ "file": "image1.jpg", "status": 200, "response": {} }]
}
```

---

### Predictions

#### `POST /predict`
Get object-detection predictions for an image URL.

**Query Parameters:**

| Param | Required | Default | Description |
|---|---|---|---|
| `image_url` | ✅ | — | Publicly accessible image URL |
| `confidence` | ❌ | `40` | Min confidence threshold (0–100) |
| `overlap` | ❌ | `30` | Max overlap / NMS threshold (0–100) |

**Response:**
```json
{
  "status": "success",
  "predictions": { "...roboflow prediction data..." }
}
```

---

### Project Info

#### `GET /project-info`
Returns project-level details including image counts, classes, and version count.

**Response:**
```json
{
  "status": "success",
  "project": {
    "id": "site-sync/entity-detection-ykdwv",
    "name": "Entity detection",
    "type": "object-detection",
    "images": 20,
    "unannotated": 17,
    "annotated": 3,
    "classes": { "Window": "#deb887", "Door": "#FF8000" },
    "splits": { "train": 16, "valid": 2, "test": 2 },
    "versions": 7
  }
}
```

---

### Version Management

#### `GET /versions`
Lists all dataset versions with training status and image counts.

**Response:**
```json
{
  "status": "success",
  "total_versions": 7,
  "versions": [
    {
      "id": "site-sync/entity-detection-ykdwv/1",
      "name": "v1",
      "images": 20,
      "splits": { "train": 16, "valid": 2, "test": 2 },
      "exports": ["yolov5pytorch"],
      "training_status": "trained",
      "model": { "...metrics..." }
    }
  ]
}
```

#### `GET /version/{version_id}`
Returns detailed info for a single version. Use this to **poll training status**.

**Path Parameters:** `version_id` (integer)

**Response:**
```json
{
  "status": "success",
  "version": {
    "id": "site-sync/entity-detection-ykdwv/7",
    "name": "2026-02-27 9:01am",
    "images": 20,
    "splits": { "train": 16, "valid": 2, "test": 2 },
    "exports": ["yolov5pytorch"],
    "generating": false,
    "preprocessing": { "auto-orient": true, "resize": { "width": 640, "height": 640 } },
    "augmentation": {},
    "training_status": "running",
    "model": null
  }
}
```

**Training Status Values:**
| Value | Meaning |
|---|---|
| `not_trained` | No training started |
| `running` | Training in progress |
| `trained` | Training complete, model ready |

---

### Version Generation & Training

#### `POST /generate-version`
Creates a new dataset version with configurable train/test/valid split.

**Request Body (JSON):**
```json
{
  "train_split": 70,
  "valid_split": 20,
  "test_split": 10,
  "preprocessing": {
    "auto-orient": true,
    "resize": { "width": 640, "height": 640, "format": "Stretch to" }
  },
  "augmentation": {}
}
```

All fields are optional — defaults to 70/20/10 split with auto-orient and 640×640 resize.

**Response:**
```json
{
  "status": "success",
  "message": "New dataset version generated",
  "version_id": "8"
}
```

#### `POST /train/{version_id}`
Triggers model training on a specific version. Auto-exports the dataset if needed.

**Path Parameters:** `version_id` (integer)

**Request Body (JSON, optional):**
```json
{
  "model_type": "object-detection",
  "speed": "accurate"
}
```

| Param | Default | Options |
|---|---|---|
| `model_type` | auto (from project) | `object-detection`, `classification`, `instance-segmentation` |
| `speed` | Roboflow default | `fast`, `accurate` |

**Response:**
```json
{
  "status": "success",
  "message": "Training started for version 7",
  "version_id": 7,
  "note": "Training runs asynchronously. Poll GET /version/{version_id} for status updates."
}
```

> ⚠️ Training is **async** — poll `GET /version/{version_id}` every 15–30s for updates.

---

## Frontend Workflow

```
1. Upload dataset     →  POST /upload-dataset
2. Generate version   →  POST /generate-version (set split)
3. Trigger training   →  POST /train/{version_id}
4. Poll status        →  GET  /version/{version_id} (every 15-30s)
5. Get predictions    →  POST /predict
```

---

## Swagger Docs

Interactive API docs available at: `http://localhost:8000/docs`

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import tempfile
import shutil

from .table.table_pipeline import recognize_table
from .matrix.matrix_pipeline import recognize_matrix
from .content_detector import detect_content

app = FastAPI()

# Allow your React app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/convert")
async def convert_image(file: UploadFile = File(...)):
    print(">>> Received request")
    print(">>> Filename:", file.filename)

    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    print(">>> Saved temp file:", tmp_path)

    # Detect content (matrix/table)
    print(">>> Running detect_content()")
    orig, detections = detect_content(tmp_path)
    print(">>> Detections:", detections)

    if not detections:
        print(">>> No detections, returning error")
        return {"latex": "ERROR: No content recognized"}

    label = detections[0][0]
    print(">>> Content type detected:", label)

    # Dispatch
    if label == "table":
        print(">>> Running recognize_table()")
        latex = recognize_table(tmp_path)
    elif label == "matrix":
        print(">>> Running recognize_matrix()")
        latex = recognize_matrix(tmp_path)
    else:
        latex = "ERROR: Unknown content type"

    print(">>> DONE. Returning LaTeX.")
    return {"latex": latex}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
from fastapi import FastAPI
from os import remove as rm
from utils.infer import detect_segment
from transformers import set_seed
from utils.geoserver import retrieve_tiff
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# entry point for machine learning model API
@app.get("/v1/{model_id}")
async def v1(
        model_id: str, 
        labels: str, 
        threshold: float = 0.1, 
        detector_id: str = 'google/owlv2-base-patch16', 
        segmenter_id: str = 'facebook/sam-vit-base', 
        seed : int = 0,
        bbox: str = None,
        width: int = 0,
        height: int = 0,
        layer: str = None
    ):
    match model_id:
        case "segment":
            set_seed(seed)

            # retrieve the temporary image from geoserver
            img_path = retrieve_tiff(bbox, width, height, layer)

            # detect and segment the image
            predictions = detect_segment(
                image_path=img_path,
                labels=[labels],
                threshold=threshold,
                detector_id=detector_id,
                segmenter_id=segmenter_id,
                transform=True
            )

            # remove the temporary image
            rm(img_path)

            # return the predictions
            return {"predictions": predictions}
        
        case default:
            return {"message": "Invalid model_id"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001)

from os import remove as rm
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import set_seed

from utils.geoserver import retrieve_tiff
from utils.infer import detect_segment


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# entry point for machine learning model API
@app.get("/api/v1/{model_id}")
async def v1(
        model_id: str, 
        labels: str, 
        det_conf: float = 5.0, 
        clf_conf: float = 70.0, 
        detector_id: str = 'google/owlv2-base-patch16', 
        segmenter_id: str = 'YxZhang/evf-sam2', 
        seed : int = 0,
        bbox: str = None,
        width: int = 0,
        height: int = 0,
        layer: str = None
    ):
    try:
        match model_id:
            case "segment":

                # set the seed
                set_seed(seed)

                # retrieve the temporary image from geoserver
                img_path = retrieve_tiff(bbox, width, height, layer)

                # detect and segment the image
                predictions = detect_segment(
                    image_path=img_path,
                    labels=[[f'a centered satellite image of a {label.strip()}' for label in labels.split(',')]],
                    det_conf=(det_conf / 100),
                    clf_conf=(clf_conf / 100),
                    detector_id=detector_id,
                    segmenter_id=segmenter_id,
                    transform=True
                )

                # remove the temporary image
                rm(img_path)

                # return the predictions
                return {"predictions": predictions}

            case default:
                raise HTTPException(status_code=404, detail="Model not found")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001)

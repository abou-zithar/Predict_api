
from ultralytics import YOLO
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile,Form, APIRouter
from fastapi.params import Body
import uvicorn
import Predict
from pydantic import BaseModel
import os
from datetime import datetime


model_path = 'best(wrinkle-4).pt'


model = YOLO(model_path)

router = APIRouter()
app =FastAPI()


# so i Can send request with HtML
app.add_middleware(
CORSMiddleware,
allow_origins=["*"],
allow_credentials=True,
allow_methods=["*"]
,allow_headers=["*"]
)

class Data(BaseModel):
    age: str
    gender: str 
    Skin_type: str | None = None
    image:UploadFile | None =None
    


@app.get("/test")
def test_app():
    results = {
        "test": "done",
        "print":"hi"
    }
    return results

@app.post("/predict")
async def predict(Images: list[UploadFile] = File(...),Age: str = Body(...), Gender: str = Body(...), Skin_type: str = Body(...)):
    final_massages=[]
    # Here you can save the file or process it as needed
    # file_location = f"temp/{file.filename}"
    
   
    if len(Images) == 0 :
        final_massages = {
            "status" :200,
            "massage" : "Please send Images "
        }
    else :
        for image in Images:
            unique_filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{image.filename}"
            file_location = f"temp/{unique_filename}"
            # Ensure the 'temp' directory exists
            os.makedirs(os.path.dirname(file_location), exist_ok=True)
            try:
                with open(file_location, "wb+") as file_object:
                    act_image =image.file.read()
                    file_object.write(act_image)
                    

            except Exception:
                return {"message": "There was an error uploading the file"}
            finally:
                print("image")
                image.file.close()
            
            
            results= Predict.predict(file_location)
            if results:
                for path in results:
                    if os.path.exists(path):
                        print(f"Image successfully saved at {path}")
                    else:
                        print(f"Error: Image not found at {path}")
            final_massage ={
            "status":200,
            "massage":"Post Done Successfully ",
            "data":{
                
                "diagnosis":[
                    {
                        "diagnos":"Wrinkle",
                        "imageURL":results[0]
                    },
                    {
                        "diagnos":"Dark Circles",
                        "imageURL":results[1]
                    },
                    {
                        "diagnos":"Acne",
                        "imageURL":results[2]
                    }
                ]
            }
            }
            final_massages.append(final_massage)

    return final_massages


app.include_router(router)

# Run the application
if __name__ == "__main__":
    uvicorn.run(app, host="127.1.1.1", port=8000)
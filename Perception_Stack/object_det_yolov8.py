#Dependencies: ultralytics/yolov8, OpenCV
from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
from ultralytics.yolo.utils.plotting import Annotator
import cv2
import pyrealsense2 as rs
import numpy as np


class object_detection():
    def cam_initialize_(self):
        #Create a context object. This object owns the handles to all connected realsense devices
        self.pipeline = rs.pipeline()

        #Create object - used to get depth & color stream
        config = rs.config()

        #Enable color streams
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30) #color stream
        print("Color stream enabled")

        #Start streaming
        self.pipeline.start(config)
        print("Stream started")


    def NN_YOLO_initialize_(self):
        self.model = YOLO("yolov8n.pt") #Importing YOLOv8 model
        # self.model.to('cuda') #Using GPU


    def get_color_frame(self):
        frames = self.pipeline.wait_for_frames() #this call waits until a new coherent set of frames is available on a device
        color_frame = frames.get_color_frame()

        if not color_frame:
            return None
        
        color_image = np.asanyarray(color_frame.get_data())
        return color_image
    

    def run_model_(self, color_image): 
        results = self.model.predict(color_image) #Run YOLOv8 model using GPU and RGB cam stream as input
        # print(results)
        return results


    def draw_bounding_box(self, results, image_in_bgr):
        #Using Annotator class in YOLO to draw bounding box
        BB_coord = None
        for r in results:
            annotator = Annotator(image_in_bgr)
            
            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0] #get box coordinates in (top-left, bottom-right) format
                c = box.cls #get class of object detected - bottle is class no 39
                if int(c) == 39: #if bottle is detected
                    annotator.box_label(b, self.model.names[int(c)]) #draw bounding box
                    BB_coord = b #save bounding box coordinates
          
        image_in_bgr = annotator.result() #get image with bounding box drawn

        if BB_coord is not None:
            return image_in_bgr, BB_coord
        else:
            return image_in_bgr, None


    def calc_mid_obj(self, bounding_box_coord):
        #Find mid point of bounding box
        mid_x = (bounding_box_coord[0] + bounding_box_coord[2])/2
        mid_y = (bounding_box_coord[1] + bounding_box_coord[3])/2
        
        return (int(mid_x), int(mid_y))
    

def main():
    obj_det = object_detection()

    #Camera & YOLOv8 model initialization
    obj_det.cam_initialize_()
    obj_det.NN_YOLO_initialize_()

    while True:
        color_image = obj_det.get_color_frame() #Imgae in BGR format
        img = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB) #Convert to RGB format for YOLOv8 model

        results = obj_det.run_model_(img) #Run YOLOv8 model and get results
        img_with_BB, BB_coord = obj_det.draw_bounding_box(results, color_image) #Draw bounding box around detected object on original BGR image

        if BB_coord is not None and img_with_BB is not None:
            print("\nObject Detected")
            obj_px_coord = obj_det.calc_mid_obj(BB_coord) #Calculate pixel coordinates of center of detected object
            print("X pixel coord: ", obj_px_coord[0])
            print("Y pixel coord: ", obj_px_coord[1])

            cv2.circle(img_with_BB, (int(obj_px_coord[0]), int(obj_px_coord[1])), 5, (0, 0, 255), -1) #Draw circle at center of detected object
            cv2.imshow("Object Detection & BB", img_with_BB) #Show image with bounding box and center of detected object
            
        elif BB_coord is None:
            print("\nNo object detected")
            cv2.imshow("Object Detection & BB", img_with_BB) #Just show the RGB cam feed

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break   



if __name__ == "__main__":
    main()
#Dependencies: pyrealsense2, numpy, opencv-python
import pyrealsense2 as rs
import cv2
import numpy as np
from matplotlib import pyplot as plt
from object_det_yolov8 import object_detection
import math
from time import sleep


class stereo_intel_dum_e():
    def __init__(self):
        #Device Config - from RealSense Viewer:
        self.baseline = 50.011 #in cm
        self.focal_length = 379.983 #in cm
        self.principal_point_x = 320.101/10 #in cm
        self.principal_point_y = 241.742/10 #in cm

        #Create a context object. This object owns the handles to all connected realsense devices
        self.pipeline = rs.pipeline()

        #Create object - used to get depth & color stream
        config = rs.config()

        #Enable depth and color streams
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30) #depth stream
        print("Depth stream enabled")
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30) #color stream
        print("Color stream enabled")
        config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30) #IR stream
        print("IR Cam 1 stream enabled")
        config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30) #IR stream
        print("IR Cam 2 stream enabled")

        #Start streaming
        self.pipeline.start(config)
        print("Stream started")

        #Align depth cam feed with color cam feed
        #to match the FoV of the depth image to the color image
        align_to = rs.stream.color 
        self.align = rs.align(align_to)


    def get_depth_frame(self):
        frames = self.pipeline.wait_for_frames() #this call waits until a new coherent set of frames is available on a device
        frames = self.align.process(frames)
        depth_frame = frames.get_depth_frame()

        #Get depth cam intrinsic matrix & get focal length
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        focal_length_from_intrinsic = [depth_intrin.fx, depth_intrin.fy]

        if not depth_frame:
            return None
        
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET) #Gives color to depth map
        return depth_colormap, depth_image, focal_length_from_intrinsic
        

    def get_color_frame(self):
        frames = self.pipeline.wait_for_frames() #this call waits until a new coherent set of frames is available on a device
        frames = self.align.process(frames)
        color_frame = frames.get_color_frame()

        #Get color cam intrinsic matrix & get focal length
        color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
        color_focal_length_from_intrinsic = [color_intrin.fx, color_intrin.fy]

        if not color_frame:
            return None
        
        color_image = np.asanyarray(color_frame.get_data())
        return color_image, color_focal_length_from_intrinsic
    

    def get_IR_frame(self):
        frames = self.pipeline.wait_for_frames()
        frames = self.align.process(frames)
        IR_frame_1 = frames.get_infrared_frame(1)
        IR_frame_2 = frames.get_infrared_frame(2)
        

        if not IR_frame_1 or not IR_frame_2:
            return None, None
        
        IR_image_1 = np.asanyarray(IR_frame_1.get_data())
        IR_image_2 = np.asanyarray(IR_frame_2.get_data())

        return IR_image_1, IR_image_2
    

    def stop_stream(self):
        self.pipeline.stop()
        print("Stream stopped")

    def show_frame_cv(self, name, image1, image2):
        images = np.hstack((image1, image2))
        cv2.imshow(name, images)
        cv2.waitKey(1)

    def show_frame_plt(self, name, image):
        plt.figure(name)
        plt.imshow(image, cmap='gray') #cmap='gray' for grayscale image
        plt.pause(0.1)
        plt.draw()
        

    def FoV_calc(self, focal_length_from_intrinsic, color_focal_length_from_intrinsic):
        #Calc HFOV & VFOV using focal length from intrinsic matrix
        #Formula: HFOV = 2 * atan (W / (2 * fx))
        #         VFOV = 2 * atan (H / (2 * fy))
        print("Focal Length of Depth Cam (Intrinsic Matrix): ", focal_length_from_intrinsic)
        print("Focal Length of Color Cam (Intrinsic Matrix): ", color_focal_length_from_intrinsic)
        HFOV = math.degrees(2 * math.atan(640 / (2 * color_focal_length_from_intrinsic[0])))
        VFOV = math.degrees(2 * math.atan(480 / (2 * color_focal_length_from_intrinsic[1])))
        print("HFOV: ", HFOV)
        print("VFOV: ", VFOV)

        #Calc pixel density - deg/px
        px_den_H = HFOV / 640
        px_den_V = VFOV / 480

        return px_den_H, px_den_V


p1 = stereo_intel_dum_e()
obj_det = object_detection()
def main():
    obj_det.NN_YOLO_initialize_() #initialize YOLOv8 model

    res = True
    while res == True:
        ##### Stereo Camera Depth Estimation #####
        #Get frames & focal length from Depth, IR & Color cam:
        depth_frame_color_map, depth_frame_orig, focal_length_from_intrinsic = p1.get_depth_frame()
        color_frame, color_focal_length_from_intrinsic = p1.get_color_frame()
        IR_frame_1, IR_frame_2 = p1.get_IR_frame()

        if depth_frame_color_map is None or color_frame is None or IR_frame_1 is None or IR_frame_2 is None:
            continue

        # p1.show_frame_cv('Depth & Color Feed', depth_frame_color_map, color_frame)
        p1.show_frame_cv('Stereo IR Cam Feed', IR_frame_1, IR_frame_2)

        #Compute disparity map:
        # stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
        stereo = cv2.StereoBM_create()
        disparity = stereo.compute(IR_frame_1,IR_frame_2)
        p1.show_frame_plt('Disparity Map', disparity)

        # Compute depth map:
        depth = (p1.baseline * p1.focal_length) / disparity
        # p1.show_frame_plt('Depth Map', depth)

        #Calculate pixel density - deg/px
        px_den_H, px_den_V = p1.FoV_calc(focal_length_from_intrinsic, color_focal_length_from_intrinsic)
        


        ##### YOLOv8 Obj Detection & 3D coordinate Calc ######
        img = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB) #Convert to RGB format for YOLOv8 model
        results = obj_det.run_model_(img) #Run YOLOv8 model and get results
        img_with_BB, BB_coord = obj_det.draw_bounding_box(results, color_frame) #Draw bounding box around detected object on original BGR image

        if BB_coord is not None and img_with_BB is not None:
            print("\nObject Detected")
            obj_px_coord = obj_det.calc_mid_obj(BB_coord) #Calculate pixel coordinates of center of detected object
            print("X pixel coord: ", obj_px_coord[0])
            print("Y pixel coord: ", obj_px_coord[1])

            cv2.circle(img_with_BB, (int(obj_px_coord[0]), int(obj_px_coord[1])), 5, (0, 0, 255), -1) #Draw circle at center of detected object
            cv2.imshow("Object Detection & BB", img_with_BB) #Show image with bounding box and center of detected object

        #Calc real world x,y,z coordinates of detected object w.r.t camera:
            #Convert the x & y pixel coordinate to mm:
            #Formula: tan(theta) = x pixel coord in mm / focal length in mm
            #         theta_x = (x pixel coord - W/2) * deg/px
            #         theta_y = (y pixel coord - H/2) * deg/px
            u1 = (obj_px_coord[0]-320) * px_den_H
            deg_to_mm_u1 = math.tan(math.radians(u1)) * color_focal_length_from_intrinsic[0]
            v1 = (obj_px_coord[1]-240) * px_den_V
            deg_to_mm_v1 = math.tan(math.radians(v1)) * color_focal_length_from_intrinsic[1]

            #Calc x,y,z real world using depth frame from camera hw:
            #Formula: real world x = x pixel coord in mm * depth / focal length in mm
            #         real world y = y pixel coord in mm * depth / focal length in mm
            #         real world z = depth  

            # a) X, Y, Z Calc from depth frame from camera hw:
            x = (deg_to_mm_u1) * depth_frame_orig[obj_px_coord[1], obj_px_coord[0]] / color_focal_length_from_intrinsic[0]
            y = (deg_to_mm_v1) * depth_frame_orig[obj_px_coord[1], obj_px_coord[0]] / color_focal_length_from_intrinsic[1]
            coord_3D_obj_det_moveit = [x+50, y+80, depth_frame_orig[obj_px_coord[1], obj_px_coord[0]]+56]

            # b) X, Y, Z Calc from depth frame from openCV:
            # x = (deg_to_mm_u1) * depth[obj_px_coord[1], obj_px_coord[0]] / color_focal_length_from_intrinsic[0]
            # y = (deg_to_mm_v1) * depth[obj_px_coord[1], obj_px_coord[0]] / color_focal_length_from_intrinsic[1]
            # coord_3D_obj_det_moveit = [x, y, depth[obj_px_coord[1], obj_px_coord[0]]]
            
            print("\nObject 3D Coordinate MoveIT: ", coord_3D_obj_det_moveit)
            sleep(0.5)
        
    
        elif BB_coord is None:
            print("\nNo object detected")
            cv2.imshow("Object Detection & BB", img_with_BB) #Just show the RGB cam feed


        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            plt.close('all')
            p1.stop_stream()
            break   

            

    
if __name__ == "__main__":
    main()
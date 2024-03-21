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
        #Device Config:
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

        align_to = rs.stream.color
        self.align = rs.align(align_to)


    def get_depth_frame(self):
        frames = self.pipeline.wait_for_frames() #this call waits until a new coherent set of frames is available on a device
        frames = self.align.process(frames)
        depth_frame = frames.get_depth_frame()
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        focal_length_from_intrinsic = [depth_intrin.fx, depth_intrin.fy]

        
        if not depth_frame:
            return None
        
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        return depth_colormap, depth_image, focal_length_from_intrinsic
        

    def get_color_frame(self):
        frames = self.pipeline.wait_for_frames() #this call waits until a new coherent set of frames is available on a device
        frames = self.align.process(frames)
        color_frame = frames.get_color_frame()
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
        plt.imshow(image, cmap='gray')
        plt.pause(0.1)
        plt.draw()
        


p1 = stereo_intel_dum_e()
obj_det = object_detection()
def main():
    obj_det.NN_YOLO_initialize_()

    x_list = []
    y_list = []
    z_list = []

    res = True
    while res == True:
        depth_frame_color_map, depth_frame_orig, focal_length_from_intrinsic = p1.get_depth_frame()
        color_frame, color_focal_length_from_intrinsic = p1.get_color_frame()
        IR_frame_1, IR_frame_2 = p1.get_IR_frame()

        if depth_frame_color_map is None or color_frame is None or IR_frame_1 is None or IR_frame_2 is None:
            continue

        # p1.show_frame_cv('Depth & Color Feed', depth_frame_color_map, color_frame)
        p1.show_frame_cv('Stereo IR Cam Feed', IR_frame_1, IR_frame_2)

        # stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
        stereo = cv2.StereoBM_create()
        disparity = stereo.compute(IR_frame_1,IR_frame_2)

        depth = (p1.baseline * p1.focal_length) / disparity

        ###Trying to calc depth "pixel by pixel" from disparity map
        # for i in range(0, 480):
        #     for j in range(0, 640):
        #         depthh = p1.baseline * p1.focal_length / disparity[i][j]

        # print("Disparity: ", disparity)
        # print("Depth: ", depth)

        p1.show_frame_plt('Disparity Map', disparity)
        # p1.show_frame_plt('Depth Map', depth)

        print("Focal Length of Depth Cam (Intrinsic Matrix): ", focal_length_from_intrinsic)
        print("Focal Length of Color Cam (Intrinsic Matrix): ", color_focal_length_from_intrinsic)
        HFOV = math.degrees(2 * math.atan(640 / (2 * color_focal_length_from_intrinsic[0])))
        VFOV = math.degrees(2 * math.atan(480 / (2 * color_focal_length_from_intrinsic[1])))
        print("HFOV: ", HFOV)
        print("VFOV: ", VFOV)
        px_den_H = HFOV / 640
        px_den_V = VFOV / 480


        ###Displaying disparity map using cv2 requires normalization
        # disparity_n = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        # depth = (p1.baseline * p1.focal_length) / disparity_n
        # p1.show_frame_cv('Disparity & Depth Map', disparity_n, depth)

        ###Converting each pixel to mm and calc real world x & y coordinates
        # for i in range(0, 480):
        #     for j in range(0, 640):
        #         u1 = j * px_den_H
        #         deg_to_mm_u1 = math.tan(math.radians(u1)) * focal_length_from_intrinsic[0]
        #         v1 = i * px_den_V
        #         deg_to_mm_v1 = math.tan(math.radians(v1)) * focal_length_from_intrinsic[1]

        #         x = (deg_to_mm_u1 - p1.principal_point_x) * depth_frame_orig[i][j] / focal_length_from_intrinsic[0]
        #         y = (deg_to_mm_v1 - p1.principal_point_y) * depth_frame_orig[i][j] / focal_length_from_intrinsic[1]
        #         x_list.append(x)
        #         y_list.append(y)
        #         z_list.append(depth_frame_orig[i][j])

        ###Plotting 3D Scatter Plot of the real world coordinates to obtain 3D image of the scene
        # plt.figure('3D Plot')
        # ax = plt.axes(projection='3d')
        # ax.scatter3D(x_list, y_list, z_list, c=z_list, cmap='Greens')
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')
        # plt.pause(0.1)
        # plt.draw()
        
        print("\nCam Min depth: ", (depth_frame_orig[479,600]))
        print("Cam depth: ", depth_frame_orig.shape)

        print("\nOur Min depth: ", (depth[479,600]))
        print("Our Cam depth: ", depth.shape)

        ##### YOLOv8 Obj Detection #####
        img = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB) #Convert to RGB format for YOLOv8 model
        results = obj_det.run_model_(img) #Run YOLOv8 model and get results
        img_with_BB, BB_coord = obj_det.draw_bounding_box(results, color_frame) #Draw bounding box around detected object on original BGR image

        if BB_coord is not None and img_with_BB is not None:
            print("\nObject Detected")
            obj_px_coord = obj_det.calc_mid_obj(BB_coord)
            # print("\nObject Pixel Coordinates: ", obj_px_coord) 
            print("X pixel coord: ", obj_px_coord[0])
            print("Y pixel coord: ", obj_px_coord[1])

            cv2.circle(img_with_BB, (int(obj_px_coord[0]), int(obj_px_coord[1])), 5, (0, 0, 255), -1)
            cv2.imshow("Object Detection & BB", img_with_BB)
            u1 = (obj_px_coord[0]-320) * px_den_H
            deg_to_mm_u1 = math.tan(math.radians(u1)) * color_focal_length_from_intrinsic[0]
            v1 = (obj_px_coord[1]-240) * px_den_V
            deg_to_mm_v1 = math.tan(math.radians(v1)) * color_focal_length_from_intrinsic[1]

            x = (deg_to_mm_u1) * depth_frame_orig[obj_px_coord[1], obj_px_coord[0]] / color_focal_length_from_intrinsic[0]
            y = (deg_to_mm_v1) * depth_frame_orig[obj_px_coord[1], obj_px_coord[0]] / color_focal_length_from_intrinsic[1]
            coord_3D_obj_det_moveit = [x+50, y+80, depth_frame_orig[obj_px_coord[1], obj_px_coord[0]]+56]
            print("\nObject 3D Coordinate MoveIT: ", coord_3D_obj_det_moveit)
            sleep(0.5)
        
    
        elif BB_coord is None:
            print("\nNo object detected")
            cv2.imshow("Object Detection & BB", img_with_BB)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            plt.close('all')
            p1.stop_stream()
            break   

        # res = False
    
    # plt.show()
            

    


if __name__ == "__main__":
    main()
    p1.stop_stream()
    cv2.destroyAllWindows()
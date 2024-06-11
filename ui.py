import dearpygui.dearpygui as dpg
import cv2
import numpy as np
import ai


bottom_text_1 = "Bottom_Text_1"     # item name for dearpygui
bottom_text_2 = "Bottom_Text_2"     

class MAIN():
    def __init__(self):


        cap = cv2.VideoCapture(0)                   # video capture from opencv, Always has to happen first for first frame as initialization
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB )     # convert color from bgr format to rgb. pag hindi mag mumuka kang papa smurf
        frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)     
        frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        video_fps = cap.get(cv2.CAP_PROP_FPS)

        data = frame.ravel()  # flatten camera data to a 1 d stricture
        data = np.asfarray(data, dtype='f')  # change data type to 32bit floats
        texture_data = np.true_divide(data, 255.0)


        dpg.create_context()    # creating context. has to happen before viewing the ui
        with dpg.texture_registry():
            dpg.add_raw_texture(frame.shape[1], frame.shape[0], texture_data, tag="texture_tag", format=dpg.mvFormat_Float_rgb)
        with dpg.window(tag="Primary Window",label="EEWS-MONITOR"):
            dpg.add_text("AI feed",tag="Top_Text_1",show=True)

            dpg.add_image("texture_tag")

            dpg.add_text("INFO\nNo Detection",tag=bottom_text_1,show=True)
            dpg.add_text("LAST DETECTION\nNo Detection",tag=bottom_text_2,show=True)

        # ui ^^^^^^
            
        dpg.create_viewport(title='EEWS', always_on_top=False,decorated=True ,width=500, height=300,resizable=True,x_pos=0,y_pos=0) # in dpg the ui isnt made as windows, instead they are created as viewport
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("Primary Window", True)


        while dpg.is_dearpygui_running():       # this is the main loop
            dpg.render_dearpygui_frame()
            ret, frame = cap.read()                                     # here we run video capture in a loop
            if frame is not None:           
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB )
                ai_data= ai.pred(frame)                             # passing image frame to the ai, it will return 4 box point locations and the box class
            if ai_data:
                x1,y1,x2,y2,class_name =ai_data
                cv2.rectangle(frame, (x1, y1), (x2, y2), color=(255,0,0), thickness=2)                  # we draw onto the frame, ready for displaying it in the ui
                dpg.configure_item(bottom_text_1,default_value=f"INFO\nPosition : ({x1},{y1},{x2},{y2})\nClass Name : {class_name}")                # text
                dpg.configure_item(bottom_text_2,default_value=f"LAST DETECTION\nPosition : ({x1},{y1},{x2},{y2})\nClass Name : {class_name}")
            else:
                dpg.configure_item(bottom_text_1,default_value="INFO\nNo Detection")
            data = frame.ravel()  # flatten camera data to a 1 d stricture
            data = np.asfarray(data, dtype='f')  # change data type to 32bit floats
            texture_data = np.true_divide(data, 255.0)
            dpg.set_value("texture_tag", texture_data)      # constantly updating the ui
            
        else:
            cap.release()
            dpg.destroy_context()
            

if __name__ == "__main__":
    MAIN()
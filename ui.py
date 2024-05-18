import dearpygui.dearpygui as dpg
import cv2
import numpy as np
import ai


bottom_text_1 = "Bottom_Text_1"
bottom_text_2 = "Bottom_Text_2"

class MAIN():
    def __init__(self):


        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB )
        frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        video_fps = cap.get(cv2.CAP_PROP_FPS)

        data = frame.ravel()  # flatten camera data to a 1 d stricture
        data = np.asfarray(data, dtype='f')  # change data type to 32bit floats
        texture_data = np.true_divide(data, 255.0)


        dpg.create_context()
        with dpg.texture_registry():
            dpg.add_raw_texture(frame.shape[1], frame.shape[0], texture_data, tag="texture_tag", format=dpg.mvFormat_Float_rgb)
        with dpg.window(tag="Primary Window",label="EEWS-MONITOR"):
            dpg.add_text("AI feed",tag="Top_Text_1",show=True)

            dpg.add_image("texture_tag")

            dpg.add_text("INFO\nNo Detection",tag=bottom_text_1,show=True)
            dpg.add_text("LAST DETECTION\nNo Detection",tag=bottom_text_2,show=True)
            
        dpg.create_viewport(title='EEWS', always_on_top=False,decorated=True ,width=500, height=300,resizable=True,x_pos=0,y_pos=0)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("Primary Window", True)


        while dpg.is_dearpygui_running():
            dpg.render_dearpygui_frame()
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB )
            ai_data= ai.pred(frame)
            if ai_data:
                x1,y1,x2,y2,class_name =ai_data
                cv2.rectangle(frame, (x1, y1), (x2, y2), color=(255,0,0), thickness=2)
                dpg.configure_item(bottom_text_1,default_value=f"INFO\nPosition : ({x1},{y1},{x2},{y2})\nClass Name : {class_name}")
                dpg.configure_item(bottom_text_2,default_value=f"LAST DETECTION\nPosition : ({x1},{y1},{x2},{y2})\nClass Name : {class_name}")
            else:
                dpg.configure_item(bottom_text_1,default_value="INFO\nNo Detection")
            data = frame.ravel()  # flatten camera data to a 1 d stricture
            data = np.asfarray(data, dtype='f')  # change data type to 32bit floats
            texture_data = np.true_divide(data, 255.0)
            dpg.set_value("texture_tag", texture_data)
            
        else:
            cap.release()
            dpg.destroy_context()
            

if __name__ == "__main__":
    MAIN()
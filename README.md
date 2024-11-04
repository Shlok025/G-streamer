
# YOLO Detection Pipeline Using GStreamer and OpenCV

## Description
This Python script implements a YOLO-based object detection pipeline using GStreamer for video processing and OpenCV's deep learning module for object detection. The script detects objects in a video stream and highlights them with bounding boxes and labels.

## Dependencies
To run this script, the following dependencies must be installed:
- **Python** (3.x)
- **GStreamer** (installed via MSYS2)
- **OpenCV** (with the DNN module)
- **Numpy**
- YOLO files: `yolov3.weights`, `yolov3.cfg`, and `coco.names`




## Installation
1. **GStreamer**: Make sure GStreamer is installed using MSYS2. You can use the following commands in your MSYS2 terminal:
   ```
   pacman -syu

   pacman -S mingw-w64-x86_64-gst-plugins-base mingw-w64-x86_64-gst-plugins-good mingw-w64-x86_64-gst-plugins-bad mingw-w64-x86_64-gst-plugins-ugly mingw-w64-x86_64-gst-python
   ```

2. **Python Libraries**: Install the required Python packages using pip:
   ```
   pip install opencv-python numpy
   ```

3. **YOLO Files**: Download the YOLOv3 configuration and weight files from [Darknet's YOLO page](https://pjreddie.com/darknet/yolo/) and place them in the same directory as the script.
   (Note:- i have not uploaded yolov3 weights kindly download from above link and place it within the directory itself)
    
## Usage
1. Place the video file (e.g., `853889-hd_1920_1080_25fps.mp4`) in the script directory or specify the correct path in `video_path`.
2. Run the script:
   ```
   python Detect.py
   ```
3. Press `q` to exit the detection window.


## Code

```python

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstVideo', '1.0')
from gi.repository import Gst, GstVideo, GLib
import cv2
import numpy as np
import os

class YOLODetectionPipeline:
    def __init__(self, video_source):
        # Initialize GStreamer
        Gst.init(None)
        print("GStreamer initialized")

        # Set display dimensions
        self.display_width = 1080
        self.display_height = 720

        # Create pipeline string
        pipeline_str = (
            f'filesrc location="{video_source}" ! '
            'decodebin ! '
            'videoconvert ! '
            'videoscale ! '
            f'video/x-raw,format=BGR,width={self.display_width},height={self.display_height} ! '
            'appsink name=sink emit-signals=true sync=false max-buffers=1 drop=true'
        )
        
        print(f"Pipeline string: {pipeline_str}")

        # Create pipeline
        self.pipeline = Gst.parse_launch(pipeline_str)
        if not self.pipeline:
            raise Exception("Failed to create pipeline")

        # Get appsink
        self.appsink = self.pipeline.get_by_name('sink')
        if not self.appsink:
            raise Exception("Failed to get appsink element")

        # Connect to appsink signals
        self.appsink.connect('new-sample', self.on_new_sample)

        # Initialize YOLO
        self.init_yolo()

    def init_yolo(self):
        # Load YOLO network
        self.net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

        # Load classes
        with open("coco.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        
        # Generate random colors for each class
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def on_new_sample(self, sink):
        sample = sink.emit("pull-sample")
        if sample:
            buffer = sample.get_buffer()
            caps = sample.get_caps()
            
            success, map_info = buffer.map(Gst.MapFlags.READ)
            if not success:
                return Gst.FlowReturn.ERROR

            # Get dimensions from caps
            caps_struct = caps.get_structure(0)
            width = caps_struct.get_value("width")
            height = caps_struct.get_value("height")

            # Create numpy array from buffer
            frame = np.ndarray(
                shape=(height, width, 3),
                dtype=np.uint8,
                buffer=map_info.data
            ).copy()

            buffer.unmap(map_info)

            # YOLO detection
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            self.net.setInput(blob)
            outs = self.net.forward(self.output_layers)

            # Detection information
            class_ids = []
            confidences = []
            boxes = []

            # Process detections
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        # Object detected
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            # Apply non-maximum suppression
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            
            # Draw boxes
            font = cv2.FONT_HERSHEY_PLAIN
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(self.classes[class_ids[i]])
                    color = self.colors[class_ids[i]]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, f"{label} {confidences[i]:.2f}", 
                               (x, y - 10), font, 1, color, 2)

            # Display frame
            cv2.imshow('YOLO Detection', frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                self.loop.quit()

            return Gst.FlowReturn.OK
        return Gst.FlowReturn.ERROR

    def run(self):
        # Create bus to get error messages
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect('message::error', self.on_error)
        bus.connect('message::eos', self.on_eos)

        # Start playing
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            raise Exception("Unable to set the pipeline to the playing state")

        print("Pipeline is playing")
        
        # Run the main loop
        self.loop = GLib.MainLoop()
        try:
            self.loop.run()
        except KeyboardInterrupt:
            pass
        finally:
            self.pipeline.set_state(Gst.State.NULL)
            cv2.destroyAllWindows()

    def on_error(self, bus, message):
        err, debug = message.parse_error()
        print(f"Error: {err.message}")
        print(f"Debug info: {debug}")
        self.loop.quit()

    def on_eos(self, bus, message):
        print("End of stream reached")
        self.loop.quit()

def main():
    video_path = "853889-hd_1920_1080_25fps.mp4"  # Replace with your video file
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return
    
    try:
        pipeline = YOLODetectionPipeline(video_path)
        pipeline.run()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

```



## Implementation
-To Implement using Msys2 terminal

![image](https://github.com/user-attachments/assets/d3a44029-11f8-4e1a-835f-aa763292b8ae)



## Demo

Demo using Msys2 terminal:-

![Human_Detect-ezgif com-speed](https://github.com/user-attachments/assets/ccfe48a2-0109-44c9-8ee8-6244ce0e74ea)
## Result

Image 1 :-
![Screenshot 2024-11-04 073642](https://github.com/user-attachments/assets/0f5a04dc-5ad9-4b52-ad1b-2ed444189ca2)

Image 2:-
![image](https://github.com/user-attachments/assets/c8c60f08-5eaf-4692-aef3-3e334b1c07f7)
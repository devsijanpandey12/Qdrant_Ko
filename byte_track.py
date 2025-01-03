# from ikomia.dataprocess.workflow import Workflow
# from ikomia.utils.displayIO import display
# import cv2


# # Replace 'your_video_path.mp4' with the actual video file path
# input_video_path = 0
# output_video_path = 'output_video.avi'

# # Init your workflow
# wf = Workflow()

# # Add object detection algorithm
# detector = wf.add_task(name="infer_yolo_v8", auto_connect=True)

# # Add ByteTrack tracking algorithm
# tracking = wf.add_task(name="infer_bytetrack", auto_connect=True)
# tracking.set_parameters({
#     "categories": "person"
# })

# # Open the video file
# stream = cv2.VideoCapture(input_video_path)
# if not stream.isOpened():
#     print("Error: Could not open video.")
#     exit()

# # Get video properties for the output
# frame_width = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
# frame_rate = stream.get(cv2.CAP_PROP_FPS)

# # Define the codec and create VideoWriter object
# # The 'XVID' codec is widely supported and provides good quality
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (frame_width, frame_height))

# while True:
#     # Read image from stream
#     ret, frame = stream.read()

#     # Test if the video has ended or there is an error
#     if not ret:
#         print("Info: End of video or error.")
#         break

#     # Run the workflow on current frame
#     wf.run_on(array=frame)

#     # Get results
#     image_out = tracking.get_output(0)
#     obj_detect_out = tracking.get_output(1)

#     # Convert the result to BGR color space for saving and displaying
#     img_res = cv2.cvtColor(image_out.get_image_with_graphics(obj_detect_out), cv2.COLOR_RGB2BGR)

#     # Save the resulting frame
#     out.write(img_res)

#     # Display
#     display(img_res, title="ByteTrack", viewer="opencv")

#     # Press 'q' to quit the video processing
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # After the loop release everything
# stream.release()
# out.release()
# cv2.destroyAllWindows()



from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display
import cv2

# Replace 'your_video_path.mp4' with the actual video file path
input_video_path = 0  # Use webcam
output_video_path = 'output_video.avi'

# Init your workflow
wf = Workflow()

# Add object detection algorithm
detector = wf.add_task(name="infer_yolo_v8", auto_connect=True)

# Add ByteTrack tracking algorithm
tracking = wf.add_task(name="infer_bytetrack", auto_connect=True)

# Set confidence score for tracking
tracking.set_parameters({
    "categories": "person",  # Track only persons
    "confidence": "0.8q"       # Set confidence score to 0.5
})

# Open the video file
stream = cv2.VideoCapture(input_video_path)
if not stream.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties for the output
frame_width = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_rate = stream.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object
# The 'XVID' codec is widely supported and provides good quality
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (frame_width, frame_height))

while True:
    # Read image from stream
    ret, frame = stream.read()

    # Test if the video has ended or there is an error
    if not ret:
        print("Info: End of video or error.")
        break

    # Run the workflow on current frame
    wf.run_on(array=frame)

    # Get results
    image_out = tracking.get_output(0)
    obj_detect_out = tracking.get_output(1)

    # Convert the result to BGR color space for saving and displaying
    img_res = cv2.cvtColor(image_out.get_image_with_graphics(obj_detect_out), cv2.COLOR_RGB2BGR)

    # Save the resulting frame
    out.write(img_res)

    # Display
    display(img_res, title="ByteTrack", viewer="opencv")

    # Press 'q' to quit the video processing
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release everything
stream.release()
out.release()
cv2.destroyAllWindows()

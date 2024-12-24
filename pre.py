import cv2
import face_recognition
import asyncio
import nest_asyncio
from qdrant_client import QdrantClient
from concurrent.futures import ThreadPoolExecutor, as_completed

nest_asyncio.apply()  # Allow asyncio in environments with a running event loop

# Connect to Qdrant instance
client = QdrantClient(host="localhost", port=6334, prefer_grpc=True)
collection_name = "embedding_collection1"

# Compare embeddings against the stored ones in Qdrant
def compare(embedding, top_k=1):
    threshold = 0.93
    search_results = client.search(
        collection_name=collection_name,
        query_vector=embedding,
        limit=top_k
    )
    if not search_results:
        return "Unknown"
    result = search_results[0]
    if result.score >= threshold:
        return result.payload["name"]
    else:
        return "Unknown"

# Predict face recognition for the current frame
async def predict(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    if not face_locations:
        return []
    
    face_encodings = face_recognition.face_encodings(rgb_frame, known_face_locations=face_locations)
    predictions = []

    with ThreadPoolExecutor() as executor:
        future_to_face = {
            executor.submit(compare, face_encoding): face_location
            for face_encoding, face_location in zip(face_encodings, face_locations)
        }
        for future in as_completed(future_to_face):
            result = future.result()
            location = future_to_face[future]
            predictions.append((result, location))
    
    return predictions

# Display predictions on the webcam feed
def show_predictions_on_frame(frame, predictions):
    for name, (top, right, bottom, left) in predictions:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)

# Process the webcam feed
async def process_video():
    video_capture = cv2.VideoCapture(0)
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        predictions = await predict(frame)
        show_predictions_on_frame(frame, predictions)
        cv2.imshow("Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(process_video())



# import cv2
# import face_recognition
# import asyncio
# import nest_asyncio
# from qdrant_client import QdrantClient
# from concurrent.futures import ThreadPoolExecutor, as_completed

# nest_asyncio.apply()  # Allow asyncio in environments with a running event loop

# # Connect to Qdrant instance
# client = QdrantClient(host="localhost", port=6334, prefer_grpc=True)
# collection_name = "embedding_collection1"

# # Compare embeddings against the stored ones in Qdrant
# def compare(embedding, top_k=1):
#     threshold = 0.93
#     search_results = client.search(
#         collection_name=collection_name,
#         query_vector=embedding,
#         limit=top_k
#     )
#     if not search_results:
#         return "Unknown"
#     result = search_results[0]
#     if result.score >= threshold:
#         return result.payload["name"]
#     else:
#         return "Unknown"

# # Process face detection and recognition in a separate thread
# def detect_and_recognize_faces(frame):
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     face_locations = face_recognition.face_locations(rgb_frame)
#     if not face_locations:
#         return []

#     face_encodings = face_recognition.face_encodings(rgb_frame, known_face_locations=face_locations)
#     results = [(compare(encoding), location) for encoding, location in zip(face_encodings, face_locations)]
#     return results

# # Display predictions on the webcam feed
# def show_predictions_on_frame(frame, predictions):
#     for name, (top, right, bottom, left) in predictions:
#         cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
#         cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
#         font = cv2.FONT_HERSHEY_DUPLEX
#         cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)

# # Process the webcam feed asynchronously
# async def process_video():
#     video_capture = cv2.VideoCapture(0)
#     frame_skip = 1  # Process every 5th frame
#     frame_count = 0

#     with ThreadPoolExecutor(max_workers=16) as executor:
#         loop = asyncio.get_event_loop()
#         while True:
#             ret, frame = video_capture.read()
#             if not ret:
#                 break

#             frame_count += 1
#             if frame_count % frame_skip == 0:
#                 # Run face detection and recognition in a thread
#                 predictions = await loop.run_in_executor(executor, detect_and_recognize_faces, frame)
#                 show_predictions_on_frame(frame, predictions)

#             # Display the frame
#             cv2.imshow("Webcam", frame)

#             # Exit on 'q' key press
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#     video_capture.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     try:
#         asyncio.run(process_video())
#     except KeyboardInterrupt:
#         print("\nExiting...")




# import av  # PyAV for video processing
# import face_recognition
# import torch
# import torchvision.transforms as T
# import asyncio
# from qdrant_client import QdrantClient
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from torchvision.utils import draw_bounding_boxes
# import numpy as np
# import cv2

# # Connect to Qdrant instance
# client = QdrantClient(host="localhost", port=6334, prefer_grpc=True)
# collection_name = "embedding_collection1"

# # Initialize transformation for the frames
# transform = T.ToTensor()

# # Compare embeddings against the stored ones in Qdrant
# def compare(embedding, top_k=1):
#     threshold = 0.93
#     search_results = client.search(
#         collection_name=collection_name,
#         query_vector=embedding.tolist(),
#         limit=top_k
#     )
#     if not search_results:
#         return "Unknown"
#     result = search_results[0]
#     if result.score >= threshold:
#         return result.payload["name"]
#     else:
#         return "Unknown"

# # Predict face recognition for the current frame
# async def predict(frame):
#     face_locations = face_recognition.face_locations(frame)
#     if not face_locations:
#         return []

#     face_encodings = face_recognition.face_encodings(frame, known_face_locations=face_locations)
#     predictions = []

#     with ThreadPoolExecutor() as executor:
#         future_to_face = {
#             executor.submit(compare, face_encoding): face_location
#             for face_encoding, face_location in zip(face_encodings, face_locations)
#         }
#         for future in as_completed(future_to_face):
#             result = future.result()
#             location = future_to_face[future]
#             predictions.append((result, location))
    
#     return predictions

# # Display predictions on the webcam feed
# def show_predictions_on_frame(frame, predictions):
#     # Convert the frame to a PyTorch tensor
#     frame_tensor = transform(frame).unsqueeze(0)
#     boxes = []
#     labels = []

#     for name, (top, right, bottom, left) in predictions:
#         boxes.append([left, top, right, bottom])
#         labels.append(name)

#     if boxes:
#         boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
#         frame_tensor = draw_bounding_boxes(
#             frame_tensor[0], boxes_tensor, labels=labels, colors="red", width=2
#         )
    
#     return frame_tensor.permute(1, 2, 0).numpy()

# # Process the webcam feed
# async def process_video():
#     # Open the webcam using a direct FFmpeg input string for PyAV
#     container = av.open('video=Integrated Webcam')  # Adjust device name for your system
#     stream = container.streams.video[0]
#     stream.thread_type = "AUTO"  # Optimize decoding

#     for packet in container.demux(stream):
#         for frame in packet.decode():
#             frame_rgb = frame.to_rgb().to_ndarray()  # Convert frame to RGB NumPy array
#             predictions = await predict(frame_rgb)
#             annotated_frame = show_predictions_on_frame(frame_rgb, predictions)

#             # Display the annotated frame using OpenCV
#             cv2.imshow("Webcam", annotated_frame[:, :, ::-1])  # Convert RGB to BGR for OpenCV
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     asyncio.run(process_video())






# tracking
# import cv2
# import face_recognition
# import asyncio
# import nest_asyncio
# from qdrant_client import QdrantClient
# from concurrent.futures import ThreadPoolExecutor, as_completed

# # Initialize Qdrant Client
# nest_asyncio.apply()
# client = QdrantClient(host="localhost", port=6334, prefer_grpc=True)
# collection_name = "embedding_collection1"

# # Initialize trackers and tracking state
# trackers = {}
# tracking_names = {}

# # Qdrant comparison
# def compare(embedding, top_k=1):
#     threshold = 0.93
#     search_results = client.search(
#         collection_name=collection_name,
#         query_vector=embedding,
#         limit=top_k
#     )
#     if not search_results:
#         return "Unknown"
#     result = search_results[0]
#     if result.score >= threshold:
#         return result.payload["name"]
#     else:
#         return "Unknown"

# # Predict face recognition
# async def predict(frame):
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     face_locations = face_recognition.face_locations(rgb_frame)
#     if not face_locations:
#         return []
    
#     face_encodings = face_recognition.face_encodings(rgb_frame, known_face_locations=face_locations)
#     predictions = []

#     with ThreadPoolExecutor() as executor:
#         future_to_face = {
#             executor.submit(compare, face_encoding): face_location
#             for face_encoding, face_location in zip(face_encodings, face_locations)
#         }
#         for future in as_completed(future_to_face):
#             result = future.result()
#             location = future_to_face[future]
#             predictions.append((result, location))
    
#     return predictions

# # Display predictions
# def show_predictions_on_frame(frame, predictions):
#     for name, (top, right, bottom, left) in predictions:
#         cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
#         cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
#         font = cv2.FONT_HERSHEY_DUPLEX
#         cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)

# # Start tracking a new face
# def start_tracking(frame, location, name):
#     (top, right, bottom, left) = location
#     tracker = None
#     try:
#         # Use CSRT if available
#         tracker = cv2.TrackerCSRT_create()
#     except AttributeError:
#         # Fallback to legacy MOSSE tracker
#         tracker = cv2.legacy.TrackerMOSSE_create()
#     tracker.init(frame, (left, top, right - left, bottom - top))
#     trackers[name] = tracker
#     tracking_names[name] = True  # Mark as currently being tracked

# # Update all trackers
# def update_trackers(frame):
#     global trackers
#     active_trackers = {}
#     for name, tracker in trackers.items():
#         success, bbox = tracker.update(frame)
#         if success:
#             left, top, width, height = map(int, bbox)
#             right = left + width
#             bottom = top + height
#             cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
#             cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
#             active_trackers[name] = tracker
#         else:
#             tracking_names.pop(name, None)  # Stop tracking if lost
#     trackers = active_trackers

# # Process video stream
# async def process_video():
#     video_capture = cv2.VideoCapture(0)
#     while True:
#         ret, frame = video_capture.read()
#         if not ret:
#             break
        
#         # Update trackers
#         update_trackers(frame)

#         # Detect new faces only if no active trackers
#         if not trackers:
#             predictions = await predict(frame)
#             for name, location in predictions:
#                 if name != "Unknown" and name not in tracking_names:
#                     start_tracking(frame, location, name)

#         # Show updated frame
#         cv2.imshow("Webcam", frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     video_capture.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     asyncio.run(process_video())






# import cv2
# import face_recognition
# import asyncio
# import nest_asyncio
# from qdrant_client import QdrantClient
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from yolox.tracker.byte_tracker import BYTETracker
# import numpy as np

# # Allow asyncio in environments with a running event loop
# nest_asyncio.apply()

# # Qdrant setup
# client = QdrantClient(host="localhost", port=6334, prefer_grpc=True)
# collection_name = "embedding_collection1"

# # Compare embeddings against the stored ones in Qdrant
# def compare(embedding, top_k=1):
#     threshold = 0.93
#     search_results = client.search(
#         collection_name=collection_name,
#         query_vector=embedding,
#         limit=top_k
#     )
#     if not search_results:
#         return "Unknown"
#     result = search_results[0]
#     if result.score >= threshold:
#         return result.payload["name"]
#     else:
#         return "Unknown"

# # Initialize ByteTrack
# tracker = BYTETracker(track_thresh=0.5, match_thresh=0.8, track_buffer=30)

# def convert_to_tracker_format(predictions):
#     """Converts predictions into ByteTrack format."""
#     tracked_objects = []
#     for name, (top, right, bottom, left) in predictions:
#         bbox = [left, top, right - left, bottom - top]  # Convert to [x, y, w, h]
#         confidence = 1.0  # Dummy confidence since face_recognition does not provide it
#         tracked_objects.append(np.array([*bbox, confidence, 0, 0]))  # Add dummy cls_id
#     return np.array(tracked_objects)

# # Predict and track faces
# async def predict_and_track(frame):
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     face_locations = face_recognition.face_locations(rgb_frame)
#     if not face_locations:
#         return []

#     face_encodings = face_recognition.face_encodings(rgb_frame, known_face_locations=face_locations)
#     predictions = []

#     with ThreadPoolExecutor() as executor:
#         future_to_face = {
#             executor.submit(compare, face_encoding): face_location
#             for face_encoding, face_location in zip(face_encodings, face_locations)
#         }
#         for future in as_completed(future_to_face):
#             result = future.result()
#             location = future_to_face[future]
#             predictions.append((result, location))

#     # Convert predictions to ByteTrack format and update tracker
#     tracked_objects = convert_to_tracker_format(predictions)
#     tracked_faces = tracker.update(tracked_objects, frame.shape[:2])

#     # Match tracked objects with predictions
#     tracked_predictions = []
#     for track in tracked_faces:
#         track_id = track.track_id
#         bbox = track.tlwh  # x, y, w, h
#         left, top, width, height = bbox
#         right, bottom = left + width, top + height
#         # Find the corresponding name from predictions
#         name = "Unknown"
#         for pred_name, (p_top, p_right, p_bottom, p_left) in predictions:
#             if abs(p_left - left) < 10 and abs(p_top - top) < 10:
#                 name = pred_name
#                 break
#         tracked_predictions.append((f"{name} (ID: {track_id})", (int(top), int(right), int(bottom), int(left))))

#     return tracked_predictions

# # Display predictions on the webcam feed
# def show_predictions_on_frame(frame, predictions):
#     for name, (top, right, bottom, left) in predictions:
#         cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)  # Green for tracked faces
#         cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
#         font = cv2.FONT_HERSHEY_DUPLEX
#         cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)

# # Process the webcam feed
# async def process_video():
#     video_capture = cv2.VideoCapture(0)
#     while True:
#         ret, frame = video_capture.read()
#         if not ret:
#             break
#         predictions = await predict_and_track(frame)
#         show_predictions_on_frame(frame, predictions)
#         cv2.imshow("Webcam", frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     video_capture.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     asyncio.run(process_video())





# import cv2
# import torch
# import asyncio
# import nest_asyncio
# from ultralytics import YOLO
# from torchvision.ops import nms
# import face_recognition
# from qdrant_client import QdrantClient
# from concurrent.futures import ThreadPoolExecutor, as_completed

# # Allow asyncio in environments with a running event loop
# nest_asyncio.apply()

# # Qdrant setup
# client = QdrantClient(host="localhost", port=6334, prefer_grpc=True)
# collection_name = "embedding_collection1"

# # NMS setup for bounding boxes
# def cpu_nms(boxes, scores, iou_threshold):
#     return nms(boxes.float().cpu(), scores.float().cpu(), iou_threshold).to(boxes.device)

# # Patch torchvision NMS
# import torchvision
# from torchvision.ops import nms as original_nms
# torchvision.ops.nms = cpu_nms

# # Paths
# model_path = r"C:\Users\sijan\Downloads\person-detection-master-20241130T130852Z-001\person-detection-master\person-detection-master\yolov8n.pt"
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# # Load YOLO model
# model = YOLO(model_path).to(device)

# # Embedding comparison function
# def compare(embedding, top_k=1):
#     threshold = 0.93
#     search_results = client.search(
#         collection_name=collection_name,
#         query_vector=embedding,
#         limit=top_k
#     )
#     if not search_results:
#         return "Unknown"
#     result = search_results[0]
#     if result.score >= threshold:
#         return result.payload["name"]
#     else:
#         return "Unknown"

# # Face recognition prediction
# async def predict_faces(frame):
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     face_locations = face_recognition.face_locations(rgb_frame)
#     if not face_locations:
#         return []

#     face_encodings = face_recognition.face_encodings(rgb_frame, known_face_locations=face_locations)
#     predictions = []

#     with ThreadPoolExecutor() as executor:
#         future_to_face = {
#             executor.submit(compare, face_encoding): face_location
#             for face_encoding, face_location in zip(face_encodings, face_locations)
#         }
#         for future in as_completed(future_to_face):
#             result = future.result()
#             location = future_to_face[future]
#             predictions.append((result, location))

#     return predictions

# # Display predictions on frame
# def annotate_frame(frame, predictions, detections):
#     for det in detections:
#         # Handle varying number of detection elements
#         if len(det) == 6:  # Full detection info
#             x1, y1, x2, y2, conf, track_id = map(int, det)
#             label = f"ID: {track_id}"
#         elif len(det) == 5:  # No track ID
#             x1, y1, x2, y2, conf = map(int, det)
#             label = f"Conf: {conf:.2f}"
#         elif len(det) == 4:  # Only bounding box
#             x1, y1, x2, y2 = map(int, det)
#             label = "Detection"
#         else:
#             continue

#         # Draw bounding box
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

#     for name, (top, right, bottom, left) in predictions:
#         cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
#         cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (255, 0, 0), cv2.FILLED)
#         font = cv2.FONT_HERSHEY_DUPLEX
#         cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)

# # Main webcam processing
# async def process_webcam():
#     cap = cv2.VideoCapture(0)
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

#     try:
#         results = model.track(
#             source=0,  # Webcam
#             persist=True,
#             stream=True,
#             conf=0.25,
#             task='track',
#             device=device
#         )

#         for result in results:
#             frame = result.orig_img
#             detections = result.boxes.xyxy.cpu().numpy() if result.boxes else []
#             faces = await predict_faces(frame)
#             annotate_frame(frame, faces, detections)
#             cv2.imshow("Webcam Tracking and Embedding Identification", frame)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#     finally:
#         cap.release()
#         cv2.destroyAllWindows()

# if __name__ == "__main__":
#     asyncio.run(process_webcam())

import cv2
from tkinter import *
from PIL import Image, ImageTk
from deepface import DeepFace
import os

class EmotionDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion Detection App")
        self.root.geometry("800x600")

        # Video capture
        self.cap = cv2.VideoCapture(0)

        # UI components
        self.video_frame = Label(root)
        self.video_frame.pack()

        self.capture_button = Button(root, text="Capture", command=self.capture_image, bg="blue", fg="white", font=("Arial", 12))
        self.capture_button.pack(pady=10)

        self.exit_button = Button(root, text="Exit", command=self.exit_app, bg="red", fg="white", font=("Arial", 12))
        self.exit_button.pack(pady=10)

        self.update_frame()

    def update_frame(self):
        # Get the latest frame
        ret, frame = self.cap.read()
        if ret:
            # Flip the frame horizontally (mirror effect)
            frame = cv2.flip(frame, 1)

            # Analyze emotion using DeepFace
            try:
                analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

                # Check if analysis is a list or dictionary
                if isinstance(analysis, list):
                    dominant_emotion = analysis[0].get('dominant_emotion', 'Unknown')
                elif isinstance(analysis, dict):
                    dominant_emotion = analysis.get('dominant_emotion', 'Unknown')

                # Display the emotion on the frame
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, f"Emotion: {dominant_emotion}", (50, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
            except Exception as e:
                print("Error in emotion detection:", e)

            # Convert frame to RGB for Tkinter display
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_frame.imgtk = imgtk
            self.video_frame.configure(image=imgtk)

        self.root.after(10, self.update_frame)

    def capture_image(self):
        ret, frame = self.cap.read()
        if ret:
            # Flip the frame horizontally (mirrored image)
            frame = cv2.flip(frame, 1)

            # Analyze the captured frame for emotions
            try:
                analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

                # Check if analysis is a list or dictionary
                if isinstance(analysis, list):
                    dominant_emotion = analysis[0].get('dominant_emotion', 'Unknown')
                elif isinstance(analysis, dict):
                    dominant_emotion = analysis.get('dominant_emotion', 'Unknown')

                print(f"Captured Emotion: {dominant_emotion}")

                # Save the captured image
                if not os.path.exists("captured_images"):
                    os.makedirs("captured_images")
                filename = f"captured_images/image_{dominant_emotion}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Image saved: {filename}")
            except Exception as e:
                print("Error in capturing image:", e)

    def exit_app(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.root.quit()

if __name__ == "__main__":
    root = Tk()
    app = EmotionDetectionApp(root)
    root.mainloop()

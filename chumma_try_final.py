import cv2
import pytesseract
import os

# Create the plates directory if it doesn't exist
if not os.path.exists("plates"):
    os.makedirs("plates")

harcascade = "/Number_plate_buppy/model/haarcascade_russian_plate_number (1).xml"
cap = cv2.VideoCapture(0)

cap.set(3, 640)  # width
cap.set(4, 480)  # height

min_area = 500
count = 0

while True:
    success, img = cap.read()

    plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)

    for (x, y, w, h) in plates:
        area = w * h

        if area > min_area:
            # Calculate the confidence as the ratio of area to min_area
            confidence = area / min_area
            if confidence > 0.75:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img, f"Number Plate (Confidence: {confidence:.2f})", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)

                img_roi = img[y: y + h, x:x + w]
                plate_text = pytesseract.image_to_string(img_roi, config='--psm 8')

                # Check if the recognized text meets certain criteria (e.g., alphanumeric characters and length)
                if confidence>0.75:
                    filename = os.path.join("plates", "scaned_img_" + str(count) + ".jpg")
                    cv2.imwrite(filename, img_roi)
                    cv2.rectangle(img, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, "Plate Saved", (150, 265), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
                    count += 1

    cv2.imshow("Result", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

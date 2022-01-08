import tensorflow as tf 
import numpy as np
import cv2

interpreter = tf.lite.Interpreter(model_path="detect.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(input_details)
print(output_details)

input_shape = input_details[0]['shape']
im = cv2.imread("buggy_image_2.jpg")  # 1920x1080

# RGB conversion
im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

# Crop center
crop_im = im[0:1080, 420:1500]  # 1080x1080

# Resize
im_rgb = cv2.resize(im_rgb, (input_shape[1], input_shape[2]))  # 300x300

# Set dimension
input_data = np.expand_dims(im_rgb, axis=0)
print(input_data.shape)
interpreter.set_tensor(input_details[0]['index'], input_data)

# Inference
interpreter.invoke()

detection_boxes = interpreter.get_tensor(output_details[0]['index'])
detection_classes = interpreter.get_tensor(output_details[1]['index'])
detection_scores = interpreter.get_tensor(output_details[2]['index'])
num_boxes = interpreter.get_tensor(output_details[3]['index'])

print("==detection_boxes==")
print(detection_boxes)
print()
print("==detection_classes==")
print(detection_classes)
print()
print("==detection_scores==")
print(detection_scores)
print()
print("==num_boxes==")
print(num_boxes)
print()

# Rendering
# img_width=1080
# img_height=1080
# for i in range(int(num_boxes[0])):
#   if detection_scores[0, i] > .5:
#        x = detection_boxes[0, i, [1, 3]] * img_width
#        y = detection_boxes[0, i, [0, 2]] * img_height
#        rectangle = [x[0], y[0], x[1], y[1]]
#        class_id = detection_classes[0, i]
#print(output_data.shape)
#print()
#print(output_data)

# Show image
# cv2.imshow("", crop_im)
# wait = True
# while wait:
#   wait = cv2.waitKey()=='q113' # hit q to exit

#output_data = interpreter.get_tensor(output_details[0]['index'])